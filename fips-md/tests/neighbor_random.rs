//! Test the neighbor list construction by randomizing the particle positions
//! and checking, if the neighbor lists update works properly

use std::ops::Range;

use fips_md::{codegen::CompiledRuntime, parser, runtime::*};
use rand::SeedableRng;
use rand_distr::Distribution;

const NUM_PARTICLES: usize = 1000;
const NUM_THREADS: usize = 4;
const POS_RNG_SEED: u64 = 12345678910;
const NUM_STEPS: usize = 10;

const CUTOFF: f64 = 1.0;
const SKIN_FACTOR: f64 = 1.2;
const DOMAIN_SIZE: f64 = 10.0;

const FIPSSRC : &'static str = r#"
particle PointLike {
    x : mut position
}

interaction test_interaction (p1: PointLike, p2: PointLike) for r < CUTOFF {}

simulation test_sim {
    default particle PointLike;
    step {
        update test_interaction;
        x = x + [random_normal(), random_normal(), random_normal()];
    }
}
"#;

#[test]
fn neighbor_random() {
    let unit = parser::fips_parser::unit(FIPSSRC).unwrap();

    let domain = Domain::Dim3 {
        x: Axis {
            low: 0.0,
            high: DOMAIN_SIZE,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        y: Axis {
            low: 0.0,
            high: DOMAIN_SIZE,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        z: Axis {
            low: 0.0,
            high: DOMAIN_SIZE,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        }
    };

    let mut runtime = RuntimeBuilder::new(unit, domain).build().unwrap();

    runtime.define_constant_f64("CUTOFF", CUTOFF).unwrap();
    runtime.seed_rngs(vec![
        5436576605279307334,
        10182137663237893456,
        3078663951407816208,
        1834164724811813848,
        9643917478817145632,
        17862601393704800001,
        10121462934265666987,
        11512985545539967636,
        13729390815792617066,
        3388980764283456606
    ]);

    runtime.enable_interaction("test_interaction", InteractionDetails {
        num_workers: NUM_THREADS,
        skin_factor: SKIN_FACTOR,
        cell_size: None,
        rebuild_interval: 1,
    }).unwrap();

    {
        let particles = runtime.create_particles("PointLike", NUM_PARTICLES, &UniformMembers::new(), NUM_THREADS).unwrap();
        let mut x = particles.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice().unwrap();
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(POS_RNG_SEED);
        let dist = rand_distr::Uniform::new(0.0, DOMAIN_SIZE);
        for position in x {
            *position = dist.sample(&mut rng);
        }
    }
    let mut compiled = runtime.compile("test_sim").unwrap();
    verify_neighbor_lists(&mut compiled);

    for _ in 0..NUM_STEPS {
        compiled.run_step();
        verify_neighbor_lists(&mut compiled);
    }
}

fn distance_sqr(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2)
        .map(|(x1,x2)| (x1-x2).abs())
        .map(|dx| if dx > DOMAIN_SIZE / 2.0 { dx - DOMAIN_SIZE } else { dx })
        .map(|dx| dx.powi(2))
        .sum()
}

fn verify_neighbor_lists(compiled: &mut CompiledRuntime) {
    compiled.rebuild_neighbor_list("test_interaction").unwrap();

    let mut neighbor_lists = compiled.get_neighbor_lists("test_interaction").unwrap();
    let index_ranges = neighbor_lists.keys().map(|x| x.clone()).collect::<Vec<_>>();

    let particles = compiled.borrow_particle_mut("PointLike").unwrap();
    let mut x = particles.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice().unwrap();
    for (i, p1) in x.chunks(3).enumerate() {
        for (j, p2) in x.chunks(3).enumerate() {
            // Check if particles should appear in neighbor lists
            if i != j && distance_sqr(&p1, &p2) < (CUTOFF * SKIN_FACTOR).powi(2) {
                let segment1 = index_ranges.iter().find(|(_, range)| range.contains(&i)).unwrap();
                let segment2 = index_ranges.iter().find(|(_, range)| range.contains(&j)).unwrap();
                // Common behavior for verification
                let mut verify_neighbors = |idx: usize, segment: &(ParticleID, Range<usize>), check: usize| {
                    let local_idx = idx - segment.1.start;
                    // Find section in neighbor list
                    let (neighbor_list_index, neighbor_list) = neighbor_lists.get_mut(&segment).unwrap();
                    let lower_offset = if local_idx == 0 { 0 } else { neighbor_list_index[local_idx - 1] };
                    let neighbor_slice = lower_offset..neighbor_list_index[local_idx];
                    let neighbors = &mut neighbor_list[neighbor_slice];
                    // Set entry to usize::MAX to mark that it was found
                    //println!("Positions: {:?}, {:?}", &x[3*idx..(3*idx+3)], &x[3*check..(3*check+3)]);
                    *neighbors.iter_mut().find(|x| **x == check)
                        .expect(&format!("Expected particle {} and {} to be neighbors", idx, check))
                        = usize::MAX;
                };
                // Easy case: both particles in the same neighbor list
                if segment1 == segment2 {
                    // Only i < j are stored, so skip the rest
                    if i > j { continue; }
                    verify_neighbors(i, segment1, j);
                }
                // Slightly more complicated case: both particles in different neighbor lists
                else {
                    verify_neighbors(i, segment1, j);
                }
            }
        }
    }

    for ((_, index_range), (neighbor_list_index, neighbor_list)) in neighbor_lists {
        for i in index_range.clone() {
            let local_idx = i - index_range.start;
            let lower_offset = if local_idx == 0 { 0 } else { neighbor_list_index[local_idx - 1] };
            let neighbor_slice = lower_offset..neighbor_list_index[local_idx];
            let neighbors = &neighbor_list[neighbor_slice];
            for entry in neighbors {
                if *entry != usize::MAX {
                    println!("Positions: {:?}, {:?}", &x[3*i..(3*i+3)], &x[3*entry..(3*entry+3)]);
                    panic!("Particles {} should not be in neighbor list of particles {}", entry, i);
                }
            }
        }
    }
}