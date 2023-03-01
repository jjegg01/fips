use anyhow::{Result, anyhow};
use ndarray::prelude::*;

use fips_md::{parser, runtime::{Axis, UniformMembers, Domain, RuntimeBuilder, InteractionDetails}};

const NUM_THREADS: usize = 2;

#[allow(non_snake_case)]
fn main() -> Result<()> {
    // Read output path from args
    let output_file = std::env::args().nth(1).ok_or(anyhow!("Usage: harmonic <output path>"))?;
    // Include source code from file
    let source = include_str!("harmonic.fips");
    // Parse unit
    let unit = parser::fips_parser::unit(source)?;
    // Define domain
    let domain = Domain::Dim3 {
        x: Axis {
            low: 0.0,
            high: 16.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        y: Axis {
            low: 0.0,
            high: 16.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        z: Axis {
            low: 0.0,
            high: 16.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        }
    };
    // Create runtime
    const DT: f64 = 1e-2;
    let mut runtime = RuntimeBuilder::new(unit, domain)
        .with_start_time(0.0) // This is actually the default value
        .with_time_step(DT)
        .build()?;
    // Define constants
    const K: f64 = 1.0;
    const M: f64 = 2.0;
    const L0: f64 = 1.25;
    runtime.define_constant_f64("CUTOFF", 3.0)?;
    runtime.define_constant_f64("K", K)?;
    runtime.define_constant_f64("L0", L0)?;
    // Seed RNGs
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
    // Enable interactions
    runtime.enable_interaction("spring", InteractionDetails {
        num_workers: NUM_THREADS,
        skin_factor: 1.2,
        cell_size: Some(4.0),
        rebuild_interval: 1,
    }).unwrap();
    // Create particles (in block to limit lifetime of references)
    #[allow(non_snake_case)]
    {
        let mut uniforms = UniformMembers::new();
        uniforms.set_uniform_member("m", M);
        // Create particles
        let particles = runtime.create_particles("PointLike", 2, &uniforms, NUM_THREADS)?;
        let mut x = particles.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice()?;
        // let mut v = particles.borrow_member_mut("v").unwrap(); let v = v.as_f64_slice()?;
        // let mut F = particles.borrow_member_mut("F").unwrap(); let F = F.as_f64_slice()?;

        x[0] = 7.0;
        x[1] = 8.0;
        x[2] = 8.0;
        x[3] = 9.0;
        x[4] = 8.0;
        x[5] = 8.0;
    }
    // Compile runtime
    let compiled = runtime.compile("VerletSim")?;

    const NUM_STEPS: usize = 1000;
    const SPEEDUP: usize = 2;
    // Allocate storage for the particles' trajectories
    let mut results_x : Array2<f64> = Array2::zeros((NUM_STEPS,2));
    let mut results_v : Array2<f64> = Array2::zeros((NUM_STEPS,2));
    let mut results_F : Array2<f64> = Array2::zeros((NUM_STEPS,2));
    let mut results_V : Array2<f64> = Array2::zeros((NUM_STEPS,2));
    let mut results_t : Array1<f64> = Array1::zeros((NUM_STEPS,));
    for i in 0..NUM_STEPS {
        for _ in 0..SPEEDUP {
            compiled.run_step();
        }
        {
            // Extract current state of each particle from the simulation
            let particle_data = compiled.borrow_particle_mut("PointLike")?;
            let mut x = particle_data.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice()?;
            let mut v = particle_data.borrow_member_mut("v").unwrap(); let v = v.as_f64_slice()?;
            let mut F = particle_data.borrow_member_mut("F").unwrap(); let F = F.as_f64_slice()?;
            let mut V = particle_data.borrow_member_mut("V").unwrap(); let V = V.as_f64_slice()?;

            results_t[i] = SPEEDUP as f64 * i as f64 * DT;
            results_x.row_mut(i).assign(&array![x[0], x[3]]);
            results_v.row_mut(i).assign(&array![v[0], v[3]]);
            results_F.row_mut(i).assign(&array![F[0], F[3]]);
            results_V.row_mut(i).assign(&array![V[0], V[1]]);

            //dbg!(0.5 * v.iter().map(|v| v*v).sum::<f64>() + V.iter().sum::<f64>());
        }
    }

    compiled.join();

    // ndarray_npy::write_npy(output_file, &results)
    //     .expect("Failed to write results.");
    let mut writer = ndarray_npy::NpzWriter::new(std::fs::File::create(output_file)?);
    writer.add_array("x", &results_x)?;
    writer.add_array("v", &results_v)?;
    writer.add_array("F", &results_F)?;
    writer.add_array("V", &results_V)?;
    writer.add_array("t", &results_t)?;
    writer.add_array("m", &array![M])?;
    writer.add_array("k", &array![K])?;
    writer.add_array("l0", &array![L0])?;

    Ok(())
}