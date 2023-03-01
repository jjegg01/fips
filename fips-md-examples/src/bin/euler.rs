use std::sync::Arc;

use anyhow::Result;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_xoshiro::Xoshiro256PlusPlus;

use fips_md::{codegen::{CallbackStateType, GlobalContext}, parser, runtime::{Axis, UniformMembers, Domain, RuntimeBuilder, InteractionDetails}};

fn main() -> Result<()> {
    // Include source code from file
    let source = include_str!("euler.fips");
    // Parse unit
    let unit = parser::fips_parser::unit(source)?;
    // Define domain
    let domain = Domain::Dim3 {
        x: Axis {
            low: 0.0,
            high: 103.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        y: Axis {
            low: 0.0,
            high: 103.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        z: Axis {
            low: 0.0,
            high: 10.0,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        }
    };
    // Create runtime
    let mut runtime = RuntimeBuilder::new(unit, domain)
        .with_start_time(0.0) // This is actually the default value
        .with_time_step(1e-1)
        .build()?;
    // Seed RNG (TODO: make this unnecessary)
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
    // Define constants
    runtime.define_constant_f64("CUTOFF", 1.0)?;
    runtime.define_constant_f64("SIGMA", 1.0)?;
    runtime.define_constant_f64("EPSILON", 1.0)?;
    // Enable interactions
    runtime.enable_interaction("myinteraction", InteractionDetails {
        num_workers: 4,
        skin_factor: 1.2,
        cell_size: None,
        rebuild_interval: 1,
    }).unwrap();
    // Create particles (in block to limit lifetime of references)
    #[allow(non_snake_case)]
    {
        // Setup all uniform members (i.e. members that have a single value per particle *type*
        // and not per individual particle)
        let mut uniforms = UniformMembers::new();
        uniforms.set_uniform_member("mass", 2.0);
        // Create 100 particles with 2 worker threads
        let particles = runtime.create_particles("PointLike", 100, &uniforms, 2)?;
        // The 'double' definition of x and v might look strange, but we just need some binding
        // for the borrowed members to keep them alive long enough
        let mut x = particles.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice()?;
        let mut v = particles.borrow_member_mut("v").unwrap(); let v = v.as_f64_slice()?;
        let mut F = particles.borrow_member_mut("F").unwrap(); let F = F.as_f64_slice()?;
        // We can mutably borrow multiple members at once (this uses RefCell internally,
        // so borrow rules are checked at runtime)
        x[0] = 1.0;
        v[0] = 1.0;
        F[0] = 2.0;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(8581862258679931785);
        let dist_xy = rand::distributions::Uniform::new(0.0, 30.0);
        let dist_z = rand::distributions::Uniform::new(0.0, 10.0);
        for i in (0..x.len()).step_by(3) {
            x[i] = dist_xy.sample(&mut rng);
            x[i+1] = dist_xy.sample(&mut rng);
            x[i+2] = dist_z.sample(&mut rng);
        }
        // For verification: check if there are actually particles close enough to interact
        for i in (0..x.len()).step_by(3) {
            let (x1,y1,z1) = (x[i], x[i+1], x[i+2]);
            for j in (0..x.len()).step_by(3) {
                if i>= j {
                    continue;
                }
                let (x2,y2,z2) = (x[j], x[j+1], x[j+2]);
                let dist_sqr = (x2-x1).powi(2) + (y2-y1).powi(2) + (z2-z1).powi(2);
                if dist_sqr < (1.2*1.2) {
                    dbg!(i/3,j/3);
                    dbg!((x1,y1,z1));
                    dbg!((x2,y2,z2));
                    dbg!(dist_sqr);
                }
            }
        }
        x[0] = 99.9;
        v[0] = 5.0;
    } // End of block => all references are dropped now
    // Print memory allocation
    println!("Allocated {} bytes of data!", runtime.get_memory_usage());
    // Compile runtime
    let mut compiled = runtime.compile("MySim")?;

    // Define callbacks (must be non-capturing!)
    // Type inference is a bit flakey here, so we need to give an explicit type
    // to the second parameter
    struct MyCallbackData {
        times_called: usize
    }
    let mycallback = |_: &Arc<GlobalContext>, data: &mut CallbackStateType| {
        let data: &mut MyCallbackData = data.downcast_mut().unwrap();
        data.times_called += 1;
        println!("Called {} time(s) so far", data.times_called);
    };
    compiled.define_callback("mycallback", mycallback, Box::new(MyCallbackData{ times_called:0 }))?;

    // Run 10 simulation steps
    for _ in 0..10 {
        compiled.run_step();
    }

    // Check if the system has relaxed into a force-free state
    #[allow(non_snake_case)]
    {
        let particle_data = compiled.borrow_particle_mut("PointLike")?;
        let mut F = particle_data.borrow_member_mut("F").unwrap(); let F = F.as_f64_slice()?;
        for i in (0..F.len()).step_by(3) {
            if F[i] != 0.0 || F[i+1] != 0.0 || F[i+2] != 0.0 {
                dbg!(i/3);
                dbg!([F[i], F[i+1], F[i+2]]);
            }
        }
    }

    // Get callback results
    let (_, callback_state) = compiled.undefine_callback("mycallback")?;
    let callback_state: &MyCallbackData = callback_state.downcast_ref().unwrap();
    println!("Callback was called {} times total!", callback_state.times_called);
    compiled.join();

    Ok(())
}