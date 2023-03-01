// use std::sync::Arc;

use anyhow::Result;

use fips_md::{
    parser, runtime::{Axis, UniformMembers, Domain, RuntimeBuilder}};

fn main() -> Result<()> {
    // Include source code from file
    let source = include_str!("brownian.fips");
    // Parse unit
    let unit = parser::fips_parser::unit(source)?;

    // Define domain
    let domain = Domain::Dim3 {
        x: Axis {
            low: 0.0,
            high: 1000e-6,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        y: Axis {
            low: 0.0,
            high: 1000e-6,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        },
        z: Axis {
            low: -10.0e-6,
            high: 10.0e-6,
            oob: fips_md::runtime::OutOfBoundsBehavior::Periodic
        }
    };
    // Create runtime
    let mut runtime = RuntimeBuilder::new(unit, domain)
        .with_start_time(0.0) // This is actually the default value
        .with_time_step(1e-1)
        .build()?;
    // Define constants
    const VISCOSITY_WATER : f64 = 1e-3;
    const SIGMA: f64 = 20e-6 as f64; // 20 um diameter
    const OMEGA_11: f64 = 5. * SIGMA * SIGMA * SIGMA;
    const K_11: f64 = 12. * SIGMA;
    const K_22: f64 = 13. * SIGMA;
    runtime.define_constant_f64("CUTOFF", 1.0)?;
    runtime.define_constant_f64("K_B", 1.380649e-23)?;
    runtime.define_constant_f64("TEMP", 273.15 + 20.)?;
    runtime.define_constant_f64("GAMMA_X_INV", 1./VISCOSITY_WATER * 1./(K_11))?;
    runtime.define_constant_f64("GAMMA_Y_INV", 1./VISCOSITY_WATER * 1./(K_22))?;
    runtime.define_constant_f64("GAMMA_ROT_INV", 1./VISCOSITY_WATER * 1./(OMEGA_11))?;

    // Seed RNGs (one seed per thread, but we only have one thread here)
    runtime.seed_rngs(vec![
        5436576605279307334
    ]);

    #[allow(non_snake_case)]
    {
        // Create a single particle
        let uniforms = UniformMembers::new();
        let particles = runtime.create_particles("PointLike", 1, &uniforms, 1)?;
        // The 'double' definition of x and v might look strange, but we just need some binding
        // for the borrowed members to keep them alive long enough
        let mut x = particles.borrow_member_mut("x").unwrap(); let x = x.as_f64_slice()?;

        x[0] = 500e-6;
        x[1] = 500e-6;
        x[2] = 0.0;
    } // End of block => all references are dropped now
    // Print memory allocation
    println!("Allocated {} bytes of data!", runtime.get_memory_usage());
    // Compile runtime
    let compiled = runtime.compile("BrownianDynamics")?;

    // Run 10 steps
    for _ in 0..10 {
        compiled.run_step();
        {
            let particles = compiled.borrow_particle_mut("PointLike")?;
            let x = particles.borrow_member("x").unwrap(); let x = x.as_f64_slice()?;
            // Log particle position so we can observe the random walk
            dbg!(x);
        }
    }

    Ok(())
}