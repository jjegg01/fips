//! All things related to running FIPS simulations

use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::codegen::{CompiledRuntime};

mod builder;
mod domain;
mod index;
mod interaction;
mod borrows;
mod particle_data;
mod types;

pub use builder::*;
pub use domain::*;
pub use index::*;
pub use interaction::*;
pub use borrows::*;
pub use particle_data::*;
pub use types::*;

use super::parser;

pub struct Runtime {
    /// Simulation domain
    pub(crate) domain: Domain,
    /// Current simulation time
    current_time: f64,
    /// Time step for simulation
    time_step: f64,
    /// RNG seeds
    pub(crate) rng_seeds: Option<Vec<u64>>,
    /// Store for particle data
    pub(crate) particle_store: ParticleStore,
    /// Index for particles
    pub(crate) particle_index: ParticleIndex,
    /// Possible simulations
    pub(crate) simulation_index: SimulationIndex,
    /// Global functions
    pub(crate) function_index: FunctionIndex,
    /// Defined interactions
    pub(crate) interaction_index: InteractionIndex,
    /// Enabled interactions
    pub(crate) enabled_interactions: HashMap<InteractionID, InteractionDetails>,
    /// Threads per particle type
    pub(crate) threads_per_particle: HashMap<ParticleID, usize>,
    /// Defined Constants
    /// (these are required for the symbol table during code generation)
    pub(crate) constants: HashMap<String, parser::SubstitutionValue>,
}

// Dimensionality of the particle system
pub(crate) const BUILTIN_CONST_NDIM: &str = "NDIM";
// Time step size
pub(crate) const BUILTIN_CONST_DT: &str = "DT";
// List of predefined constants
pub(crate) const BUILTIN_CONSTANTS: &[&str] = &[
    BUILTIN_CONST_NDIM,
    BUILTIN_CONST_DT
];

impl Runtime {
    pub fn new(unit: parser::Unit, domain: Domain, current_time: f64, time_step: f64) -> Result<Self> {
        let ndim = domain.get_dim();
        // Unpack parsed data from unit into runtime
        // (TODO: Allow more than one unit -> include concept)
        let particle_index = ParticleIndex::new(unit.particles)?;
        let simulation_index = SimulationIndex::new(unit.simulations, &particle_index)?;
        let interaction_index = InteractionIndex::new(unit.interactions, &particle_index)?;
        let function_index = FunctionIndex::new(unit.extern_functiondecls)?;
        let mut runtime = Self {
            domain,
            current_time, time_step,
            rng_seeds: None,
            particle_store: ParticleStore::new(),
            particle_index, simulation_index, interaction_index, function_index,
            enabled_interactions: HashMap::new(),
            threads_per_particle: HashMap::new(),
            constants: HashMap::new()
        };
        // Pre-defined constants
        runtime.define_constant(BUILTIN_CONST_NDIM.into(), parser::SubstitutionValue::Usize(ndim), true)?;
        // Return initialized runtime
        Ok(runtime)
    }

    pub fn create_particles<'a>(&'a mut self, particle_name: &str, count: usize,
        uniform_members: &UniformMembers, num_threads: usize) -> Result<ParticleBorrowMut<'a>>
    {
        // Find particle definition in index
        let particle = match self.particle_index.get_particle_by_name(particle_name) {
            None => return Err(anyhow!("Cannot find particle {}", particle_name)),
            Some(particle) => particle
        };
        // Create particle
        let particle_data = self.particle_store.create_particles(particle, count, &uniform_members)?;
        self.threads_per_particle.insert(particle.0, num_threads);
        // Return particle handle
        let (particle_id, particle_definition) = particle;
        Ok(ParticleBorrowMut::new(particle_id, particle_definition, particle_data))
    }

    /// Define a named constant of type i64
    pub fn define_constant_i64<S: Into<String>>(&mut self, name: S, value: i64) -> Result<()> {
        let name = name.into();
        let value = parser::SubstitutionValue::I64(value);
        self.define_constant(name, value, false)
    }

    /// Define a named constant of type f64
    pub fn define_constant_f64<S: Into<String>>(&mut self, name: S, value: f64) -> Result<()> {
        let name = name.into();
        let value = parser::SubstitutionValue::F64(value);
        self.define_constant(name, value, false)
    }

    /// Define a named constant of type usize
    pub fn define_constant_usize<S: Into<String>>(&mut self, name: S, value: usize) -> Result<()> {
        let name = name.into();
        let value = parser::SubstitutionValue::Usize(value);
        self.define_constant(name, value, false)
    }

    /// Internal function for defining constants
    fn define_constant(&mut self, name: String, value: parser::SubstitutionValue, builtin: bool) -> Result<()> {
        // Reject redefinition
        if self.constants.contains_key(&name) {
            return Err(anyhow!("Cannot redefine constant {}", name));
        };
        // Reject overriding of built-in constants
        if !builtin && BUILTIN_CONSTANTS.iter().any(|builtin_name| *builtin_name == name) {
            return Err(anyhow!("Cannot define constant {} due to built-in constant with the same name", name))
        };
        // Perform substitution
        self.particle_index.substitute_constant(&name, &value)?;
        self.interaction_index.substitute_constant(&name, &value)?;
        self.function_index.substitute_constant(&name, &value)?;
        // Remember substitution for debugging
        self.constants.insert(name, value);
        Ok(())
    }

    pub fn enable_interaction(&mut self, name: &str, details: InteractionDetails) -> Result<()> {
        let (interaction,_) = self.interaction_index.get_interaction_by_name(name)
            .ok_or(anyhow!("Cannot find interaction with name {}", name))?;
        if self.enabled_interactions.contains_key(&interaction) {
            return Err(anyhow!("Interaction with name {} is already enabled", name));
        }
        self.enabled_interactions.insert(interaction, details);
        Ok(())
    }

    /// Get the current simulation time
    pub fn get_time(&self) -> f64 {
        self.current_time
    }

    /// Set the current simulation time
    pub fn set_time(&mut self, time: f64) {
        self.current_time = time
    }

    /// Get the current time step
    pub fn get_time_step(&self) -> f64 {
        self.time_step
    }

    /// Set the current time step
    pub fn set_time_step(&mut self, time_step: f64) {
        self.time_step = time_step
    }

    /// Seed thread RNGs
    pub fn seed_rngs(&mut self, seeds: Vec<u64>) {
        self.rng_seeds = Some(seeds);
    }

    /// Get an (estimate) of the memory currently allocated
    /// This includes:
    /// - Memory allocated for particle data
    pub fn get_memory_usage(&self) -> usize {
        let mut memory = 0;
        memory += self.particle_store.get_memory_usage();
        memory
    }

    /// Verify and compile runtime for a given simulation
    pub fn compile(self, simulation_name: &str) -> Result<CompiledRuntime> {
        // Gather all simulation ids
        let (simulation, _) = self.simulation_index.get_simulation_by_name(simulation_name)
            .ok_or(anyhow!("Cannot find simulation with name {}", simulation_name))?;
        CompiledRuntime::new(self, simulation)
    }

    /// Borrow a particles data
    pub fn borrow_particle_mut(&self, particle_name: &str) -> Result<ParticleBorrowMut> {
        // Find particle definition in index
        let (particle_id, particle_definition) = match self.particle_index.get_particle_by_name(particle_name) {
            None => return Err(anyhow!("Cannot find particle {}", particle_name)),
            Some(particle) => particle
        };
        let particle_data = self.particle_store.get_particle_mut(particle_id)
            .ok_or(anyhow!("Particle type {} exists, but has not been initialized yet (call create_particles() first)", particle_name))?;
        Ok(ParticleBorrowMut::new(particle_id, particle_definition, particle_data))
    }
}