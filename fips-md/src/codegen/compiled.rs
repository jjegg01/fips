use std::{collections::{HashMap, HashSet}, ops::Range, sync::{self, Arc, RwLock}};

use anyhow::{anyhow, Result};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::{codegen::{analysis::SimulationGraph, util, ExecutorContext}, runtime::{ParticleBorrowMut, Runtime}};

use crate::runtime::{SimulationID, ParticleID};

use super::{CallbackStateType, CallbackThread, CallbackType, GlobalContext, ThreadContext, WorkerThread, analysis::SymbolTable, neighbors::NeighborList, util::IndexRange};

/// Wrapper struct for runtime after compilation
///
/// This wrapping is intended to make all changes to the internal state impossible
/// that would break the compiled simulation (thus wrapping all the unsafety of
/// code generation)
pub struct CompiledRuntime {
    /// ID of the simulation we compiled for
    _simulation: SimulationID,
    /// Simulation executor (this spawn the threads for the actual computation)
    executor: SimulationExecutor,
    /// Active particle types (i.e. particle types that have at least one instance)
    _active_particles: HashSet<ParticleID>,
    /// Global compilation context
    global_context: Arc<GlobalContext>,
}

impl CompiledRuntime {
    /// Create a new compiled runtime from a regular runtime structure
    /// The second parameter determines the simulations that will be compiled
    pub(crate) fn new(runtime: Runtime, simulation: SimulationID) -> Result<Self> {
        // Create index shorthands
        let indices = util::IndicesRef::new(
            &runtime.particle_index,
            &runtime.simulation_index,
            &runtime.interaction_index,
        );
        // Construct global symbol table
        let mut global_symbols = SymbolTable::new();
        for (name, value) in &runtime.constants {
            global_symbols.add_constant_from_substitution(name.clone(), value)?;
        }
        for (function_id, function_def) in runtime.function_index.get_functions() {
            global_symbols.add_function(function_def.get_name().to_string(), function_id)?;
        }
        // Insert additional constants
        global_symbols.add_constant_f64("DT".into(), runtime.get_time_step())?;
        // Determine active particles
        let active_particles = runtime.particle_store.get_particles()
            .map(|(particle_id, _)| particle_id)
            .collect::<HashSet<_>>();
        // Construct simulation graphs
        let simgraph = SimulationGraph::new(active_particles.clone(), simulation, indices)?;
        // Create global context (factored out because it is borrowed to all worker threads)
        let global_context = Arc::new(GlobalContext {
            runtime, global_symbols, simgraph
        });
        // Create simulation contexts
        let executor = SimulationExecutor::new(global_context.clone(), &active_particles)?;
        // Return compiled runtime
        Ok(Self {
            global_context: global_context.clone(),
            executor, _simulation: simulation,
            _active_particles: active_particles
            // callbacks: HashMap::new()
        })
    }

    /// Run a single time step
    pub fn run_step(&self) {
        // BIG TODO
        self.executor.run();
        // We need to wait for all workers to complete (otherwise we could create
        // a data race)
        self.wait();
    }

    fn wait(&self) {
        // BIG TODO
        self.executor.wait();
    }

    pub fn join(self) {
        // BIG TODO
        self.executor.join();
    }

    // Borrow data belonging to a particle type
    pub fn borrow_particle_mut(&self, particle_name: &str) -> Result<ParticleBorrowMut> {
        self.global_context.borrow_particle_mut(particle_name)
    }

    /// Define a new callback
    pub fn define_callback(&mut self, name: &str,
        callback: CallbackType,
        callback_state: CallbackStateType) 
    -> Result<()> {
        self.executor.register_callback(name, callback, callback_state)
    }

    /// Undefine an existing callback (to get its context back)
    pub fn undefine_callback(&mut self, name: &str)
    -> Result<(CallbackType, CallbackStateType)> {
        self.executor.unregister_callback(name)
    }

    /// Manually rebuild the neighbor list for a given interaction
    pub fn rebuild_neighbor_list(&mut self, interaction_name: &str) -> Result<()> {
        self.executor.rebuild_neighbor_list(interaction_name)
    }

    pub fn get_neighbor_lists(&mut self, interaction_name: &str) -> Result<HashMap<(ParticleID, Range<usize>), (Vec<usize>, Vec<usize>)>> {
        self.executor.get_neighbor_lists(interaction_name)
    }
}

/// Execution handle to a single simulation
pub(crate) struct SimulationExecutor {
    /// Worker threads
    workers: Vec<WorkerThread>,
    /// Callback thread
    callback_thread: CallbackThread,
    // /// Global context
    // global_context: Arc<GlobalContext>,
    /// Executor context
    executor_context: ExecutorContext
}

impl SimulationExecutor {
    pub(crate) fn new(global_context: Arc<GlobalContext>, active_particles: &HashSet<ParticleID>) -> Result<Self> {
        let simgraph = &global_context.simgraph;
        let particle_store = &global_context.runtime.particle_store;
        let threads_per_particle = &global_context.runtime.threads_per_particle;
        let interaction_index = &global_context.runtime.interaction_index;
        let particle_index = &global_context.runtime.particle_index;
        let enabled_interactions = &global_context.runtime.enabled_interactions;
        let domain = &global_context.runtime.domain;
        let rng_seeds = &global_context.runtime.rng_seeds;
        // Determine the index ranges for the worker threads
        let mut worker_index_ranges = HashMap::new();
        for particle_id in active_particles {
            let particle_count = particle_store
                .get_particle(*particle_id).unwrap().get_particle_count();
            worker_index_ranges.insert(*particle_id, 
                IndexRange::new(0, particle_count).split(
                    *threads_per_particle.get(particle_id).unwrap()
                )
            );
        }
        // Determine neighbor lists to maintain
        let mut neighbor_lists = HashMap::new();
        for (interaction, interaction_def) in interaction_index.iter() {
            // Skip interactions that are dead
            let affected_particles = interaction_def.get_affected_particles(particle_index)?;
            if let None = affected_particles.intersection(&active_particles).next() {
                continue;
            }
            // Skip interactions that have not been enabled
            match enabled_interactions.get(&interaction) {
                Some(details) => {
                    let position_blocks = affected_particles.intersection(&active_particles)
                        .map(|particle_id| 
                            (*particle_id, worker_index_ranges.get(particle_id).unwrap().clone()))
                        .collect::<HashMap<_,_>>();
                    let cutoff_length = details.skin_factor * util::unwrap_f64_constant(&interaction_def.get_cutoff())?;
                    let bin_size = details.cell_size.unwrap_or(cutoff_length);
                    neighbor_lists.insert(interaction, Arc::new(RwLock::new(
                        NeighborList::new(
                            bin_size, cutoff_length,
                            domain.clone(),
                            details.num_workers,
                            details.rebuild_interval,
                            position_blocks,
                            particle_index,
                            particle_store
                        )?))
                    );
                },
                None => { continue; }
            }
        }
        // Create barriers
        let barriers = simgraph.barriers.iter()
            .map(|(barrier_id, barrier_def)| (
                barrier_id,
                // Barrier count is sum of thread counts for each affected particle
                Arc::new(sync::Barrier::new(
                    barrier_def.affected_particles.iter()
                        .map(|particle_id| worker_index_ranges.get(particle_id).unwrap().len())
                        .sum()
                ))
            ))
            .collect::<HashMap<_,_>>();
        // Total number of workers is the sum of the number of workers of each particle type
        let num_workers = worker_index_ranges.iter().map(|(_,ranges)| ranges.len()).sum();
        let step_barrier = Arc::new(sync::Barrier::new(num_workers));
        let call_end_barrier = Arc::new(sync::Barrier::new(num_workers+1));
        // The step counter always starts at zero
        let step_counter = Arc::new(RwLock::new(0));
        // Create the callback thread
        let callback_thread = CallbackThread::new(call_end_barrier.clone(), num_workers, global_context.clone());
        // Create context
        let executor_context = ExecutorContext {
            barriers, step_barrier, step_counter,
            call_end_barrier,
            neighbor_lists,
            call_sender: callback_thread.get_sender(),
            global_context: global_context.clone()
        };
        // Create worker threads
        let rng_seeds = match rng_seeds {
            None => unimplemented!("Explicit RNG seeding required for now"),
            Some(rng_seeds) => {
                if rng_seeds.len() >= num_workers {
                    Ok(rng_seeds)
                }
                else {
                    Err(anyhow!("Not enough RNG seeds (need {}, got {})", num_workers, rng_seeds.len()))
                }
            }
        }?;
        let workers = worker_index_ranges.into_iter()
            .zip(&rng_seeds[0..num_workers])
            .map(|((particle_id, index_ranges), rng_seed)| {
                let executor_context = &executor_context;
                index_ranges.into_iter().map(move |particle_range| {
                    // Create thread specific context
                    let rng = Xoshiro256PlusPlus::seed_from_u64(*rng_seed);
                    let normal_dist = rand_distr::Normal::new(0.0, 1.0)
                        .expect("Math is broken. All is lost.");
                    let thread_context = ThreadContext {
                        particle_id, particle_range, rng, normal_dist,
                        executor_context: executor_context.clone()
                    };
                    // Spawn worker thread
                    WorkerThread::spawn(thread_context)
                })
            })
            .flatten()
            .collect::<Vec<_>>();
        // Wait for workers to finish compiling
        for worker in &workers {
            worker.wait_for_compilation();
        }
        Ok(Self {
            workers, callback_thread, executor_context
        })
    }

    pub(crate) fn run(&self) {
        // BIG TODO
        for worker in &self.workers {
            worker.run_step();
        }
    }

    pub(crate) fn wait(&self) {
        // No need to wait for the callback thread, since the workers will only
        // be idle if the callback worker is too
        for worker in &self.workers {
            worker.wait();
        }
    }

    pub(crate) fn join(self) {
        for worker in self.workers {
            worker.join();
        }
        self.callback_thread.join();
    }

    pub(crate) fn register_callback(&mut self, name: &str,
        callback: CallbackType,
        callback_state: CallbackStateType) 
    -> Result<()> {
        // Try to resolve callback
        let barriers = self.executor_context.global_context.simgraph.callbacks.get(name)
            .ok_or(anyhow!("No callback named {} found in simulation graph", &name))?
            .clone();
        self.callback_thread.register_callback(barriers, callback, callback_state)
    }

    pub(crate) fn unregister_callback(&mut self, name: &str) 
    -> Result<(CallbackType, CallbackStateType)> {
        let barriers = self.executor_context.global_context.simgraph.callbacks.get(name)
            .ok_or(anyhow!("No callback named {} found in simulation graph", &name))?
            .clone();
        self.callback_thread.unregister_callback(barriers)
    }

    pub(crate) fn rebuild_neighbor_list(&mut self, interaction_name: &str) -> Result<()> {
        let (interaction_id,_) = self.executor_context.global_context.runtime.interaction_index
            .get_interaction_by_name(interaction_name)
            .ok_or(anyhow!("Cannot find interaction with name {}", interaction_name))?;
        self.executor_context.neighbor_lists.get_mut(&interaction_id)
            .ok_or(anyhow!("There is no neighbor list for interaction with name {}", interaction_name))?
            .write().unwrap()
            .rebuild(&self.executor_context.global_context.runtime.particle_index, 
                &self.executor_context.global_context.runtime.particle_store);
        Ok(())
    }

    pub fn get_neighbor_lists(&mut self, interaction_name: &str) -> Result<HashMap<(ParticleID, Range<usize>), (Vec<usize>, Vec<usize>)>> {
        // Check if interaction exists
        let (interaction_id,_) = self.executor_context.global_context.runtime.interaction_index
            .get_interaction_by_name(interaction_name)
            .ok_or(anyhow!("Cannot find interaction with name {}", interaction_name))?;
        // Get read handle to neighbor list structure
        let neighbor_list = self.executor_context.neighbor_lists.get(&interaction_id)
            .ok_or(anyhow!("There is no neighbor list for interaction with name {}", interaction_name))?
            .read().unwrap();
        // Zip pos blocks and (sub) neighbor lists together and clone lists for return value
        Ok(neighbor_list.pos_blocks.iter()
            .zip(neighbor_list.neighbor_lists.iter())
            .map(|((particle_id, index_range), (neighbor_list_index, neighbor_list))| {
                ((*particle_id, index_range.start..index_range.end), 
                (neighbor_list_index.clone(), neighbor_list.clone()))
            })
            .collect::<HashMap<_,_>>())
    }
}

