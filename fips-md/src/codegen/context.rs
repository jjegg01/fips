//! Context data for the global level, simulation level and thread level

use std::{collections::HashMap, sync::{self, Arc, RwLock, mpsc}};
use anyhow::Result;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::runtime::{InteractionID, ParticleBorrowMut, ParticleID, Runtime};

use super::{CallbackMessage, analysis::BarrierID, analysis::{SimulationGraph, SymbolTable}, neighbors::NeighborList, util};

/// Context for global data (i.e. data shared by all threads)
/// Every thread gets an Arc reference of this
pub struct GlobalContext {
    /// Runtime that is compiled
    pub(crate) runtime: Runtime,
    /// Global symbol table
    pub(crate) global_symbols: SymbolTable<()>,
    /// Simulation graph of the compiled runtime
    pub(crate) simgraph: SimulationGraph,
}

/// Context of the executor (usually shared synchronization primitives)
/// Every threads gets a clone of this
#[derive(Clone)]
pub(crate) struct ExecutorContext {
    /// Synchronization barriers
    pub(crate) barriers: HashMap<BarrierID, Arc<sync::Barrier>>,
    /// The simulation global step barrier
    pub(crate) step_barrier: Arc<sync::Barrier>,
    /// The simulation step counter
    pub(crate) step_counter: Arc<RwLock<usize>>,
    /// The simulation global barrier for ending a call to Rust
    /// (this unblocks the worker threads)
    pub(crate) call_end_barrier: Arc<sync::Barrier>,
    /// Sender for communication with the callback thread
    pub(crate) call_sender: mpsc::Sender<CallbackMessage>,
    /// Neighbor lists for interactions
    pub(crate) neighbor_lists: HashMap<InteractionID, Arc<RwLock<NeighborList>>>,
    /// Immutable reference to compiled runtime
    pub(crate) global_context: Arc<GlobalContext>
}

/// Representation of all data in a single worker thread
pub(crate) struct ThreadContext {
    /// Particle (type) ID this thread is associated with
    pub(crate) particle_id: ParticleID,
    /// Range of particle numbers this thread is associated with
    pub(crate) particle_range: util::IndexRange,
    /// Thread-local random number generator
    pub(crate) rng: Xoshiro256PlusPlus,
    /// Normal distribution (TODO: Move this somewhere more sensible)
    pub(crate) normal_dist: rand_distr::Normal<f64>,
    /// Immutable reference to simulation context
    pub(crate) executor_context: ExecutorContext,
}

impl GlobalContext {
    // Borrow data belonging to a particle type
    pub fn borrow_particle_mut(&self, particle_name: &str) -> Result<ParticleBorrowMut> {
        self.runtime.borrow_particle_mut(particle_name)
    }
}