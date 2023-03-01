//! Components related to barriers in the simulation graph

use std::collections::HashSet;

use slotmap::{DefaultKey};

use crate::runtime::{InteractionID, InteractionQuantityID, ParticleID};

/// Different barriers can be shared by having the same barrier ID
pub type BarrierID = DefaultKey;

/// A synchronization barrier (i.e. the data part of a barrier node in the
/// simulation graph)
#[derive(Clone)]
pub struct Barrier {
    /// Particle types affected by this barrier
    pub affected_particles: HashSet<ParticleID>,
    /// Kind of barrier
    pub kind: BarrierKind
}

impl Barrier {
    pub(crate) fn new(affected_particles: HashSet<ParticleID>, kind: BarrierKind) -> Self {
        Self {
            affected_particles, kind
        }
    }
}

/// Types of synchronization barriers
#[derive(Clone)]
pub enum BarrierKind {
    /// Barrier due to particle interaction
    InteractionBarrier(InteractionID, Option<InteractionQuantityID>),
    /// Barrier due to a call to Rust
    CallBarrier(String)
}