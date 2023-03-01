//! Structures related to interaction runtime information

pub struct InteractionDetails {
    /// Number of worker threads for neighbor list construction
    pub num_workers: usize,
    /// Skin factor (this factor is multiplied to the cutoff region to guarantee
    /// the correctness of the neighbor list for more than one tick)
    pub skin_factor: f64,
    /// Explicit cell size
    pub cell_size: Option<f64>,
    /// The neighbor list is rebuild after this number of time steps
    pub rebuild_interval: usize
}