//! Components related to managing neighbor lists

//! Runtime representation of a pair-wise interaction between particle types

use std::{collections::HashMap, sync::{self, Arc, RwLock}, usize};

use anyhow::{Result, anyhow};
use num::Integer;

use crate::{runtime::{ParticleIndex, ParticleStore, OutOfBoundsBehavior}, utils::prime_factors};

use crate::runtime::{Domain, ParticleID};

use super::util::IndexRange;

fn argmax(s: &[usize]) -> usize {
    assert!(s.len() >= 1);
    let mut current_max_pos = 0;
    let mut current_max = s[0];
    for (i,x) in s.iter().enumerate() {
        if *x > current_max {
            current_max_pos = i;
            current_max = *x;
        }
    }
    return current_max_pos
}

fn ceil_div(x: usize, y: usize) -> usize {
    (x+y-1)/y
}

/// Try to split a domain into sensible subdomains (i.e. try to minimize contact surface)
/// Returns a tuple of the number of subdomains in each dimension as well as the size of each subdomain
fn split_domain(mut domain: [usize;3], num: usize) -> ([usize;3],[usize;3]) {
    let prime_facs = prime_factors(num);
    let mut subdomain_size = [1,1,1];
    for fac in prime_facs {
        let i = argmax(&domain);
        // If domain is not divisible by fac, the last subdomain in a dimension need to be smaller!
        domain[i] = ceil_div(domain[i], fac);
        subdomain_size[i] *= fac;
    }
    return (subdomain_size, domain)
}

// Extract the 3 coordinates for a given particle index
macro_rules! get_pos_3d {
    ($pos:expr, $i:expr) => {
        ($pos[3*$i], $pos[3*$i+1], $pos[3*$i+2])
    };
}

// Transform a real 3d position to a cell position (i.e. 3d integer position)
macro_rules! get_cell_coordinates_3d {
    ($pxyz:expr, $xyz:expr, $bin_size:expr, $cell_topo:expr) => {
        (
            ((($pxyz.0 - $xyz.0.low) / $bin_size).floor() as usize).min($cell_topo.0 - 1),
            ((($pxyz.1 - $xyz.1.low) / $bin_size).floor() as usize).min($cell_topo.1 - 1),
            ((($pxyz.2 - $xyz.2.low) / $bin_size).floor() as usize).min($cell_topo.2 - 1)
        )
    };
}

// Transform a cell position to subdomain indices
macro_rules! get_subdomain_coordinates_3d {
    ($cxyz:expr, $subdomain_size:expr) => {
        (
            (
                $cxyz.0 / $subdomain_size.0,
                $cxyz.1 / $subdomain_size.1,
                $cxyz.2 / $subdomain_size.2
            )
        )
    };
}

macro_rules! get_index_from_coords_3d {
    ($ixyz:expr, $nx:expr, $ny:expr) => {
        $ixyz.0 + $ixyz.1 * $nx + $ixyz.2 * $nx * $ny
    };
}

macro_rules! decompose_index_to_coords_3d {
    ($idx:expr, $nx:expr, $ny:expr) => { {
        let (tmp, ix) = $idx.div_rem(&$nx);
        let (iz, iy) = tmp.div_rem(&$ny);
        (ix,iy,iz)
    }};
}

macro_rules! subtract_3d {
    ($a:expr, $b:expr) => {
        (
            $a.0 - $b.0,
            $a.1 - $b.1,
            $a.2 - $b.2
        )
    };
}

macro_rules! add_3d {
    ($a:expr, $b:expr) => {
        (
            $a.0 + $b.0,
            $a.1 + $b.1,
            $a.2 + $b.2
        )
    };
}

macro_rules! length_sqr_3d {
    ($xyz:expr) => {
        $xyz.0 * $xyz.0 + $xyz.1 * $xyz.1 + $xyz.2 * $xyz.2
    };
}

macro_rules! to_isize_3d {
    ($a:expr) => {
       (
           $a.0 as isize,
           $a.1 as isize,
           $a.2 as isize,
       ) 
    };
}

pub(crate) struct NeighborList {
    /// Size of a single cell
    bin_size: f64,
    /// Cutoff length square (must not be smaller than square of bin_size)
    cutoff_length_sqr: f64,
    /// Copy of the simulation domain
    domain: Domain,
    /// Total topology of cells the domain is divided into
    cell_topo: (usize, usize, usize),
    /// Linearized storage for all the subdomains
    subdomains: Vec<RwLock<Subdomain>>,
    /// Topology of the subdomains
    subdomain_topo: (usize, usize, usize),
    /// Size of each subdomain (note: the last domain in each dimension might be larger)
    subdomain_size: (usize, usize, usize),
    /// Size of the index blocks
    pub(crate) pos_block_size: usize,
    /// Index blocks
    pub (crate) pos_blocks: Vec<(ParticleID, IndexRange)>,
    /// Thread pool for neighbor list rebuilding
    thread_pool: rayon::ThreadPool,
    /// Queues for particles that have moved across subdomain boundaries
    subdomain_queues: Vec<RwLock<Vec<Vec<(usize,(usize,usize,usize))>>>>,
    /// Barrier for workers
    worker_barrier: Arc<sync::Barrier>,
    /// Number of worker threads
    num_workers: usize,
    /// Particle lookup tables (maps particle index to cell index)
    lookup_tables: Vec<Vec<usize>>,
    /// The actual neighbor list structure
    pub(crate) neighbor_lists: Vec<(Vec<usize>, Vec<usize>)>,
    /// Rebuild interval in steps
    rebuild_interval: usize,
}

impl NeighborList {
    /// Create a new neighbor list
    pub(crate) fn new(bin_size: f64, cutoff_length: f64, domain: Domain, num_workers: usize, 
        rebuild_interval: usize, pos_blocks: HashMap<ParticleID, Vec<IndexRange>>,
        particle_index: &ParticleIndex, particle_store: &ParticleStore) -> Result<Self> {
        // Determine cell topology
        let cell_topo = match &domain {
            Domain::Dim2 { x, y } => {
                let nx = (x.size() / bin_size).floor() as usize;
                let ny = (y.size() / bin_size).floor() as usize;
                (nx, ny, 1)
            }
            Domain::Dim3 { x, y, z } => {
                let nx = (x.size() / bin_size).floor() as usize;
                let ny = (y.size() / bin_size).floor() as usize;
                let nz = (z.size() / bin_size).floor() as usize;
                (nx, ny, nz)
            }
        };
        // Try to split cell topology into sensible subdomains
        let (nx,ny,nz) = cell_topo;
        let (subdomain_topo, subdomain_size) = split_domain([nx,ny,nz], num_workers);
        let subdomain_topo = (subdomain_topo[0], subdomain_topo[1], subdomain_topo[2]);
        let subdomain_size = (subdomain_size[0], subdomain_size[1], subdomain_size[2]);

        // Construct subdomains
        fn get_lo_hi(ix: usize, sd_size_x: usize, sd_topo_x: usize, cell_topo_x: usize) -> (usize, usize) {
            let lo = ix*sd_size_x;
            let hi = if ix+1 == sd_topo_x {
                cell_topo_x
            }
            else {
                (ix+1)*sd_size_x
            };
            (lo,hi)
        }

        let mut subdomains = vec![];
        for iz in 0..subdomain_topo.2 {
            let (z_lo, z_hi) = get_lo_hi(iz, subdomain_size.2, subdomain_topo.2, cell_topo.2);
            for iy in 0..subdomain_topo.1 {
                let (y_lo, y_hi) = get_lo_hi(iy, subdomain_size.1, subdomain_topo.1, cell_topo.1);
                for ix in 0..subdomain_topo.0 {
                    let (x_lo, x_hi) = get_lo_hi(ix, subdomain_size.0, subdomain_topo.0, cell_topo.0);
                    subdomains.push(RwLock::new(Subdomain::new((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))));
                }
            }
        }

        // dbg!(cell_topo);
        // dbg!(subdomain_topo);
        // dbg!(subdomain_size);

        // Create index blocks (order must remain fixed for all time!)
        let pos_blocks = pos_blocks.into_iter()
            .map(|(particle_id, ranges)| ranges.into_iter().map(move |range| (particle_id, range)))
            .flatten()
            .collect::<Vec<_>>(); // Just recollect into Vec

        // Get max size of any index block
        let pos_block_size = pos_blocks.iter()
            .map(|(_,range)| range.len())
            .max().unwrap();

        // Create thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()?;

        // Create barrier
        let subdomain_queues = (0..num_workers)
            .map(|_| RwLock::new(vec![]))
            .collect::<Vec<_>>();
        let worker_barrier = Arc::new(sync::Barrier::new(num_workers));

        // Create empty neighbor list (one for each position block)
        let neighbor_lists = pos_blocks.iter()
            .map(|_| (vec![], vec![]))
            .collect::<Vec<_>>();

        // Create preallocated lookup table
        let lookup_tables = pos_blocks.iter()
            .map(|(_, range)| vec![0; range.len()])
            .collect::<Vec<_>>();

        let mut result = Self {
            bin_size, 
            cutoff_length_sqr: cutoff_length * cutoff_length,
            domain,
            cell_topo,
            subdomain_topo, subdomain_size,
            subdomains,
            pos_block_size,
            pos_blocks,
            thread_pool,
            subdomain_queues,
            worker_barrier,
            num_workers,
            lookup_tables,
            neighbor_lists,
            rebuild_interval
        };
        result.build(particle_index, particle_store)?;
        Ok(result)
    }

    fn get_dim(&self) -> usize {
        match &self.domain {
            Domain::Dim2 { .. } => { 2 }
            Domain::Dim3 { .. } => { 3 }
        }
    }

    fn build(&mut self, particle_index: &ParticleIndex, particle_store: &ParticleStore) -> Result<()> {
        // Get the positions (slightly mad due to RwLockReadGuards everywhere)
        let position_members = self.pos_blocks.iter()
            .map(|(particle_id, _)| particle_index.get(*particle_id).unwrap().get_position_member().unwrap().0);
        let particle_borrows = self.pos_blocks.iter()
            .map(|(particle_id,_)| particle_store.get_particle(*particle_id).unwrap())
            .collect::<Vec<_>>();
        let member_borrows = particle_borrows.iter()
            .zip(position_members)
            .map(|(particle_borrow, position_member)| particle_borrow.borrow_member(&position_member).unwrap())
            .collect::<Vec<_>>();
        let positions = member_borrows.iter()
            .zip(self.pos_blocks.iter())
            .map(|(member_borrow, (_, index_range))| &member_borrow.as_f64_slice()[index_range.to_range(self.get_dim())])
            .collect::<Vec<_>>();
        // Get write handle on all subdomains
        let mut subdomains = self.subdomains.iter()
            .map(|sd| sd.write().unwrap())
            .collect::<Vec<_>>();
        match &self.domain {
            Domain::Dim2 { .. } => {
                unimplemented!("2D neighbor lists currently unimplemented")
                // for (ipos, pos) in positions.iter().enumerate() {
                //     // Index offset for this index block
                //     let index_offset = ipos * self.index_block_size;
                //     for i in (0..pos.len()).step_by(2) {
                //         // Calculate index for this particle
                //         let idx = index_offset + i/2;
                //         // Get particle positions
                //         let px = pos[i];
                //         let py = pos[i+1];
                //         // Check if position is valid
                //         if !x.is_on_axis(px) || !y.is_on_axis(py) {
                //             return Err(anyhow!("Particle {} not in domain: position is {}, {}", idx, px, py));
                //         }
                //         // Calculate indices of subdomain this particle must be inserted in
                //         let ix = ((px - x.low) / self.bin_size).floor() as usize;
                //         let iy = ((py - y.low) / self.bin_size).floor() as usize;
                //         let sx = ix / self.subdomain_size.0;
                //         let sy = iy / self.subdomain_size.1;
                //         // Insert into correct cell list
                //         let (nx, _, _) = self.subdomain_topo;
                //         subdomains[sy * nx + sx].insert(idx, ix, iy, 0);
                //     }
                // }
            }
            Domain::Dim3 { x, y, z } => {
                for (ipos, pos) in positions.iter().enumerate() {
                    // Index offset for this position block
                    let index_offset = ipos * self.pos_block_size;
                    for i in 0..(pos.len()/3) {
                        // Calculate index for this particle
                        let idx = index_offset + i;
                        // Get particle positions
                        let pxyz = get_pos_3d!(pos, i);
                        let (px,py,pz) = pxyz;
                        // Check if position is valid
                        if !x.is_on_axis(px) || !y.is_on_axis(py) || !z.is_on_axis(pz) {
                            return Err(anyhow!("Particle {} not in domain: position is {}, {}, {}", idx, px, py, pz));
                        }
                        // Calculate index of subdomain this particle must be inserted in
                        let cxyz = get_cell_coordinates_3d!(pxyz, (x,y,z), self.bin_size, self.cell_topo);
                        let sxyz = get_subdomain_coordinates_3d!(cxyz, self.subdomain_size);
                        // dbg!(cxyz);
                        // dbg!(sxyz);
                        // Insert into correct cell list
                        let (snx, sny, _) = self.subdomain_topo;
                        // dbg!(subdomains[get_total_index_3d!(sxyz, snx, sny)].cxyz_lo);
                        // dbg!(subdomains[get_total_index_3d!(sxyz, snx, sny)].cxyz_hi);
                        subdomains[get_index_from_coords_3d!(sxyz, snx, sny)].insert(idx, cxyz);
                    }
                }
            }
        }
        std::mem::drop(subdomains);
        self.build_neighbor_list(positions);
        Ok(())
    }

    pub(crate) fn rebuild_if_required(&mut self, step: usize, particle_index: &ParticleIndex, particle_store: &ParticleStore) {
        if step % self.rebuild_interval == 0 {
            self.rebuild(particle_index, particle_store);
        }
    }

    pub(crate) fn rebuild(&mut self, particle_index: &ParticleIndex, particle_store: &ParticleStore) {
        // Get the positions (slightly mad due to RwLockReadGuards everywhere)
        let position_members = self.pos_blocks.iter()
            .map(|(particle_id, _)| particle_index.get(*particle_id).unwrap().get_position_member().unwrap().0);
        let particle_borrows = self.pos_blocks.iter()
            .map(|(particle_id,_)| particle_store.get_particle(*particle_id).unwrap())
            .collect::<Vec<_>>();
        let member_borrows = particle_borrows.iter()
            .zip(position_members)
            .map(|(particle_borrow, position_member)| particle_borrow.borrow_member(&position_member).unwrap())
            .collect::<Vec<_>>();
        let positions = member_borrows.iter()
            .zip(self.pos_blocks.iter())
            .map(|(member_borrow, (_, index_range))| &member_borrow.as_f64_slice()[index_range.to_range(self.get_dim())])
            .collect::<Vec<_>>();
        // Create copies / references to self fields (so we don't send self in its entirety)
        let cell_topo = &self.cell_topo;
        let subdomains = &self.subdomains;
        let subdomain_topo = self.subdomain_topo;
        let pos_block_size = self.pos_block_size;
        let bin_size = self.bin_size;
        let num_workers = self.num_workers;
        let subdomain_size = self.subdomain_size;
        let domain = &self.domain;
        let subdomain_queues = &self.subdomain_queues;
        let worker_barrier = &self.worker_barrier;
        // Rebuild the acceleration structure in the threadpool
        self.thread_pool.scope(|s| {
            subdomains.iter()
                .enumerate()
                .for_each(|(subdomain_idx, subdomain)| {
                    // Only hand reference down to the worker threads
                    let positions = &positions;
                    let worker_barrier = worker_barrier.clone();
                    s.spawn(move |_| {
                        // Get a write handle to the subdomain for this worker
                        let mut subdomain = subdomain.write().unwrap();
                        // Mark all cell lists (this will help us to discern particles we have checked vs. ones we did not yet)
                        subdomain.mark_cells();
                        // Get subdomain constants
                        let (snx, sny, _) = subdomain_topo;
                        let sxyz = decompose_index_to_coords_3d!(subdomain_idx, snx, sny);
                        let (cnx, cny, _) = subdomain.cnxyz;
                        // Create local queues for particles that have left this subdomain
                        let mut local_queues: Vec<Vec<(usize,(usize,usize,usize))>> = vec![vec![];num_workers];
                        // Dummy cell list to swap in for the current one
                        let mut cell = CellList::new(); 
                        match &domain {
                            Domain::Dim2 { .. } => { unimplemented!("2D neighbor lists currently unimplemented") }
                            Domain::Dim3 { x, y, z } => {
                                for cell_idx in 0..subdomain.cells.len() {
                                    // Swap cell with dummy so we can edit it
                                    std::mem::swap(&mut cell, &mut subdomain.cells[cell_idx]);
                                    let mut write_idx = 0; // Particles that remain in this cell will be inserted at this point
                                    // Some cell constants
                                    let cxyz = add_3d!(decompose_index_to_coords_3d!(cell_idx, cnx, cny), subdomain.cxyz_lo);
                                    for i in 0..cell.mark {
                                        let particle_idx = cell.list[i];
                                        // Get index of position block and index within block
                                        let (block_idx, local_idx) = particle_idx.div_rem(&pos_block_size);
                                        // Get positions
                                        let pos = positions[block_idx];
                                        let pxyz_new = get_pos_3d!(pos, local_idx);
                                        // Calculate index of subdomain this particle must be inserted in
                                        let cxyz_new = get_cell_coordinates_3d!(pxyz_new,(x,y,z),bin_size,cell_topo);
                                        let sxyz_new = get_subdomain_coordinates_3d!(cxyz_new, subdomain_size);
                                        // Has the particle moved from this subdomain?
                                        if sxyz != sxyz_new {
                                            // Queue particle for another subdomain to deal with (we recycle cxyz though)
                                            let subdomain_idx_new = get_index_from_coords_3d!(sxyz_new, snx, sny);
                                            local_queues[subdomain_idx_new].push((particle_idx, cxyz_new));
                                        }
                                        else {
                                            // Has the particle moved from its cell?
                                            if cxyz != cxyz_new {
                                                // Insert particle in correct cell
                                                subdomain.insert(particle_idx, cxyz_new);
                                            }
                                            else {
                                                // Just write the particle back into the cell list
                                                cell.list[write_idx] = particle_idx;
                                                write_idx += 1;
                                            }
                                        }
                                    }
                                    // Shift all particles beyond the mark to the new end
                                    for i in cell.mark..cell.list.len() {
                                        cell.list[write_idx] = cell.list[i];
                                        write_idx += 1;
                                    }
                                    // Truncate list to new consolidated length
                                    cell.list.truncate(write_idx);
                                    // Swap cell back
                                    std::mem::swap(&mut cell, &mut subdomain.cells[cell_idx]);
                                }
                            }
                        }
                        // Send local queues to the global queues
                        local_queues.into_iter()
                            .zip(subdomain_queues.iter())
                            .for_each(|(local_queue, subdomain_queue)| {
                                // Acquire write access to subdomain_queue
                                let mut subdomain_queue = subdomain_queue.write().unwrap();
                                subdomain_queue.push(local_queue);
                            });
                        // Wait for other subdomains to finish
                        worker_barrier.wait();
                        // Swap global queue with empty one
                        let mut subdomain_queue = subdomain_queues[subdomain_idx].write().unwrap();
                        let mut queue = vec![];
                        std::mem::swap(&mut queue, &mut subdomain_queue);
                        // Insert all elements of the queue
                        for chunk in queue {
                            for (particle_idx, cxyz) in chunk {
                                subdomain.insert(particle_idx, cxyz);
                            }
                        }
                    });
                });
        });
        // For validation:
        // let mut count = 0;
        // for subdomain in subdomains {
        //     let subdomain = subdomain.read().unwrap();
        //     for cell in &subdomain.cells {
        //         count += cell.list.len();
        //     }
        // }
        // dbg!(count);
        self.build_neighbor_list(positions);
    }

    /// Construct the actual neighbor list (this has to be done completely even on a rebuild)
    fn build_neighbor_list(&mut self, positions: Vec<&[f64]>) {
        // TODO: Just assume 3d periodic boundary conditions on all axes for now
        match &self.domain {
            Domain::Dim2 { .. } => { unimplemented!("2D neighbor lists currently unimplemented") }
            Domain::Dim3 { x, y, z } => {
                assert!(matches!(x.oob, OutOfBoundsBehavior::Periodic));
                assert!(matches!(y.oob, OutOfBoundsBehavior::Periodic));
                assert!(matches!(z.oob, OutOfBoundsBehavior::Periodic));
            }
        }
        // Spawn worker threads (note that we spawn a thread for every position block, so ideally
        // the number of subdomains and the number of position blocks should be the same)
        let cell_topo = &self.cell_topo;
        let domain_size = match &self.domain {
            Domain::Dim2 { .. } => { unimplemented!("2D neighbor lists currently unimplemented") }
            Domain::Dim3 { x, y, z } => {
                (x.size(), y.size(), z.size())
            }
        };
        let pos_block_size = self.pos_block_size;
        let lookup_tables = &mut self.lookup_tables;
        let subdomains = &self.subdomains;
        let subdomain_size = &self.subdomain_size;
        let subdomain_topo = &self.subdomain_topo;
        let cutoff_length_sqr = self.cutoff_length_sqr;
        let neighbor_lists = &mut self.neighbor_lists;
        self.thread_pool.scope(|s| {
            // Spawn one thread for each position block
            positions.iter()
                .enumerate()
                .zip(neighbor_lists)
                .zip(lookup_tables.iter_mut())
                .for_each(|(((ipos, pos), neighbor_list), lookup_table)| {
                    let positions = positions.clone();
                    s.spawn(move|_| {
                        // The range of indices of interest for this worker
                        let myrange = (pos_block_size*ipos)..(pos_block_size*(ipos+1));
                        // Refresh the lookup table
                        let subdomains = subdomains.iter()
                            .map(|subdomain| subdomain.read().unwrap())
                            .collect::<Vec<_>>();
                        for subdomain in &subdomains {
                            let (cnx, cny, _) = subdomain.cnxyz;
                            // Search for particles of this position block in all cells
                            for (cell_idx_local, cell) in subdomain.cells.iter().enumerate() {
                                let cxyz = decompose_index_to_coords_3d!(cell_idx_local, cnx, cny);
                                let cell_idx_global = get_index_from_coords_3d!(add_3d!(subdomain.cxyz_lo, cxyz), cell_topo.0, cell_topo.1);
                                for particle_idx in &cell.list {
                                    // If the particle is in myrange, we update the lookup table
                                    if myrange.contains(particle_idx) {
                                        let particle_idx_local = particle_idx % pos_block_size;
                                        lookup_table[particle_idx_local] = cell_idx_global;
                                    }
                                }
                            }
                        }

                        // Clear the neighbor list
                        neighbor_list.0.clear();
                        neighbor_list.1.clear();

                        // Now we try to find all neighbors of each particle
                        // Stencil for the 3x3x3 cube of cells centered on the current particle
                        let stencil = [
                            (-1,-1,-1),(-1,-1,0),(-1,-1,1),
                            (-1, 0,-1),(-1, 0,0),(-1, 0,1),
                            (-1, 1,-1),(-1, 1,0),(-1, 1,1),
                            ( 0,-1,-1),( 0,-1,0),( 0,-1,1),
                            ( 0, 0,-1),( 0, 0,0),( 0, 0,1),
                            ( 0, 1,-1),( 0, 1,0),( 0, 1,1),
                            ( 1,-1,-1),( 1,-1,0),( 1,-1,1),
                            ( 1, 0,-1),( 1, 0,0),( 1, 0,1),
                            ( 1, 1,-1),( 1, 1,0),( 1, 1,1)
                        ];
                        // Macro for correcting under- and overflow of cell coordinates
                        macro_rules! correct_cx2 {
                            ($cx2:expr, $cnx:expr) => {
                                match $cx2 {
                                    cx2 if cx2 < 0 => ((cx2 + $cnx as isize) as usize, -1),
                                    cx2 if cx2 >= $cnx as isize => ((cx2 - $cnx as isize) as usize, 1),
                                    _ => ($cx2 as usize, 0)
                                }
                            }
                        }
                        // For all particles...
                        for particle_idx_local in 0..(pos.len()/3) {
                            // Get the cell indices for this particle's cell
                            let pxyz = get_pos_3d!(pos, particle_idx_local);
                            let cell_idx = lookup_table[particle_idx_local];
                            let (cnx, cny, _) = cell_topo;
                            let cxyz = decompose_index_to_coords_3d!(cell_idx, cnx, cny);
                            // For all possible offsets...
                            for cxyz_offset in &stencil {
                                // TODO: maybe some caching?
                                // Get the uncorrected cell coordinates
                                let (cx2, cy2, cz2) = add_3d!(to_isize_3d!(cxyz), cxyz_offset);
                                // Correct all coordinates
                                let (cx2, corr_x) = correct_cx2!(cx2, cell_topo.0);
                                let (cy2, corr_y) = correct_cx2!(cy2, cell_topo.1);
                                let (cz2, corr_z) = correct_cx2!(cz2, cell_topo.2);
                                let cxyz_2 = (cx2, cy2, cz2);
                                // Create the correction vector
                                let pxyz_corr = (
                                    domain_size.0 * corr_x as f64,
                                    domain_size.1 * corr_y as f64,
                                    domain_size.2 * corr_z as f64
                                );
                                // Get the correct cell
                                let sxyz_2 = get_subdomain_coordinates_3d!(cxyz_2, subdomain_size);
                                let (snx, sny, _) = subdomain_topo;
                                let subdomain_idx = get_index_from_coords_3d!(sxyz_2, snx, sny);
                                let subdomain = &subdomains[subdomain_idx];
                                let cell_idx_2 = subdomain.get_cell_index(cxyz_2);
                                let cell = &subdomain.cells[cell_idx_2];
                                // Now test for all particles in this cell
                                for particle_idx_2_global in &cell.list {
                                    // Get the position block and the local particle index for particle 2
                                    let (ipos_2, particle_idx_2_local) = particle_idx_2_global.div_rem(&pos_block_size);
                                    // If in the same position block: skip unnecessary elements of the neighbor list
                                    if ipos_2 == ipos {
                                        if particle_idx_2_local <= particle_idx_local {
                                            continue;
                                        }
                                    }
                                    // Get corrected particle position
                                    let pxyz_2 = get_pos_3d!(positions[ipos_2], particle_idx_2_local);
                                    let pxyz_2 = add_3d!(pxyz_2, pxyz_corr);
                                    // Test distance
                                    let dist_sqr = length_sqr_3d!(subtract_3d!(pxyz, pxyz_2));
                                    if dist_sqr < cutoff_length_sqr {
                                        // If closer than cutoff, add to neighbor list
                                        neighbor_list.1.push(*particle_idx_2_global);
                                        // dbg!(pxyz);
                                        // dbg!(pxyz_2);
                                    }
                                }
                            }
                            // Not we have found all interactions for this particle, so we update the
                            // indexing part of the neighbor list
                            neighbor_list.0.push(neighbor_list.1.len());
                        }
                    });
                })
        });
        // for (x,y) in &self.neighbor_lists {
        //     dbg!(x);
        //     dbg!(y);
        // }
    }
}

struct Subdomain {
    cxyz_lo: (usize, usize, usize),
    _cxyz_hi: (usize, usize, usize),
    cnxyz: (usize, usize, usize),
    cells: Vec<CellList>
}

impl Subdomain {
    // Create a new subdomain including all cells from cxyz_lo to cxyz_hi (not inclusive)
    fn new(cxyz_lo: (usize, usize, usize), cxyz_hi: (usize, usize, usize)) -> Self {
        let cnxyz = subtract_3d!(cxyz_hi, cxyz_lo);
        let num_cells = cnxyz.0 * cnxyz.1 * cnxyz.2;
        Self {
            cxyz_lo,
            _cxyz_hi: cxyz_hi,
            cnxyz,
            cells: vec![CellList::new(); num_cells]
        }
    }

    fn get_cell_index(&self, cxyz: (usize, usize, usize)) -> usize {
        // Subtract offset of this subdomain to get the local index
        let cxyz = subtract_3d!(cxyz, self.cxyz_lo);
        let (cnx,cny,_) = self.cnxyz;
        get_index_from_coords_3d!(cxyz, cnx, cny)
    }

    fn insert(&mut self, idx: usize, cxyz: (usize, usize, usize)) {
        let cxyz = self.get_cell_index(cxyz);
        self.cells[cxyz].push(idx);
    }

    /// Mark current cell list length for all cells
    fn mark_cells(&mut self) {
        for cell in &mut self.cells {
            cell.mark = cell.list.len();
        }
    }
}

/// Non-linked cell list: Rebuild procedure
/// 1. Mark the current length of all cells
/// 2. For each cell: check all particles up to the marked length and move them
///    to a different cell if necessary
/// 3. Consolidate every cell list by moving all indices past the original length
///    to the last non-moved index (must be shorter or equal to the old tail)
/// 4. Reset vector size (not capacity!) to actual size (length of non-moved)
#[derive(Clone)]
pub(crate) struct CellList {
    list: Vec<usize>,
    mark: usize
}

impl CellList {
    pub fn new() -> Self {
        Self {
            list: Vec::new(),
            mark: 0
        }
    }

    pub fn push(&mut self, idx: usize) {
        self.list.push(idx);
    }
}