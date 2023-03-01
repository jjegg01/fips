//! Structs for managing runtime data for particle types

use anyhow::{Result, anyhow};
use aligned_box::AlignedBox;

use core::panic;
use std::{collections::HashMap, sync::{RwLock, RwLockReadGuard, RwLockWriteGuard}, unimplemented, usize};

use crate::{runtime::{ParticleID, MemberID, MemberIndexEntry, ParticleIndexEntry}, utils::FipsValue};

// For now we just enforce 32 byte alignment (enough for AVX2 and older)
// TODO: Better allocation management once custom allocators become stable
const ALIGNMENT: usize = 32;

/// Options for allocating data for constant members
#[derive(Clone)]
pub struct UniformMembers {
    single_allocations: HashMap<String, FipsValue>
    // /// Only allocate a single instance for all particles of the same kind
    // Single(HashMap<String, parser::SubstitutionValue>),
    // /// Allocate a separate instance for every particle
    // PerParticle
}

impl UniformMembers {
    pub fn new() -> Self {
        Self {
            single_allocations: HashMap::new()
        }
    }

    pub fn set_uniform_member<S: Into<String>, T: Into<FipsValue>>(&mut self, name: S, value: T) {
        self.single_allocations.insert(name.into(), value.into());
    }

    fn get_member(&self, name: &str) -> Option<&FipsValue> {
        self.single_allocations.get(name)
    }
}

/// Store (and thus root owner) for all particle data used in a simulation
pub struct ParticleStore {
    /// All stored particle data
    particles: HashMap<ParticleID, RwLock<ParticleData>>
}

impl ParticleStore {
    /// Create new particle store
    pub(crate) fn new() -> Self {
        Self {
            particles: HashMap::new()
        }
    }

    /// Allocate data for a given number of particles
    ///
    /// Note that you have to define a default method for allocating constant members.
    /// If mixed allocations of constants are desired, manually reallocate the
    /// corresponding members after this call.
    pub(crate) fn create_particles<'a>(&'a mut self, particle: (ParticleID, &ParticleIndexEntry),
        count: usize, uniform_members: &UniformMembers) -> Result<RwLockWriteGuard<'a, ParticleData>>
    {
        let (particle_id, particle_definition) = particle;
        // Check if particle is already allocated
        if self.particles.contains_key(&particle_id) {
            return Err(anyhow!("Particle with name {} has already been allocated",
                particle_definition.get_name()));
        };
        // Create new particle allocation
        let particle_data = ParticleData::new(particle_definition, count, uniform_members)?;
        self.particles.insert(particle_id, RwLock::new(particle_data));
        Ok(self.particles.get(&particle_id).unwrap().write().unwrap())
    }

    /// Get the (approximate) memory used for particle data in bytes
    pub(crate) fn get_memory_usage(&self) -> usize {
        let mut memory = 0;
        for (_, particle_data) in &self.particles {
            memory += particle_data.read().unwrap().get_memory_usage();
        }
        memory
    }

    /// Return particles for which data is stored as key-value pairs
    pub(crate) fn get_particles(&self) -> impl Iterator<Item = (ParticleID, &RwLock<ParticleData>)> {
        self.particles.iter().map(|(k,v)| (*k,v))
    }

    /// Get particle data for a given particle ID (automatically borrows)
    pub(crate) fn get_particle<'a>(&self, particle_id: ParticleID) -> Option<RwLockReadGuard<ParticleData>> {
        self.particles.get(&particle_id).map(|x| x.read().unwrap())
    }

    /// Get mutable particle data for a given particle ID (automatically borrows)
    pub(crate) fn get_particle_mut<'a>(&self, particle_id: ParticleID) -> Option<RwLockWriteGuard<ParticleData>> {
        self.particles.get(&particle_id).map(|x| x.write().unwrap())
    }
}

/// Data for all instances of a single particle type
pub struct ParticleData {
    /// Number of particles
    count: usize,
    // /// Array of particle positions
    // positions: Pin<AlignedBox<[f64]>>,
    /// Array of byte arrays containing other per-particle member data
    members: HashMap<MemberID, RwLock<MemberData>>
}

impl ParticleData {
    /// Create new allocations for a single particle type
    pub(crate) fn new(particle: &ParticleIndexEntry, count: usize, uniform_members: &UniformMembers) -> Result<Self>
    {
        let mut members = HashMap::new();
        for (member_id, member) in particle.get_members() {
            members.insert(member_id, RwLock::new(MemberData::new(member, count, uniform_members)?));
        }
        Ok(Self {
            count,
            // positions: Pin::new(AlignedBox::slice_from_value(ALIGNMENT, count*domain.get_dim(), 0.0)
            //     .map_err(|e| anyhow!("Cannot allocate particle positions: {}", e))?),
            members 
        })
    }

    /// Get number of particles
    pub(crate) fn get_particle_count(&self) -> usize {
        self.count
    }

    /// Get memory used for the data of this particle type
    pub(crate) fn get_memory_usage(&self) -> usize {
        let mut memory = 0;
        //memory += self.positions.len() * std::mem::size_of::<f64>();
        for (_, member_data) in &self.members {
            memory += member_data.read().unwrap().get_memory_usage();
        }
        memory
    }

    pub(crate) fn borrow_member(&self, member_id: &MemberID) -> Option<RwLockReadGuard<MemberData>> {
        match self.members.get(member_id) {
            None => None,
            Some(member) => {
                Some(member.read().unwrap())
            }
        }
    }

    pub(crate) fn borrow_member_mut(&self, member_id: &MemberID) -> Option<RwLockWriteGuard<MemberData>> {
        match self.members.get(member_id) {
            None => None,
            Some(member) => {
                Some(member.write().unwrap())
            }
        }
    }
}

pub enum MemberData {
    /// One value for every particle
    PerParticle {
        /// Data as byte array
        data: AlignedBox<[u8]>,
        /// Flag for constant members
        mutable: bool
    },
    /// The same value for all particles (this must be constant)
    Uniform(FipsValue)

    // /// Mutable particle data (definitely per particle)
    // Mutable {
    //     data: AlignedBox<[u8]>
    // },
    // /// Constant particle data (maybe per particle, but maybe constant for all particles)
    // Constant(ConstMemberData)
}

impl MemberData {
    /// Create a new member data storage
    pub(crate) fn new(member: &MemberIndexEntry, count: usize, uniform_members: &UniformMembers) -> Result<Self> {
        // Is member in 
        match uniform_members.get_member(member.get_name()) {
            Some(value) => {
                Ok(Self::Uniform(value.clone()))
            }
            None => {
                let data = AlignedBox::slice_from_value(ALIGNMENT, count*member.get_member_size()?, 0u8)
                    .map_err(|e| anyhow!("Cannot allocate particle member data: {}", e))?;
                let mutable = member.is_mutable();
                Ok(Self::PerParticle {
                    data, mutable
                })
            }
        }
    }

    /// Get allocated memory size in bytes
    pub(crate) fn get_memory_usage(&self) -> usize {
        match self {
            MemberData::PerParticle { data, .. } => { data.len() }
            MemberData::Uniform(value) => { value.get_size() }
        }
    }

    // /// Returns true if allocations are made per-particle
    // pub(crate) fn is_per_particle(&self) -> bool {
    //     match self {
    //         MemberData::PerParticle{..} => true,
    //         MemberData::Uniform(_) => false
    //     }
    // }

    /// Returns true if this member is allocated uniformly
    /// (i.e. only one instance for all particles of this type)
    pub(crate) fn is_uniform(&self) -> bool {
        match self {
            MemberData::PerParticle{..} => false,
            MemberData::Uniform(_) => true
        }
    }

    pub(crate) fn as_i64(&self) -> i64 {
        match self {
            Self::PerParticle {data, ..} => *bytemuck::from_bytes::<i64>(data),
            Self::Uniform(value) => match value {
                FipsValue::Int64(value) => *value,
                _ => panic!("Cannot access member data as i64 (type is {})", value.get_type()) 
            }
        }
    }

    pub(crate) fn as_f64(&self) -> f64 {
        match self {
            Self::PerParticle {data, ..} => *bytemuck::from_bytes::<f64>(data),
            Self::Uniform(value) => match value {
                FipsValue::Double(value) => *value,
                _ => panic!("Cannot access member data as f64 (type is {})", value.get_type()) 
            }
        }
    }

    pub(crate) fn as_i64_slice(&self) -> &[i64] {
        match self {
            Self::PerParticle {data, ..} => bytemuck::cast_slice::<u8,i64>(data),
            Self::Uniform(_) => unimplemented!()
        }
    }
    
    pub(crate) fn as_f64_slice(&self) -> &[f64] {
        match self {
            Self::PerParticle {data, ..} => bytemuck::cast_slice::<u8,f64>(data),
            Self::Uniform(_) => unimplemented!()
        }
    }

    pub(crate) fn as_i64_slice_mut(&mut self) -> &mut [i64] {
        match self {
            Self::PerParticle {data, ..} => bytemuck::cast_slice_mut::<u8,i64>(data),
            Self::Uniform(_) => unimplemented!()
        }
    }

    pub(crate) fn as_f64_slice_mut(&mut self) -> &mut [f64] {
        match self {
            Self::PerParticle {data, ..} => bytemuck::cast_slice_mut::<u8,f64>(data),
            Self::Uniform(_) => unimplemented!()
        }
    }
}