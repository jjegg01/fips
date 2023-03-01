//! Structs representing borrows into the internal particle data

use std::{sync::{RwLockReadGuard, RwLockWriteGuard}};

use anyhow::{anyhow, Result};

use crate::{
    runtime::{ParticleID, ParticleIndexEntry, ParticleData},
    parser::FipsType
};

use super::MemberData;

/// Struct representing an immutable borrow of data belonging to a single particle member
pub struct FipsBorrow<'a> {
    /// The borrowed member data
    member_data: RwLockReadGuard<'a, MemberData>,
    /// The type of data (used in error checking to prevent unsound casts)
    typ: FipsType
}

/// Struct representing an immutable borrow of data belonging to a single particle member
pub struct FipsBorrowMut<'a> {
    /// The borrowed member data
    member_data: RwLockWriteGuard<'a, MemberData>,
    /// The type of data (used in error checking to prevent unsound casts)
    typ: FipsType
}

/// A mutable borrow of all data relating to one particle type
pub struct ParticleBorrowMut<'a> {
    particle_id: ParticleID,
    particle_definition: &'a ParticleIndexEntry,
    particle_data: RwLockWriteGuard<'a, ParticleData>
}

impl<'a> ParticleBorrowMut<'a> {
    pub(crate) fn new(particle_id: ParticleID, particle_definition: &'a ParticleIndexEntry, 
        particle_data: RwLockWriteGuard<'a, ParticleData>) -> Self
    {
        Self {
            particle_id, particle_data, particle_definition
        }
    }

    /// Get particle id
    pub fn get_particle_id(&self) -> ParticleID {
        self.particle_id
    }

    /// Get number of particles of this type
    pub fn get_particle_count(&self) -> usize {
        self.particle_data.get_particle_count()
    }

    /// Borrow all data of a member of this particle type immutably
    pub fn borrow_member (&self, name: &str) -> Option<FipsBorrow> {
        match self.particle_definition.get_member_by_name(name) {
            None => None,
            Some((member_id, member_definition)) => {
                let member_data = self.particle_data.borrow_member(&member_id);
                match member_data {
                    None => return None,
                    Some(member_data) => {
                        let typ = member_definition.get_type();
                        Some(FipsBorrow {
                            member_data,
                            typ: typ.clone()
                        })
                    }
                }
            }
        }
    }

    /// Borrow all data of a member of this particle type mutably
    pub fn borrow_member_mut (&self, name: &str) -> Option<FipsBorrowMut> {
        match self.particle_definition.get_member_by_name(name) {
            None => None,
            Some((member_id, member_definition)) => {
                let member_data = self.particle_data.borrow_member_mut(&member_id);
                match member_data {
                    None => return None,
                    Some(member_data) => {
                        if member_data.is_uniform() {
                            panic!("Cannot borrow uniform member mutably. Use the uniform_members parameter in create_particles to initialize uniform members.");
                        }
                        else {
                            let typ = member_definition.get_type();
                            Some(FipsBorrowMut {
                                member_data, typ: typ.clone()
                            })
                        }
                    }
                }
            }
        }
    }
}

impl<'a> FipsBorrow<'a> {
    /// Interpret the data as a single i64 (only sensible for constants)
    pub fn as_i64(&self) -> Result<i64> {
        if !self.member_data.is_uniform() {
            Err(anyhow!("Cannot access member as single i64 since it is allocated per-particle. Did you mean as_i64_slice()?"))
        }
        else {
            // Type must be scalar and based on i64
            if self.typ.is_scalar() && self.typ.is_i64_derived() {
                Ok(self.member_data.as_i64())
            }
            else {
                Err(anyhow!("Member cannot be cast to single i64 (type was {})", self.typ))
            }
        }
    }

    /// Interpret the data as a single f64 (only sensible for constants)
    pub fn as_f64(&self) -> Result<f64> {
        if !self.member_data.is_uniform() {
            Err(anyhow!("Cannot access member as single f64 since it is allocated per-particle. Did you mean as_f64_slice()?"))
        }
        else {
            // Type must be scalar and based on f64
            if self.typ.is_scalar() && self.typ.is_f64_derived() {
                Ok(self.member_data.as_f64())
            }
            else {
                Err(anyhow!("Member cannot be cast to single f64 (type was {})", self.typ))
            }
        }
    }

    /// Interpret the data as a slice of double values
    pub fn as_f64_slice(&self) -> Result<&[f64]> {
        // Anything can be cast to a slice as long as it is based on f64 values
        if self.typ.is_f64_derived() {
            Ok(self.member_data.as_f64_slice())
        }
        else {
            Err(anyhow!("Member cannot be cast to slice of f64 (type was {})", self.typ))
        }
    }

    /// Interpret the data as a slice of double values
    pub fn as_i64_slice(&self) -> Result<&[i64]> {
        // Anything can be cast to a slice as long as it is based on f64 values
        if self.typ.is_i64_derived() {
            Ok(self.member_data.as_i64_slice())
        }
        else {
            Err(anyhow!("Member cannot be cast to slice of i64 (type was {})", self.typ))
        }
    }
}

impl<'a> FipsBorrowMut<'a> {
    // TODO: Remove as_i64 and as_f64? Constants cannot be borrowed mutably anyway

    /// Interpret the data as a single i64 (only sensible for constants)
    pub fn as_i64(&self) -> Result<i64> {
        if !self.member_data.is_uniform() {
            Err(anyhow!("Cannot access member as single i64 since it is allocated per-particle. Did you mean as_i64_slice()?"))
        }
        else {
            // Type must be scalar and based on i64
            if self.typ.is_scalar() && self.typ.is_i64_derived() {
                Ok(self.member_data.as_i64())
            }
            else {
                Err(anyhow!("Member cannot be cast to single i64 (type was {})", self.typ))
            }
        }
    }

    /// Interpret the data as a single f64 (only sensible for constants)
    pub fn as_f64(&self) -> Result<f64> {
        if !self.member_data.is_uniform() {
            Err(anyhow!("Cannot access member as single f64 since it is allocated per-particle. Did you mean as_f64_slice()?"))
        }
        else {
            // Type must be scalar and based on f64
            if self.typ.is_scalar() && self.typ.is_f64_derived() {
                Ok(self.member_data.as_f64())
            }
            else {
                Err(anyhow!("Member cannot be cast to single f64 (type was {})", self.typ))
            }
        }
    }

    /// Interpret the data as a slice of double values
    pub fn as_f64_slice(&mut self) -> Result<&mut [f64]> {
        // Anything can be cast to a slice as long as it is based on f64 values
        if self.typ.is_f64_derived() {
            Ok(self.member_data.as_f64_slice_mut())
        }
        else {
            Err(anyhow!("Member cannot be cast to slice of f64 (type was {})", self.typ))
        }
    }

    /// Interpret the data as a slice of double values
    pub fn as_i64_slice(&mut self) -> Result<&mut [i64]> {
        // Anything can be cast to a slice as long as it is based on f64 values
        if self.typ.is_i64_derived() {
            Ok(self.member_data.as_i64_slice_mut())
        }
        else {
            Err(anyhow!("Member cannot be cast to slice of i64 (type was {})", self.typ))
        }
    }
}