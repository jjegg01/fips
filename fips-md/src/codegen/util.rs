//! Small utility structures

use std::{iter, ops::Range};

use anyhow::{anyhow, Result};

use crate::{parser::CompileTimeConstant, runtime::{InteractionIndex, ParticleIndex, SimulationIndex}};

/// Compact struct for particle, simulation and interaction indices
#[derive(Clone, Copy)]
pub(crate) struct IndicesRef<'a> {
    pub particles: &'a ParticleIndex,
    pub simulations: &'a SimulationIndex,
    pub interactions: &'a InteractionIndex,
}

impl<'a> IndicesRef<'a> {
    pub(crate) fn new(
        particles: &'a ParticleIndex,
        simulations: &'a SimulationIndex,
        interactions: &'a InteractionIndex,
    ) -> Self {
        Self { particles, simulations, interactions }
    }
}

pub(crate) fn unwrap_usize_constant(constant: &CompileTimeConstant<usize>) -> Result<usize> {
    match constant {
        CompileTimeConstant::Literal(value) => Ok(*value),
        CompileTimeConstant::Identifier(name) => Err(anyhow!("Unresolved identifier {}", name)),
        CompileTimeConstant::Substituted(value, _) => Ok(*value)
    }
}

pub(crate) fn unwrap_f64_constant(constant: &CompileTimeConstant<f64>) -> Result<f64> {
    match constant {
        CompileTimeConstant::Literal(value) => Ok(*value),
        CompileTimeConstant::Identifier(name) => Err(anyhow!("Unresolved identifier {}", name)),
        CompileTimeConstant::Substituted(value, _) => Ok(*value)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct IndexRange {
    pub(crate) start: usize,
    pub(crate) end: usize
}

impl IndexRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start, end
        }
    }

    /// Split range into n roughly equal subranges
    pub fn split(&self, n: usize) -> Vec<IndexRange> {
        let len = self.end - self.start;
        let newlen = (len + n - 1)/n;
        (0..(n-1))
            .map(|i| IndexRange::new(self.start+i*newlen, self.start+(i+1)*newlen))
            .chain(iter::once(IndexRange::new(self.start+(n-1)*newlen, self.end)))
            .collect::<Vec<_>>()
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn to_range(&self, element_size: usize) -> Range<usize> {
        (self.start * element_size)..(self.end * element_size)
    }
}