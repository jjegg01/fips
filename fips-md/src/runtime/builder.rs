//! Runtime builder

use anyhow::Result;

use crate::{ parser::Unit, runtime::{ Domain, Runtime }};

/// Builder for `Runtime` with default values
pub struct RuntimeBuilder {
    start_time: f64,
    time_step: f64,
    domain: Domain,
    unit: Unit
}

impl RuntimeBuilder {
    pub fn new(unit: Unit, domain: Domain) -> Self {
        Self {
            start_time: 0.0,
            time_step: 1e-4,
            domain,
            unit
        }
    }

    pub fn with_start_time(mut self, time: f64) -> Self {
        self.start_time = time;
        self
    }

    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    pub fn build(self) -> Result<Runtime> {
        Runtime::new(self.unit, self.domain, self.start_time, self.time_step)
    }
}