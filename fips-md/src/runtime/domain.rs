//! Structures related to defining a simulation domain

#[derive(Clone)]
pub enum Domain {
    Dim2{x: Axis, y:Axis}, Dim3{x: Axis, y:Axis, z:Axis}
}

impl Domain {
    pub fn get_dim(&self) -> usize {
        match self {
            Domain::Dim2{..} => 2,
            Domain::Dim3{..} => 3,
        }
    }
}

/// Definition of a single coordinate axis (i.e. bounds and out-of-bounds behavior)
#[derive(Clone)]
pub struct Axis {
    pub low: f64,
    pub high: f64,
    pub oob: OutOfBoundsBehavior
}

/// Definition of the out-of-bounds behavior
#[derive(Clone)]
pub enum OutOfBoundsBehavior {
    Periodic
}

impl Axis {
    pub(crate) fn size(&self) -> f64 {
        self.high - self.low
    }

    #[inline(always)]
    pub(crate) fn is_on_axis(&self, x: f64) -> bool {
        x >= self.low && x <= self.high
    }
}
