//! Helper struct for translating between Rust values and FIPS values

use crate::parser::FipsType;

#[derive(Clone)]
pub enum FipsValue {
    Double(f64),
    Int64(i64),
    // TODO: Array values
}

impl Into<FipsValue> for f64 {
    fn into(self) -> FipsValue {
        FipsValue::Double(self)
    }
}

impl Into<FipsValue> for i64 {
    fn into(self) -> FipsValue {
        FipsValue::Int64(self)
    }
}

impl FipsValue {
    // Get the corresponding type
    pub(crate) fn get_type(&self) -> FipsType {
        match self {
            Self::Double(_) => FipsType::Double,
            Self::Int64(_) => FipsType::Int64
        }
    }

    pub(crate) fn get_size(&self) -> usize {
        match self {
            Self::Double(_) | Self::Int64(_) => std::mem::size_of::<Self>()
        }
    }
}