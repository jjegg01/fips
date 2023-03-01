//! Structures related to typing

use std::fmt::Display;

use super::CompileTimeConstant;

/// A type in the FIPS language
#[derive(Clone,Eq,PartialEq,Debug)]
pub enum FipsType {
    Double,
    Int64,
    //Position, // This is an alias for [f64; NDIM]
    Array{
        typ: Box<FipsType>,
        length: CompileTimeConstant<usize>
    }
}

impl FipsType {
    /// Returns true if type is scalar
    pub(crate) fn is_scalar(&self) -> bool {
        match self {
            FipsType::Double | FipsType::Int64 => true,
            FipsType::Array {..} => false
        }
    }

    /// Returns true if the type is derived from i64 (i.e. i64 or [i64] or [[i64]] ...)
    pub(crate) fn is_i64_derived(&self) -> bool {
        match self {
            FipsType::Int64 => true,
            FipsType::Double => false,
            FipsType::Array { typ, ..} => typ.is_i64_derived()
        }
    }

    /// Returns true if the type is derived from f64 (i.e. f64 or [f64] or [[f64]] ...)
    pub(crate) fn is_f64_derived(&self) -> bool {
        match self {
            FipsType::Int64 => false,
            FipsType::Double => true,
            FipsType::Array { typ, ..} => typ.is_f64_derived()
        }
    }
}

impl Display for FipsType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FipsType::Double => f.write_str("f64"),
            FipsType::Int64 => f.write_str("i64"),
            FipsType::Array { typ, length } => f.write_fmt(format_args!("[{},{}]", typ, length))
        }
    }
}