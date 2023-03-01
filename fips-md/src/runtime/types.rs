//! Runtime type information

use anyhow::{Result, anyhow};

use core::mem::size_of;

use crate::parser::{FipsType, CompileTimeConstant};

impl FipsType {
    pub fn get_size(&self) -> Result<usize> {
        match self {
            FipsType::Double => Ok(size_of::<f64>()),
            FipsType::Int64 => Ok(size_of::<u64>()),
            FipsType::Array { typ, length } => match length {
                CompileTimeConstant::Literal(length) | CompileTimeConstant::Substituted(length,_)
                    => Ok(typ.get_size()? * length),
                CompileTimeConstant::Identifier(name) => Err(
                    anyhow!("Cannot determine size due to array length {} being undefined", name))
            }
        }
    }
}