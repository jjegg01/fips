//! Thin wrapper around LLVM to keep the unsafe away

mod bits;
mod module;
mod context;
mod builder;
mod types;
mod value;
mod basicblock;

pub use bits::*;
pub use context::Context;
pub use module::Module;
pub use builder::IRBuilder;
pub use types::{TypeKind, Type, FunctionType};
pub use value::Value;
pub use basicblock::BasicBlock;