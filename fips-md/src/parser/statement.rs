//! Representation of FIPS statements

use super::{CompileTimeConstant, Expression, FipsType};

/// All kinds of statements in Rust
#[derive(Clone,PartialEq,Debug)]
pub enum Statement {
    Let(LetStatement),
    Assign(AssignStatement),
    Update(UpdateStatement),
    Call(CallStatement)
}

/// Statement for introducing a new binding
#[derive(Clone,PartialEq,Debug)]
pub struct LetStatement {
    pub name: String,
    pub typ: FipsType,
    pub initial: Expression
}

/// Statement for assigning a value to an existing binding
#[derive(Clone,PartialEq,Debug)]
pub struct AssignStatement {
    pub assignee: String,
    pub value: Expression,
    pub index: Option<CompileTimeConstant<usize>>
}

/// Statement for updating an interaction quantity
#[derive(Clone,PartialEq,Debug)]
pub struct UpdateStatement {
    pub interaction: String,
    pub quantity: Option<String>
}

/// Statement for calling back to Rust
#[derive(Clone,PartialEq,Debug)]
pub struct CallStatement {
    pub name: String
}