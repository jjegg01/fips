//! External function declarations

use super::FipsType;

#[derive(Debug,PartialEq)]
pub struct ExternFunctionDecl {
    pub name: String,
    pub parameter_types: Vec<FipsType>,
    pub return_type: FipsType
}