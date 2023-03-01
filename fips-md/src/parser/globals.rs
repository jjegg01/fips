//! Structs related to global state members

use super::FipsType;

#[derive(Debug,PartialEq)]
pub(crate) struct GlobalStateMember {
    pub name: String,
    pub typ: FipsType,
    pub mutable: bool
}