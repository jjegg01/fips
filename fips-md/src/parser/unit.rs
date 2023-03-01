//! A single unit of FIPS code (typically one source file)

use super::{ExternFunctionDecl, Interaction, Particle, Simulation, GlobalStateMember};

pub struct Unit {
    pub(crate) global_state_members: Vec<GlobalStateMember>,
    pub(crate) particles: Vec<Particle>,
    pub(crate) interactions: Vec<Interaction>,
    pub(crate) simulations: Vec<Simulation>,
    pub(crate) extern_functiondecls: Vec<ExternFunctionDecl>
}

impl Unit {
    pub fn new() -> Self {
        Unit {
            global_state_members: vec![],
            particles: vec![],
            interactions: vec![],
            simulations: vec![],
            extern_functiondecls: vec![],
        }
    }
}

/// All top level tokens of FIPS
#[derive(Debug,PartialEq)]
pub(crate) enum UnitMember {
    GlobalStateMember(GlobalStateMember),
    Particle(Particle),
    Interaction(Interaction),
    Simulation(Simulation),
    ExternFunctionDecl(ExternFunctionDecl)
}