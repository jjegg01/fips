//! Structs relating to the particle, interaction and simulation indices

use std::{collections::{HashMap, HashSet}};

use slotmap::{SlotMap, DefaultKey};
use anyhow::{Result, anyhow};
use strum::IntoEnumIterator;
use strum_macros::{EnumIter};

use crate::parser::{self, CompileTimeConstant, Identifier};
use crate::parser::ConstantSubstitution;

pub type ParticleID = DefaultKey;
pub type SimulationID = DefaultKey;
pub type InteractionID = DefaultKey;
pub type InteractionQuantityID = DefaultKey;
pub type FunctionID = DefaultKey;

/* -- General index structure -- */

/// Combination of a slotmap with a hashmap for easy name resolution
#[derive(Debug)]
struct Index<ID: slotmap::Key, T> {
    /// Internal registry for holding indexed elements of type T
    elements: SlotMap<ID, T>,
    /// Fast lookup table for names
    name_table: HashMap<String, ID>
}

impl<ID: slotmap::Key, T> Index<ID,T> {
    pub fn new() -> Self {
        Self {
            elements: SlotMap::with_key(),
            name_table: HashMap::new()
        }
    }

    pub fn insert(&mut self, element: T, name: String) -> Option<T> {
        // Try to remove any existing element with the same name
        let result = match self.name_table.remove(&name) {
            Some(key) => Some(self.elements.remove(key).unwrap()),
            None => None
        };
        // Insert new element
        let key = self.elements.insert(element);
        self.name_table.insert(name, key);
        result
    }

    pub fn get(&self, id: &ID) -> Option<&T> {
        self.elements.get(*id)
    }

    pub fn get_with_name(&self, name: &str) -> Option<(ID, &T)> {
        let key = match self.name_table.get(name) {
            None => return None,
            Some(key) => key
        };
        Some((*key, self.elements.get(*key).unwrap()))
    }

    pub fn iter(&self) -> impl Iterator<Item=(ID, &T)> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(ID, &mut T)> {
        self.elements.iter_mut()
    }
}

/* -- Particle index -- */

/// Central storage for all particle metadata
pub struct ParticleIndex {
    /// Internal registry
    index: Index<ParticleID, ParticleIndexEntry>
}

impl ParticleIndex {
    /// Construct new particle index from parsed particle data
    pub(crate) fn new(mut parsed_particles: Vec<parser::Particle>) -> Result<Self> {
        // Create internal registry
        let mut particle_index = Self { index: Index::new() };
        // Try to insert all particles
        while !parsed_particles.is_empty() {
            while let Some(particle) = parsed_particles.pop() {
                let name = particle.name.clone();
                // Abort, if particle was redefined
                match particle_index.index.insert(ParticleIndexEntry::new(particle, &particle_index)?, name) {
                    None => {},
                    Some(entry) => { return Err(
                        anyhow!("Particle {} illegally redefined", entry.get_name())
                    )}
                }
            }
        }
        // Fill registry with parsed particles
        
        Ok(particle_index)
    }

    pub(crate) fn get(&self, particle_id: ParticleID) -> Option<&ParticleIndexEntry> {
        self.index.get(&particle_id)
    }

    pub(crate) fn get_particle_by_name(&self, particle_name: &str) -> Option<(ParticleID, &ParticleIndexEntry)> {
        self.index.get_with_name(particle_name)
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        for (_, particle_definition) in &mut self.index.iter_mut() {
            particle_definition.substitute_constant(name, value)?
        }
        Ok(())
    }

}

pub type MemberID = DefaultKey;

/// Index entry for a single particle type
#[derive(Debug)]
pub struct ParticleIndexEntry {
    /// Parsed particle information (members are stripped from this)
    particle: parser::Particle,
    /// Internal registry holding member data
    member_index: Index<MemberID, MemberIndexEntry>
}

impl ParticleIndexEntry {
    /// Construct new particle index entry from parsed particle
    pub(crate) fn new(mut parsed_particle: parser::Particle, _particle_index: &ParticleIndex) -> Result<Self> {
        // Create internal registry
        let mut member_index = Index::new();
        // Fill registry
        for member in parsed_particle.members {
            let name = member.name.clone();
            // Abort, if particle was redefined
            match member_index.insert(MemberIndexEntry::new(member), name) {
                None => {},
                Some(entry) => { return Err(
                    anyhow!("Member {} of particle {} illegally redefined",
                        entry.get_name(), parsed_particle.name)
                )}
            }
        }
        // Fill void left by taking all the members
        parsed_particle.members = vec![];
        Ok(ParticleIndexEntry {
            particle: parsed_particle,
            member_index
        })
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        for (_, member_definition) in &mut self.member_index.iter_mut() {
            member_definition.substitute_constant(name, value)?
        }
        Ok(())
    }

    pub fn get_members(&self) -> impl Iterator<Item = (MemberID, &MemberIndexEntry)> {
        self.member_index.iter()
    }

    pub fn get_name(&self) -> &str {
        &self.particle.name
    }

    pub fn get_member_by_name<'a>(&'a self, member_name: &str) -> Option<(MemberID, &'a MemberIndexEntry)> {
        self.member_index.get_with_name(member_name)
    }

    // TODO: this should never fail after passing verification
    pub fn get_position_member<'a>(&'a self) -> Option<(MemberID, &'a MemberIndexEntry)> {
        for (member_id, member) in self.member_index.iter() {
            if member.is_position() {
                return Some((member_id, member));
            }
        }
        None
    }

    pub fn get_member<'a>(&'a self, member_id: &MemberID) -> Option<&'a MemberIndexEntry> {
        self.member_index.get(member_id)
    }
}

/// Index entry for a single particle member
#[derive(Debug, Clone)]
pub struct MemberIndexEntry {
    member: parser::ParticleMember
}

impl MemberIndexEntry {
    pub(crate) fn new(parsed_member: parser::ParticleMember) -> Self {
        Self {
            member: parsed_member
        }
    }

    pub(crate) fn get_member_size(&self) -> Result<usize> {
        self.member.typ.get_size()
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        self.member.substitute_constant(name, value)
    }

    pub(crate) fn is_mutable(&self) -> bool {
        self.member.mutable
    }

    pub(crate) fn is_position(&self) -> bool {
        self.member.is_position
    }

    pub(crate) fn get_type(&self) -> &parser::FipsType {
        &self.member.typ
    }

    pub(crate) fn get_name(&self) -> &str {
        &self.member.name
    }
}

/* -- Simulation index -- */

/// Index of simulations
pub struct SimulationIndex {
    index: Index<SimulationID, SimulationIndexEntry>
}

impl SimulationIndex {
    /// Construct new simulation index from parsed particle data
    pub(crate) fn new(parsed_simulations: Vec<parser::Simulation>,
        particle_index: &ParticleIndex)
    -> Result<Self> {
        // Create internal registry
        let mut index = Index::new();
        // Fill registry with parsed simulations
        for simulation in parsed_simulations {
            let name = simulation.name.clone();
            // Abort, if particle was redefined
            match index.insert(SimulationIndexEntry::new(simulation, particle_index)?, name) {
                None => {},
                Some(entry) => { return Err(
                    anyhow!("Particle {} illegally redefined", entry.get_name())
                )}
            }
        }
        Ok(Self { index })
    }

    pub(crate) fn get_simulation_by_name(&self, name: &str) -> Option<(SimulationID, &SimulationIndexEntry)> {
        self.index.get_with_name(name)
    }

    pub(crate) fn get_simulation(&self, simulation_id: &SimulationID) -> Option<&SimulationIndexEntry> {
        self.index.get(simulation_id)
    }

    // pub(crate) fn iter(&self) -> impl Iterator<Item = (SimulationID, &SimulationIndexEntry)> {
    //     self.index.iter()
    // }

    // pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
    //     for (_, simulation_definition) in &mut self.index.iter_mut() {
    //         simulation_definition.substitute_constant(name, value)?
    //     }
    //     Ok(())
    // }
}

pub struct SimulationIndexEntry {
    /// Parsed simulation (stripped of all statement blocks)
    simulation: parser::Simulation,
    /// Resolved ID of the default particle
    default_particle: Option<ParticleID>,
    /// Simulation blocks
    blocks: Vec<SimulationBlock>
}

impl SimulationIndexEntry {
    pub(crate) fn new(mut simulation: parser::Simulation,
        particle_index: &ParticleIndex)
    -> Result<Self> {
        // Resolve default particle
        let default_particle = simulation.default_particle.as_ref()
            .map(|default_particle_name| 
                particle_index.get_particle_by_name(&default_particle_name)
                    .map(|(particle_id,_)| particle_id)
                    .ok_or(anyhow!("Cannot find default particle {} for simulation {}",
                        default_particle_name, simulation.name))
            ).transpose()?;
        // Extract simulation blocks (swap in empty vector)
        let blocks = simulation.blocks;
        simulation.blocks = vec![];
        // Transform parser::SimulationBlock (close to syntax) to
        // SimulationBlock (more useful for analysis)
        let blocks = blocks.into_iter()
            .map(|simulation_block| {
                // Flatten parser::SimulationBlock enum to single kind field
                let (kind, subblocks) = match simulation_block {
                    parser::SimulationBlock::Once(once_block) => (
                        SimulationBlockKind::Once(once_block.step),
                        once_block.subblocks
                    ),
                    parser::SimulationBlock::Step(step_block) => (
                        SimulationBlockKind::Step(step_block.step_range),
                        step_block.subblocks
                    ),
                };
                // Extract statement blocks
                let statement_blocks = subblocks.into_iter().map(|subblock| {
                    let particle_filter = match subblock.particle {
                        None => SimulationParticleFilter::Default,
                        Some(particle_name) => {
                            // Resolve particle references
                            let (particle_id,_) = particle_index.get_particle_by_name(&particle_name)
                                .ok_or(anyhow!("Cannot find particle {} referenced in simulation {}", 
                                    particle_name, simulation.name))?;
                            SimulationParticleFilter::Single(particle_id)
                        }
                    };
                    Ok(SimulationStatementBlock {
                        particle_filter,
                        statements: subblock.statements
                    })
                }).collect::<Result<Vec<_>>>()?;
                Ok(SimulationBlock {
                    kind, statement_blocks
                })
        }).collect::<Result<Vec<_>>>()?;
        // Construct simulation entry
        Ok(Self {
            simulation,
            default_particle,
            blocks
        })
    }

    pub(crate) fn get_name(&self) -> &str {
        &self.simulation.name
    }

    pub(crate) fn get_default_particle(&self) -> Option<ParticleID> {
        self.default_particle
    }

    // pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
    //     self.simulation.substitute_constant(name, value)
    // }

    pub(crate) fn get_blocks(&self) -> &[SimulationBlock] {
        &self.blocks
    }
}

/// Kind of simulation block
pub enum SimulationBlockKind {
    /// Block that is executed only for one time step
    Once(parser::CompileTimeConstant<usize>),
    /// Block that is executed in multiple time steps
    Step(parser::StepRange)
}

/// A simulation block, i.e. some instructions that are executed together in the
/// same time step
pub struct SimulationBlock {
    /// Kind of simulation block
    pub kind: SimulationBlockKind,
    /// Blocks with simulation instructions
    pub statement_blocks: Vec<SimulationStatementBlock>
}

/// Particles affected by a statement block
pub enum SimulationParticleFilter {
    /// Statement block only affects the default particle 
    Default,
    /// Statement blocks affects a single particle
    Single(ParticleID)
}

pub struct SimulationStatementBlock {
    /// Filter for particles affected by this statement block
    pub particle_filter: SimulationParticleFilter,
    /// Actual statements
    pub statements: Vec<parser::Statement>
}

/* -- Interaction index -- */

pub struct InteractionIndex {
    index: Index<InteractionID, InteractionIndexEntry>
}

impl InteractionIndex {
    pub(crate) fn new(parsed_interactions: Vec<parser::Interaction>, 
        particle_index: &ParticleIndex) -> Result<Self> 
    {
        // Create internal registry
        let mut index = Index::new();
        // Fill registry with parsed simulations
        for interaction in parsed_interactions {
            let name = interaction.name.clone();
            // Abort, if particle was redefined
            match index.insert(InteractionIndexEntry::new(interaction, particle_index)?, name) {
                None => {},
                Some(entry) => { return Err(
                    anyhow!("Interaction {} illegally redefined", entry.get_name())
                )}
            }
        }
        Ok(Self { index })
    }

    pub(crate) fn get_interaction_by_name(&self, interaction_name: &str) -> Option<(InteractionID, &InteractionIndexEntry)> {
        self.index.get_with_name(interaction_name)
    }

    pub(crate) fn get(&self, interaction_id: InteractionID) -> Option<&InteractionIndexEntry> {
        self.index.get(&interaction_id)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item=(InteractionID, &InteractionIndexEntry)> {
        self.index.iter()
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        for (_, interaction_definition) in &mut self.index.iter_mut() {
            interaction_definition.substitute_constant(name, value)?
        }
        Ok(())
    }
}

pub struct InteractionIndexEntry {
    /// Parser result (quantities are stripped from this)
    interaction: parser::Interaction,
    /// Resolved particle ID of interacting particle A
    particle_id_a: ParticleID,
    /// Resolved particle ID of interacting particle B
    particle_id_b: ParticleID,
    /// Index of quantities
    quantity_index: Index<InteractionQuantityID, InteractionQuantityIndexEntry>
}

impl InteractionIndexEntry {
    pub(crate) fn new(mut interaction: parser::Interaction, particle_index: &ParticleIndex) -> Result<Self> {
        // Create internal registry
        let mut quantity_index = Index::new();
        // Fill registry
        for quantity in interaction.quantities {
            let name = quantity.name.clone();
            // Abort, if quantity was redefined
            match quantity_index.insert(InteractionQuantityIndexEntry::new(quantity), name) {
                None => {},
                Some(entry) => { return Err(
                    anyhow!("Quantitiy {} of interaction {} illegally redefined",
                        entry.get_name(), interaction.name)
                )}
            }
        }
        // Fill void left by taking all the quantities
        interaction.quantities = vec![];
        // Resolve particle ids
        let particle_id_a = particle_index.get_particle_by_name(&interaction.type_a)
            .ok_or(anyhow!("Cannot resolve particle type {} in interaction {}",
                &interaction.type_a, interaction.name))?.0;
        let particle_id_b = particle_index.get_particle_by_name(&interaction.type_b)
            .ok_or(anyhow!("Cannot resolve particle type {} in interaction {}",
                &interaction.type_a, interaction.name))?.0;
        // Create entry
        Ok(Self {
            interaction,
            quantity_index,
            particle_id_a, particle_id_b
        })
    }

    /// Get name of interaction
    pub(crate) fn get_name(&self) -> &str {
        &self.interaction.name
    }

    pub(crate) fn get_type_a(&self) -> &str {
        &self.interaction.type_a
    }

    pub(crate) fn get_type_b(&self) -> &str {
        &self.interaction.type_b
    }

    pub(crate) fn get_name_a(&self) -> &Identifier {
        &self.interaction.name_a
    }

    pub(crate) fn get_name_b(&self) -> &Identifier {
        &self.interaction.name_b
    }

    pub(crate) fn get_distance_vec(&self) -> Option<&String> {
        self.interaction.distance_vec.as_ref()
    }

    pub(crate) fn get_quantity(&self, quantity_id: InteractionQuantityID) -> Option<&InteractionQuantityIndexEntry> {
        self.quantity_index.get(&quantity_id)
    }

    pub(crate) fn get_cutoff(&self) -> CompileTimeConstant<f64> {
        self.interaction.cutoff.clone()
    }

    pub(crate) fn get_distance_identifier(&self) -> &Identifier {
        &self.interaction.distance
    }

    pub(crate) fn get_common_block(&self) -> Option<&Vec<parser::Statement>> {
        self.interaction.common_block.as_ref()
    }

    /// Get interaction quantity by name
    pub(crate) fn get_quantity_by_name(&self, quantity: &str) -> Option<(InteractionQuantityID, &InteractionQuantityIndexEntry)> {
        self.quantity_index.get_with_name(quantity)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (InteractionQuantityID, &InteractionQuantityIndexEntry)> {
        self.quantity_index.iter()
    }

    /// Get a list of all particles affected by this interaction
    pub(crate) fn get_affected_particles(&self, _particle_index: &ParticleIndex) -> Result<HashSet<ParticleID>> {
        let mut affected_particles = HashSet::new();
        // The direct types are affected for sure
        affected_particles.insert(self.particle_id_a);
        affected_particles.insert(self.particle_id_b);
        // In the future we might have particle polymorphism again...
        Ok(affected_particles)
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        self.interaction.cutoff.substitute_constant(name, value)
    }
}

pub struct InteractionQuantityIndexEntry {
    quantity: parser::InteractionQuantity
}

impl InteractionQuantityIndexEntry {
    /// Construct new quantity index entry from parsed particle
    pub(crate) fn new(quantity: parser::InteractionQuantity) -> Self {
        Self {
            quantity
        }
    }

    pub(crate) fn get_name(&self) -> &str {
        &self.quantity.name
    }

    pub(crate) fn get_target_a(&self) -> &str {
        &self.quantity.target_a
    }

    pub(crate) fn get_target_b(&self) -> &str {
        &self.quantity.target_b
    }

    pub(crate) fn get_expression(&self) -> &parser::Expression {
        &self.quantity.expression
    }

    pub(crate) fn get_reduction_method(&self) -> &parser::ReductionMethod {
        &self.quantity.reduction_method
    }

    pub(crate) fn get_symmetry(&self) -> parser::InteractionSymmetry {
        self.quantity.symmetry
    }
}


/* -- Function index -- */

/// Index of all global functions (including builtins!)
pub(crate) struct FunctionIndex {
    index: Index<FunctionID, FunctionIndexEntry>
}

impl FunctionIndex {
    /// Create new function index (automatically instantiates builtins)
    pub(crate) fn new(externs: Vec<parser::ExternFunctionDecl>) -> Result<Self> {
        let mut index = Index::new();
        for builtin in BuiltinFunction::get_all() {
            let name = builtin.get_name().to_string();
            match index.insert(FunctionIndexEntry::Builtin(builtin), name) {
                Some(entry) => { return Err(anyhow!("Internal: builtin name {} duplicated", entry.get_name())) },
                None => {},
            }
        }
        for externfunc in externs {
            let name = externfunc.name.clone();
            match index.insert(FunctionIndexEntry::Extern(externfunc), name) {
                Some(entry) => { return Err(anyhow!("Cannot redefine function {}", entry.get_name())) },
                None => {},
            }
        }

        Ok(Self {
            index
        })
    }

    pub(crate) fn get(&self, function_id: FunctionID) -> Option<&FunctionIndexEntry> {
        self.index.get(&function_id)
    }

    pub(crate) fn get_functions(&self) -> impl Iterator<Item = (FunctionID, &FunctionIndexEntry)> {
        self.index.iter()
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        for (_, function_def) in &mut self.index.iter_mut() {
            function_def.substitute_constant(name, value)?
        }
        Ok(())
    }
}

pub(crate) enum FunctionIndexEntry {
    Extern(parser::ExternFunctionDecl),
    Builtin(BuiltinFunction)
}

impl FunctionIndexEntry {
    pub(crate) fn get_name(&self) -> &str {
        match self {
            FunctionIndexEntry::Extern(externfunc) => { &externfunc.name },
            FunctionIndexEntry::Builtin(builtin) => { builtin.get_name() },
        }
    }

    pub(crate) fn needs_callback_target_ptr(&self) -> bool {
        match self {
            FunctionIndexEntry::Extern(_) => { false }, // Not supported currently
            FunctionIndexEntry::Builtin(builtin) => { builtin.needs_callback_target_ptr() },
        }
    }

    pub(crate) fn returns_array(&self) -> bool {
        match self {
            FunctionIndexEntry::Extern(externfunc) => { 
                match externfunc.return_type {
                    parser::FipsType::Double | parser::FipsType::Int64 => false,
                    parser::FipsType::Array { .. } => true,
                }
            },
            FunctionIndexEntry::Builtin(builtin) => { builtin.returns_array() },
        }
    }

    pub(crate) fn substitute_constant(&mut self, name: &str, value: &parser::SubstitutionValue) -> Result<()> {
        match self {
            FunctionIndexEntry::Extern(func_decl) => { func_decl.substitute_constant(name, value) },
            FunctionIndexEntry::Builtin(_) => Ok(()),
        }
    }
}

#[derive(EnumIter)]
pub(crate) enum BuiltinFunction {
    Sqrt,
    Sin,
    Cos,
    RandomNormal
}

impl BuiltinFunction {
    /// Get all builtin functions
    pub(crate) fn get_all() -> Vec<Self> {
        Self::iter().collect::<Vec<_>>()
    }

    /// Get name this function is called by
    pub(crate) fn get_name(&self) -> &'static str {
        match self {
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::RandomNormal => "random_normal",
        }
    }

    /// Some functions require the callback pointer as first argument (e.g. random functions)
    /// The caller must inject this pointer into the parameter list!
    pub(crate) fn needs_callback_target_ptr(&self) -> bool {
        match self {
            BuiltinFunction::Sqrt => false,
            BuiltinFunction::Sin => false,
            BuiltinFunction::Cos => false,
            BuiltinFunction::RandomNormal => true,
        }
    }

    pub(crate) fn returns_array(&self) -> bool {
        match self {
            Self::Sqrt => false,
            Self::Sin => false,
            Self::Cos => false,
            Self::RandomNormal => false,
        }
    }
}


    // // If true, the function returns via its last parameter (the call just returns void)
    // pub(crate) returns_array: bool