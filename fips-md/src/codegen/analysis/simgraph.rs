//! Sort of intermediate representation of the simulation code as a set of timelines
//! for every particle type connected by synchronization barriers

use anyhow::{anyhow, Result};
use slotmap::SlotMap;

use std::{collections::{BTreeSet, HashMap, HashSet}, usize};

use crate::{codegen::util, parser, parser::{Statement}, runtime::{InteractionID, InteractionQuantityID, ParticleID, SimulationBlockKind, SimulationID, SimulationParticleFilter}};

use super::{Barrier, BarrierID, BarrierKind, SymbolTable};

/// A simulation graph consists of a timeline for every type of particle
/// Timelines can be connected by common synchronization barriers due to mutual
/// interactions
pub struct SimulationGraph {
    pub timelines: HashMap<ParticleID, Timeline>,
    pub barriers: SlotMap<BarrierID, Barrier>,
    pub callbacks: HashMap<String, BTreeSet<BarrierID>>
}

/// A timeline is a chain of simulation nodes for a single particle type
pub struct Timeline {
    pub(crate) nodes: Vec<SimulationNode>,
    pub(crate) particle_symbols: SymbolTable<()>
}

impl Timeline {
    pub fn new(nodes: Vec<SimulationNode>, particle_symbols: SymbolTable<()>) -> Self {
        Self { nodes, particle_symbols }
    }
}

/// Internal proto-simgraph created during construction
struct UnresolvedTimeline {
    nodes: Vec<MaybeResolvedSimNode>
}

/// A single simulation node that is considered atomic for synchronization 
/// purposes
pub enum SimulationNode {
    /// A block of statements
    StatementBlock(StatementBlock),
    /// A shared barrier between multiple timelines
    CommonBarrier(BarrierID)
}

/// Simulation node that might not have been resolved yet
enum MaybeResolvedSimNode {
    /// Resolved node
    Resolved(SimulationNode),
    /// Unresolved barrier caused by interaction update
    UnresolvedInteractionBarrier(InteractionID, Option<InteractionQuantityID>),
    /// Unresolved barrier caused by call to Rust
    UnresolvedCallBarrier(String)
}

#[derive(Clone)]
pub struct StepRange {
    start: usize,
    end: usize,
    step: usize
}

pub struct StatementBlock {
    pub step_range: StepRange,
    pub statements: Vec<parser::Statement>,
    pub local_symbols: SymbolTable<()>
}

impl SimulationGraph {
    pub(crate) fn new<'a>(active_particles: HashSet<ParticleID>,
        simulation_id: SimulationID,
        indices: util::IndicesRef<'a>
    ) -> Result<Self> {
        // Convert active particles to set
        let active_particles = active_particles.iter()
            .map(|x| *x ).collect::<HashSet<_>>();
        // Construct unresolved simulation graph
        let mut unresolved_graph = active_particles.iter()
            .map(|particle_id| {
                let timeline = UnresolvedTimeline::new(*particle_id, simulation_id, indices);
                timeline.map(|timeline| (*particle_id, timeline))
            })
            .collect::<Result<Vec<_>>>()?;
        // Collect all interactions from the graph
        let mut used_interactions = HashSet::new();
        for (_, timeline) in &unresolved_graph {
            for node in &timeline.nodes {
                match node {
                    MaybeResolvedSimNode::UnresolvedInteractionBarrier(interaction_id, _) => {
                        used_interactions.insert(*interaction_id);
                    }
                    MaybeResolvedSimNode::Resolved(_) |
                    MaybeResolvedSimNode::UnresolvedCallBarrier(_) => {}
                }
            }
        }
        // Create hashmap of affected particles for every interaction
        let affected_particles_map = used_interactions.iter()
            .map(|interaction_id| {
                // Unwrap is safe due to interaction ids being resolved previously
                indices.interactions.get(*interaction_id).unwrap()
                    .get_affected_particles(indices.particles)
                    // Map successful result to intersection with the list of active particles
                    .map(|affected_particles| (
                        *interaction_id,
                        affected_particles.intersection(&active_particles)
                            .map(|x| *x).collect::<HashSet<_>>()
                    ))
            })
            .collect::<Result<HashMap<_,_>>>()?;
        // Initialize barrier and callback maps
        let mut barriers: SlotMap<BarrierID, Barrier> = SlotMap::new();
        let mut callbacks: HashMap<String, BTreeSet<BarrierID>> = HashMap::new();
        let mut timelines: HashMap<ParticleID, Timeline> = HashMap::new();
        // Resolve all unresolved nodes for every timeline
        while let Some((particle_id, timeline)) = unresolved_graph.pop() {
            // Construct particle symbol table
            let particle_def = indices.particles.get(particle_id).unwrap();
            let particle_symbols = SymbolTable::from_particle(particle_def);
            // Resolve nodes
            let mut resolved_nodes = vec![];
            for node in timeline.nodes {
                resolved_nodes.push(match node {
                    // Node already resolved => Passthrough
                    MaybeResolvedSimNode::Resolved(node) => node,
                    // Interaction barrier? Try to find all other barriers
                    MaybeResolvedSimNode::UnresolvedInteractionBarrier(interaction_id, quantity_id) => {
                        // Create a new barrier
                        let affected_particles = affected_particles_map
                            .get(&interaction_id).unwrap().clone();
                        let barrier = Barrier::new(affected_particles,
                            BarrierKind::InteractionBarrier(interaction_id, quantity_id));
                        // Register barrier
                        let barrier_id = barriers.insert(barrier);
                        let affected_particles = &barriers.get(barrier_id)
                            .unwrap().affected_particles; // "Trivially" safe unwrap
                        // Resolve corresponding nodes in other timelines
                        for (other_particle, other_timeline) in unresolved_graph.iter_mut()
                            // Filter for particles affected by the same interaction
                            .filter(|(other_particle,_)| affected_particles.contains(other_particle))
                        {
                            // If interaction and quantity id match, replace node with this barrier
                            let mut found = false;
                            for other_node in &mut other_timeline.nodes {
                                if let MaybeResolvedSimNode::UnresolvedInteractionBarrier(other_interaction_id, other_quantity_id) = other_node {
                                    if interaction_id == *other_interaction_id && quantity_id == *other_quantity_id {
                                        *other_node = MaybeResolvedSimNode::Resolved(SimulationNode::CommonBarrier(barrier_id));
                                        // Break from loop (we only want to replace the first matching node)
                                        found = true;
                                        break
                                    }
                                }
                            }
                            // Return error if no matching update statement is found for a particle type that interacts with
                            // the current particle type
                            if !found {
                                return Err(anyhow!(
                                    "No matching update statement for interaction quantity {}:{} found for particle {} (required by interaction with particle {})",
                                    indices.interactions.get(interaction_id).unwrap().get_name(),
                                    match quantity_id {
                                        None => "",
                                        Some(quantity_id) => indices.interactions.get(interaction_id).unwrap().get_quantity(quantity_id).unwrap().get_name(),
                                    },
                                    indices.particles.get(*other_particle).unwrap().get_name(),
                                    indices.particles.get(particle_id).unwrap().get_name(),
                                ));
                            }
                        }
                        SimulationNode::CommonBarrier(barrier_id)
                    }
                    // Call barrier? Try to find corresponding call barriers in other timelines
                    // or add one at the end of every other timeline
                    MaybeResolvedSimNode::UnresolvedCallBarrier(call_name) => {
                        // Create a new barrier (calls affect all particles)
                        let barrier = Barrier::new(active_particles.clone(),
                            BarrierKind::CallBarrier(call_name.clone()));
                        // Register barrier
                        let barrier_id = barriers.insert(barrier);
                        match callbacks.get_mut(&call_name) {
                            Some(barriers) => {
                                barriers.insert(barrier_id);
                            }
                            None => {
                                let mut barriers = BTreeSet::new();
                                barriers.insert(barrier_id);
                                callbacks.insert(call_name.clone(), barriers);
                            }
                        }
                        // Find barriers in other timelines
                        for (_, other_timeline) in unresolved_graph.iter_mut() {
                            let mut found = false;
                            for other_node in &mut other_timeline.nodes {
                                if let MaybeResolvedSimNode::UnresolvedCallBarrier(other_call_name) = other_node {
                                    if *other_call_name == call_name {
                                        *other_node = MaybeResolvedSimNode::Resolved(SimulationNode::CommonBarrier(barrier_id));
                                        found = true;
                                        break
                                    }
                                }
                            }
                            // If not found, insert new call barrier at the end
                            if !found {
                                other_timeline.nodes.push(
                                    MaybeResolvedSimNode::Resolved(SimulationNode::CommonBarrier(barrier_id))
                                )
                            }
                        }
                        SimulationNode::CommonBarrier(barrier_id)
                    }
                })
            }
            // Push resolved nodes to timelines
            timelines.insert(particle_id, Timeline::new(resolved_nodes, particle_symbols));
        }

        // for (pid, timeline) in &timelines {
        //     dbg!(indices.particles.get(*pid).unwrap().get_name());
        //     for node in &timeline.nodes {
        //         match node {
        //             SimulationNode::StatementBlock(_) => {println!("STATEMENT BLOCK")},
        //             SimulationNode::CommonBarrier(_) => {println!("BARRIER")},
        //         }
        //     }
        // }

        Ok(Self {
            timelines,
            barriers,
            callbacks
        })
    }
}

impl UnresolvedTimeline {
    pub(crate) fn new<'a>(particle_id: ParticleID,
        simulation_id: SimulationID,
        indices: util::IndicesRef<'a>
    ) -> Result<Self> {
        // Unwrap is safe due to caller (Runtime::compile) resolving simulation before
        let simulation = indices.simulations.get_simulation(&simulation_id).unwrap();
        // Resolve default particle (if any exists)
        let default_particle_id = simulation.get_default_particle();
        // Create a timeline for every particle type
        let mut nodes = vec![];
        for simblock in simulation.get_blocks() {
            // Canonize step ranges from step and once blocks
            let step_range = match &simblock.kind {
                // Once block: step range is <only step>:<only step + 1>:1
                SimulationBlockKind::Once(step) => {
                    let step = util::unwrap_usize_constant(step)?;
                    StepRange::new(step, step+1, 1)
                }
                // Step block: step range is <start or 0>:<end or MAX>:<step>
                SimulationBlockKind::Step(step_range) => {
                    let (start,end,step) = (
                        util::unwrap_usize_constant(&step_range.start)?,
                        util::unwrap_usize_constant(&step_range.end)?,
                        util::unwrap_usize_constant(&step_range.step)?
                    );
                    StepRange::new(start, end, step)
                }
            };
            // Create new statement block and fill it with statements
            let mut current_node = vec![];
            // Quick macro for pushing node to timeline and creating a fresh one
            let push_node = |current_node: Vec<Statement>, nodes: &mut Vec<MaybeResolvedSimNode>| -> Result<Vec<Statement>> {
                if !current_node.is_empty() {
                    nodes.push(MaybeResolvedSimNode::Resolved(
                        SimulationNode::StatementBlock(
                            StatementBlock::new(step_range.clone(),current_node)?)));
                }
                Ok(vec![])
            };
            for statement_block in &simblock.statement_blocks {
                // Check if subblock affects this particle
                let affected = match &statement_block.particle_filter {
                    // Check default particle
                    SimulationParticleFilter::Default => match default_particle_id {
                        None => false, // TODO: Should we issue a warning here?
                        Some(default_particle_id) => particle_id == default_particle_id
                    },
                    SimulationParticleFilter::Single(filter_particle_id) => 
                        particle_id == *filter_particle_id
                };
                // Add statements if particle is affected
                if affected {
                    for statement in &statement_block.statements {
                        match &statement {
                            // Let and assign statements just get added to the current node
                            Statement::Let(_) | Statement::Assign(_) => { 
                                current_node.push(statement.clone()) 
                            }
                            // Update statements cause a barrier
                            Statement::Update(update_statement) => {
                                let interaction_name = &update_statement.interaction;
                                // Try to resolve interaction (and quantity)
                                let (interaction_id, interaction) = indices.interactions.get_interaction_by_name(&interaction_name)
                                    .ok_or(anyhow!("Cannot resolve interaction with name {}", &interaction_name))?;
                                let quantity_id = match &update_statement.quantity {
                                    None => None,
                                    Some(quantity_name) => Some(interaction.get_quantity_by_name(&quantity_name)
                                        .ok_or(anyhow!("Cannot resolve quantity {} of interaction {}",
                                            quantity_name, interaction_name))?.0)
                                };
                                // Push old statement block and barrier for update
                                current_node = push_node(current_node, &mut nodes)?;
                                nodes.push(MaybeResolvedSimNode::UnresolvedInteractionBarrier(
                                    interaction_id, quantity_id
                                ))
                            }
                            // Call statements as well
                            Statement::Call(call_statement) => {
                                // Push old statement block and barrier for call
                                current_node = push_node(current_node, &mut nodes)?;
                                nodes.push(MaybeResolvedSimNode::UnresolvedCallBarrier(
                                    call_statement.name.clone()
                                ))
                            }
                        }
                    }
                }
            }
            push_node(current_node, &mut nodes)?;
        }
        // Return finished timeline
        Ok(Self { nodes })
    }
}

impl StatementBlock {
    pub fn new(step_range: StepRange, statements: Vec<parser::Statement>) -> Result<Self> {
        // Extract all locally defined symbols and create symbol table
        let mut local_symbols = SymbolTable::new();
        for statement in &statements {
            if let parser::Statement::Let(statement) = statement {
                local_symbols.add_local_symbol(statement.name.clone(), statement.typ.clone())?
            }
        }
        Ok(Self {
            step_range,
            statements,
            local_symbols
        })
    }
}

impl StepRange {
    pub fn new(start: usize, end: usize, step: usize) -> Self {
        Self {
            start, end, step
        }
    }
}