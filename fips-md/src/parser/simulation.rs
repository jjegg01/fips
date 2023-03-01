//! Structures related to simulation blocks

use super::{CompileTimeConstant, Statement};

/// Structure representing a simulation
#[derive(Debug,PartialEq)]
pub struct Simulation {
    pub name: String,
    pub default_particle: Option<String>,
    pub blocks: Vec<SimulationBlock>
}

/// All possible blocks inside a simulation
#[derive(Debug,PartialEq)]
pub enum SimulationBlock {
    Once(OnceBlock),
    Step(StepBlock)
}

/// Block representing a one-time only action
#[derive(Debug,PartialEq)]
pub struct OnceBlock {
    pub step: CompileTimeConstant<usize>,
    pub subblocks: Vec<SimulationSubBlock>
}

/// Block representing an action that is to be repeated according to a step range
#[derive(Debug,PartialEq)]
pub struct StepBlock {
    pub step_range: StepRange,
    pub subblocks: Vec<SimulationSubBlock>
}

/// Range of steps with start <= i < end 
#[derive(Debug,PartialEq)]
pub struct StepRange {
    pub start: CompileTimeConstant<usize>,
    pub end: CompileTimeConstant<usize>,
    pub step: CompileTimeConstant<usize>
}

impl Default for StepRange {
    fn default() -> Self {
        Self {
            start: CompileTimeConstant::Literal(0),
            end: CompileTimeConstant::Literal(usize::MAX),
            step: CompileTimeConstant::Literal(1)
        }
    }
}

/// Subblocks in a simulation that contain the actual statements
#[derive(Debug,PartialEq)]
pub struct SimulationSubBlock {
    pub particle: Option<String>,
    pub statements: Vec<Statement>
}

mod tests {
    #[allow(unused_imports)] // Rust analyzer bug
    use super::super::*;

    #[test]
    fn step_range() {
        assert!(fips_parser::step_range("1..2..3").is_err(),
            "1..2..3 not caught as invalid");
        let test_cases = std::collections::HashMap::from([
            ("..",    (0,usize::MAX,1)),
            ("1..",   (1,usize::MAX,1)),
            ("..2",   (0,2,1)),
            ("1..2",  (1,2,1)),
            ("..,3",  (0,usize::MAX,3)),
            ("1..,3", (1,usize::MAX,3)),
            ("..2,3", (0,2,3)),
            ("1..2,3",(1,2,3)),
        ]);
        for (s,(start,end,step)) in test_cases {
            assert_eq!(fips_parser::step_range(s).expect(&format!("Cannot parse {} as step range", s)),
            StepRange {
                start: start.into(),
                end: end.into(),
                step: step.into(),
            });
        }
    }
}