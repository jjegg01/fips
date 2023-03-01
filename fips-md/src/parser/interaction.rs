//! Structures representing interaction primary blocks

use super::{CompileTimeConstant, Expression, Identifier, Statement};

/// Interaction between particle types
#[derive(Clone,PartialEq,Debug)]
pub struct Interaction {
    pub name: String,
    pub name_a: Identifier,
    pub type_a: String,
    pub name_b: Identifier,
    pub type_b: String,
    pub distance: Identifier,
    pub distance_vec: Option<String>,
    pub cutoff: CompileTimeConstant<f64>,
    pub common_block: Option<Vec<Statement>>,
    pub quantities: Vec<InteractionQuantity>
}

/// Specific quantity determined by the interaction of two particle types
#[derive(Clone,PartialEq,Debug)]
pub struct InteractionQuantity {
    pub name: String,
    pub reduction_method: ReductionMethod,
    pub target_a: String,
    pub target_b: String,
    pub symmetry: InteractionSymmetry,
    // pub negated_b: bool,
    pub expression: Expression
}

#[derive(Clone,Eq,PartialEq,Debug)]
pub enum ReductionMethod {
    Sum
}

#[derive(Clone,Copy,Eq,PartialEq,Debug)]
pub enum InteractionSymmetry {
    Symmetric,
    Antisymmetric,
    Asymmetric
}

mod tests {
    #[allow(unused_imports)] // Rust analyzer bug
    use super::super::*;

    #[test]
    fn coulomb_potential() {
        let string = r#"interaction myinteraction (p1: PointLike, p2: PointLike) for r < CUTOFF {
            quantity myforce -[sum]-> (F, F) {
                4.0*EPSILON*((SIGMA/r))
            }
        }"#;
        fips_parser::interaction(string).expect("Cannot parse string");
    }
}