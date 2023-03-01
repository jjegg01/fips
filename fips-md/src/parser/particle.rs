//! Structure for the `particle` primary block

use super::FipsType;

#[derive(Clone,Eq,PartialEq,Debug)]
/// Node for the information of a `particle` block
pub struct Particle {
    /// Name of this particle type
    pub name: String,
    //pub superparticle: Option<String>,
    pub members: Vec<ParticleMember>
}

#[derive(Clone,Eq,PartialEq,Debug)]
pub struct ParticleMember {
    pub name: String,
    pub mutable: bool,
    pub typ: FipsType,
    pub is_position: bool
}

mod tests {
    #[allow(unused_imports)] // Rust analyzer bug
    use super::super::*;

    #[test]
    fn pointlike_commented() {
        let string = r#"particle PointLike {
            // Position is special and cannot be redefined, but aliased (array specifier is not allowed,
            // as position is always NDIM dimensional)
            x : mut position,
            v : mut [f64; NDIM], // NDIM is dimensionality of problem
            F : mut [f64; NDIM], // Quantities bound to interaction calculations later also need to be declared
            mass: f64            // Constants have to be defined in Rust code (can be either per-particle or
                                    // per-type)
        }"#;
        fips_parser::particle(string).expect("Cannot parse string");
    }

    #[test]
    fn pointlike() {
        let string = r#"particle PointLike {
            x : mut position,
            v : mut [f64; NDIM],
            F : mut [f64; NDIM],
            mass: f64
        }"#;
        fips_parser::particle(string).expect("Cannot parse string");
    }

    #[ignore]
    #[test]
    fn extends() {
        let string = "particle Orientable extends PointLike {}";
        fips_parser::particle(string).expect("Cannot parse string");
    }
}