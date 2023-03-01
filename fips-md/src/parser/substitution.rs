//! Compile time constant substitution

use std::fmt::Display;

use anyhow::{Result, anyhow};

use crate::utils::FipsValue;

use super::*;

/// Helper to contain possible values for substitution 
/// (kind of a reverse of FipsValue: this is used to represent *Rust* types given
/// for substitution without assuming that they are also valid types in FIPS)
#[derive(Debug, Clone)]
pub enum SubstitutionValue {
    /// i64 analogous to Int64 FIPS type
    I64(i64),
    /// f64 analogous to Double FIPS type
    F64(f64),
    /// usize for arrays
    Usize(usize)
}

// Try to convert the sub
impl Into<FipsValue> for SubstitutionValue {
    fn into(self) -> FipsValue {
        match self {
            Self::I64(value) => FipsValue::Int64(value),
            Self::F64(value) => FipsValue::Double(value),
            Self::Usize(value) => FipsValue::Int64(value as i64),
        }
    }
}

impl Display for SubstitutionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstitutionValue::I64(v) => write!(f, "i64 ({})", v),
            SubstitutionValue::F64(v) => write!(f, "f64 ({})", v),
            SubstitutionValue::Usize(v) => write!(f, "usize ({})", v),
        }
    }
}

pub trait ConstantSubstitution {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()>;
}

/* -- Base cases -- */

impl ConstantSubstitution for CompileTimeConstant<i64> {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            CompileTimeConstant::Identifier(ident_name) if ident_name == name => {
                if let SubstitutionValue::I64(value) = value {
                    self.substitute(*value);
                    Ok(())
                }
                else {
                    Err(anyhow!("Invalid type for substitution: expected i64, but got {}", value))
                }
            }
            // TODO: might be sensible to replace existing substitutions as well
            _ => Ok(())
        }
    }
}

impl ConstantSubstitution for CompileTimeConstant<f64> {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            CompileTimeConstant::Identifier(ident_name) if ident_name == name => {
                if let SubstitutionValue::F64(value) = value {
                    self.substitute(*value);
                    Ok(())
                }
                else {
                    Err(anyhow!("Invalid type for substitution: expected f64, but got {}", value))
                }
            } 
            _ => Ok(())
        }
    }
}

impl ConstantSubstitution for CompileTimeConstant<usize> {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            CompileTimeConstant::Identifier(ident_name) if ident_name == name => {
                if let SubstitutionValue::Usize(value) = value {
                    self.substitute(*value);
                    Ok(())
                }
                else {
                    Err(anyhow!("Invalid type for substitution: expected usize, but got {}", value))
                }
            } 
            _ => Ok(())
        }
    }
}

// impl ConstantSubstitution for Atom {
//     fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
//         match self {
//             Atom::Literal(_) => Ok(()),
//             Atom::Variable(variable_name) => {
//                 if name == variable_name {
//                     // Create new literal
//                     let literal = match value {
//                         SubstitutionValue::I64(value) => Literal::Int64(*value),
//                         SubstitutionValue::Double(value) => Literal::Double(*value),
//                         // TODO: There is no usize type in FIPS yet, so we just
//                         // silently cast to i64. Maybe there is a better solution
//                         SubstitutionValue::Usize(value) => Literal::Int64(*value as i64),
//                     };
//                     let tmp = Atom::Literal(literal);
//                     std::mem::swap(self, &mut tmp);
//                     todo!()
//                 }
//                 else {
//                     Ok(())
//                 }
//             }
//         }
//     }
// }

/* -- General -- */

impl ConstantSubstitution for FipsType {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Substitution can only happen in array types
        match self {
            FipsType::Double | FipsType::Int64 => Ok(()),
            FipsType::Array {typ, length} => {
                typ.substitute_constant(name, value)?;
                length.substitute_constant(name, value)
            }
        }
    }
}

impl ConstantSubstitution for Statement {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            Statement::Let(statement) => statement.substitute_constant(name, value),
            Statement::Assign(statement) => statement.substitute_constant(name, value),
            Statement::Update(_) => Ok(()),
            Statement::Call(_) => Ok(())
        }
    }
}

impl ConstantSubstitution for LetStatement {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Error if trying to shadow the constant with a local binding
        if self.name == name {
            return Err(anyhow!("Cannot shadow compile-time constant {} with local binding", name));
        }
        self.initial.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for AssignStatement {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Error if trying to assign to the constant
        if self.assignee == name {
            return Err(anyhow!("Cannot assign to compile-time constant {}", name));
        }
        // TODO: Assignee substitutions once place expressions exist
        if let Some(index) = self.index.as_mut() {
            index.substitute_constant(name, value)?;
        }
        self.value.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for Expression {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            // Somewhat surprisingly, atoms are not substituted here
            // They are resolved with the other variables at a later stage
            Expression::Atom(_) => Ok(()),
            Expression::BinaryOperation(binop) => binop.substitute_constant(name, value),
            Expression::FunctionCall(call) => call.substitute_constant(name, value),
            Expression::Block(block) => block.substitute_constant(name, value),
            Expression::Indexing(indexing) => indexing.substitute_constant(name, value),
            Expression::AdHocArray(adhocarray) => adhocarray.substitute_constant(name, value),
        }
    }
}

impl ConstantSubstitution for BinaryOperation {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        self.lhs.substitute_constant(name, value)?;
        self.rhs.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for FunctionCall {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        for parameter in &mut self.parameters {
            parameter.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

impl ConstantSubstitution for BlockExpression {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        for statement in &mut self.statements {
            statement.substitute_constant(name, value)?;
        }
        self.expression.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for AtIndex {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        self.index.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for AdHocArrayExpression {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        for element in &mut self.elements {
            element.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

/* -- Particles -- */

impl ConstantSubstitution for Particle {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Substitution might happen in members
        for member in &mut self.members {
            member.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

impl ConstantSubstitution for ParticleMember {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Substitution might happen in type
        self.typ.substitute_constant(name, value)
    }
}

/* -- Simulation -- */

impl ConstantSubstitution for Simulation {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        // Substitution might happen in blocks
        for block in &mut self.blocks {
            block.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

impl ConstantSubstitution for SimulationBlock {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        match self {
            SimulationBlock::Once(block) => block.substitute_constant(name, value),
            SimulationBlock::Step(block) => block.substitute_constant(name, value)
        }
    }
}

impl ConstantSubstitution for OnceBlock {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        self.step.substitute_constant(name, value)?;
        for subblock in &mut self.subblocks {
            subblock.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

impl ConstantSubstitution for StepBlock {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        self.step_range.substitute_constant(name, value)?;
        for subblock in &mut self.subblocks {
            subblock.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

impl ConstantSubstitution for StepRange {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        self.start.substitute_constant(name, value)?;
        self.end.substitute_constant(name, value)?;
        self.step.substitute_constant(name, value)
    }
}

impl ConstantSubstitution for SimulationSubBlock {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        for statement in &mut self.statements {
            statement.substitute_constant(name, value)?;
        }
        Ok(())
    }
}

/* -- Externs -- */

impl ConstantSubstitution for ExternFunctionDecl {
    fn substitute_constant(&mut self, name: &str, value: &SubstitutionValue) -> Result<()> {
        for parameter_type in &mut self.parameter_types {
            parameter_type.substitute_constant(name, value)?;
        }
        self.return_type.substitute_constant(name, value)
    }
}