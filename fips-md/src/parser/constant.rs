//! Structs for compile time constants

use std::fmt::Display;

/// A compile time constant, i.e. a literal, some pre-defined constant (or an expression of such, TODO)
#[derive(Clone,Eq,PartialEq,Debug)]
pub enum CompileTimeConstant<T:PartialEq> {
    /// A literal (i.e. some concrete value)
    Literal(T),
    /// An identifier (i.e. some not yet substituted value)
    Identifier(String),
    /// An identifier after substitution (the old name is saved for debugging)
    Substituted(T,String)
}

impl<T:PartialEq> CompileTimeConstant<T> {
    /// In-place-ish substitution (does nothing, if substitution not possible)
    pub fn substitute(&mut self, value: T) {
        match self {
            CompileTimeConstant::Identifier(old_name) => {
                // Switch names around
                let mut tmp = String::new();
                std::mem::swap(&mut tmp, old_name);
                // Create new variant
                let new = CompileTimeConstant::Substituted(value, tmp);
                // Switch around round 2
                *self = new;
            }
            _ => {}
        }

    }
}

impl<T:Display+PartialEq> Display for CompileTimeConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileTimeConstant::Literal(x) => x.fmt(f),
            CompileTimeConstant::Identifier(ident) => f.write_str(&ident),
            CompileTimeConstant::Substituted(x, ident) => f.write_fmt(format_args!("{} (substituted from '{}')", x, ident))
        }
    }
}

impl<T:PartialEq> From<T> for CompileTimeConstant<T> {
    fn from(val: T) -> Self {
        CompileTimeConstant::Literal(val)
    }
}