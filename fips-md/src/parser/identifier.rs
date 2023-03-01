//! Enumeration representing an identifier

#[derive(Clone,Eq,PartialEq,Debug)]
pub enum Identifier {
    Named(String),
    Ignored // "_" identifier
}