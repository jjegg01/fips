//! Structs related to symbol table construction and manipulation

use anyhow::{anyhow, Result};

use std::collections::HashMap;

use crate::{parser::{FipsType, SubstitutionValue}, runtime::{FunctionID, MemberID, ParticleIndexEntry}, utils::FipsValue};

/// Symbol table used by the code generator. T is the type of data attached to the symbols.
#[derive(Clone)]
pub struct SymbolTable<T> {
    /// The actual symbol table
    symbols: HashMap<String, FipsSymbol<T>>,
    /// Additional symbol tables used for resolution (after this table go
    /// _backwards_ through the vec)
    subtables: Vec<SymbolTable<T>>
}

// Quick macro for to print error messages that can still refer to the key used
macro_rules! insert_or_error {
    ($map:expr, $key:expr, $val:expr, $err:expr) => {
        if $map.contains_key(&$key) { // Key is only borrowed here
            Err($err)                 // <- Usage here is fine
        }
        else {
            $map.insert($key, $val);  // <- only moved if successful
            Ok(())
        }
    }
}

impl<T> SymbolTable<T> {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            subtables: vec![]
        }
    }

    /// Create symbol table from particle definition
    pub fn from_particle(particle_def: &ParticleIndexEntry) -> Self {
        let mut symbol_table = Self::new();
        for (member_id, member_def) in particle_def.get_members() {
            symbol_table.add_particle_member(member_def.get_name().into(), member_id)
                .expect("Inconsistent member names!");
        }
        symbol_table
    }

    /// Add a new constant entry to the symbol table from a substitution value
    pub fn add_constant_from_substitution(&mut self, name: String, value: &SubstitutionValue) -> Result<()> {
        insert_or_error!(self.symbols, name, 
            FipsSymbol::from_substitution(&value),
            anyhow!("Cannot redefine constant with name {}", name))
    }

    pub fn add_constant_f64(&mut self, name: String, value: f64) -> Result<()> {
        insert_or_error!(self.symbols, name, 
            FipsSymbol::from_f64(value),
            anyhow!("Cannot redefine constant with name {}", name))
    }

    pub fn add_local_symbol(&mut self, name: String, typ: FipsType) -> Result<()> {
        insert_or_error!(self.symbols, name,
            FipsSymbol::new_local(typ),
            anyhow!("Cannot redefine local variable with name {}", name)
        )
    }

    pub fn add_local_symbol_with_value(&mut self, name: String, typ: FipsType, value: T) -> Result<()> {
        insert_or_error!(self.symbols, name,
            FipsSymbol::new_local_with_value(typ, value),
            anyhow!("Cannot redefine local variable with name {}", name)
        )
    }

    pub fn add_particle_member(&mut self, name: String, member_id: MemberID) -> Result<()> {
        insert_or_error!(self.symbols, name,
            FipsSymbol::new_particle_member(member_id),
            anyhow!("Cannot redefine particle member variable with name {}", name)
        )
    }

    pub fn add_function(&mut self, name: String, function_id: FunctionID) -> Result<()> {
        insert_or_error!(self.symbols, name,
            FipsSymbol::new_function(function_id),
            anyhow!("Cannot redefine function with name {}", name)
        )
    }

    /// Add another symbol table to this symbol table
    pub fn push_table(&mut self, symbols: SymbolTable<T>) {
        self.subtables.push(symbols)
    }

    /// Pop the last symbol table pushed
    pub fn pop_table(&mut self) -> Option<SymbolTable<T>> {
        self.subtables.pop()
    }

    /// Convert the symbol table to a different type parameter (resets all symbol values)
    pub fn convert<U>(self) -> SymbolTable<U> {
        let symbols = self.symbols.into_iter()
            .map(|(name, symbol)| (name, symbol.convert()))
            .collect::<HashMap<_,_>>();
        let subtables = self.subtables.into_iter()
            .map(|table| table.convert())
            .collect::<Vec<_>>();
        SymbolTable {
            symbols,
            subtables
        }
    }

    /// Create an iterator over all symbols
    pub fn iter_mut(&mut self) -> impl Iterator<Item=(&String, &mut FipsSymbol<T>)> {
        // I cannot get this to work with just chaining IterMut due to recursive
        // return type (i.e. concrete return type cannot be resolved)
        // There is probably a nicer, more performant solution...
        let mut total_symbols = vec![];
        for symbol_pair in &mut self.symbols {
            total_symbols.push(symbol_pair)
        }
        for subtable in &mut self.subtables {
            for symbol_pair in subtable.iter_mut() {
                total_symbols.push(symbol_pair);
            }
        }
        total_symbols.into_iter()
    }

    /// Create an iterator over all symbols
    pub fn iter(&self) -> impl Iterator<Item=(&String, &FipsSymbol<T>)> {
        // I cannot get this to work with just chaining IterMut due to recursive
        // return type (i.e. concrete return type cannot be resolved)
        // There is probably a nicer, more performant solution...
        let mut total_symbols = vec![];
        for symbol_pair in &self.symbols {
            total_symbols.push(symbol_pair)
        }
        for subtable in &self.subtables {
            for symbol_pair in subtable.iter() {
                total_symbols.push(symbol_pair);
            }
        }
        total_symbols.into_iter()
    }

    pub fn resolve_symbol(&self, name: &str) -> Option<&FipsSymbol<T>> {
        // Check self
        if let Some(symbol) = self.symbols.get(name) {
            return Some(symbol)
        };
        for subtable in self.subtables.iter().rev() {
            if let Some(symbol) = subtable.resolve_symbol(name) {
                return Some(symbol)
            }
        };
        None
    }
}



/// A symbol, i.e. an LLVM Value attached to a name like a variable, function, constant, etc.
#[derive(Clone)]
pub struct FipsSymbol<T> {
    pub(crate) kind: FipsSymbolKind,
    pub(crate) value: Option<T>
}

impl<T> FipsSymbol<T> {
    pub fn from_substitution(value: &SubstitutionValue) -> Self {
        Self {
            kind: FipsSymbolKind::from_subsitution(value),
            value: None
        }
    }

    pub fn from_f64(value: f64) -> Self {
        Self {
            kind: FipsSymbolKind::from_f64(value),
            value: None
        }
    }

    pub fn new_local(typ: FipsType) -> Self {
        Self {
            kind: FipsSymbolKind::LocalVariable(typ),
            value: None
        }
    }

    pub fn new_local_with_value(typ: FipsType, value: T) -> Self {
        Self {
            kind: FipsSymbolKind::LocalVariable(typ),
            value: Some(value)
        }
    }

    pub fn new_particle_member(member_id: MemberID) -> Self {
        Self {
            kind: FipsSymbolKind::ParticleMember(member_id),
            value: None
        }
    }

    pub fn new_function(function_id: FunctionID) -> Self {
        Self {
            kind: FipsSymbolKind::Function(function_id),
            value: None
        }
    }

    pub fn convert<U>(self) -> FipsSymbol<U> {
        FipsSymbol {
            kind: self.kind,
            value: None
        }
    }

    pub fn set_value(&mut self, value: T) {
        self.value = Some(value)
    }
}

/// Different kinds of symbols
#[derive(Clone)]
pub enum FipsSymbolKind {
    /// A symbolic constant
    Constant(FipsValue),
    /// A local variable
    LocalVariable(FipsType),
    /// A particle member (we don't need the particle ID since a single thread only works
    /// on one type of particle)
    ParticleMember(MemberID),
    /// A global function
    Function(FunctionID)
}

impl FipsSymbolKind {
    /// Crate new symbol from substitution value
    pub fn from_subsitution(val: &SubstitutionValue) -> Self {
        Self::Constant(val.clone().into())
    }

    pub fn from_f64(val: f64) -> Self {
        Self::Constant(val.into())
    }
}