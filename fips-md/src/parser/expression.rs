//! Structures relating to expressions

use super::{CompileTimeConstant, Statement};

/// Types of Expressions
#[derive(Clone,PartialEq,Debug)]
pub enum Expression {
    /// Atomic Expression
    Atom(Atom),
    /// Binary Operation
    BinaryOperation(BinaryOperation),
    /// Function call
    FunctionCall(FunctionCall),
    /// Block expression
    Block(BlockExpression),
    /// Array indexing
    Indexing(AtIndex),
    /// Ad-hoc array creation (i.e. "[1,2,a,b]")
    AdHocArray(AdHocArrayExpression)
}

/// An atomic expression
#[derive(Clone,PartialEq,Debug)]
pub enum Atom {
    /// Some (yet unresolved) variable name
    Variable(String),
    /// Some variable with a 'namespace' (used in interaction expressions)
    NamespaceVariable { namespace: String, name: String },
    /// Some literal
    Literal(Literal)
}

/// A literal
#[derive(Clone,PartialEq,Debug)]
pub enum Literal {
    Double(f64),
    Int64(i64)
}

/// An operation with two operands
#[derive(Clone,PartialEq,Debug)]
pub struct BinaryOperation {
    /// Left side of operation
    pub lhs: Box<Expression>,
    /// Right side of operation
    pub rhs: Box<Expression>,
    /// Operator of operation
    pub op: BinaryOperator
}

/// A binary operand
#[derive(Copy,Clone,Eq,PartialEq,Debug)]
pub enum BinaryOperator {
    Add, Sub,
    Mul, Div
}

/// Call of a named function
#[derive(Clone,PartialEq,Debug)]
pub struct FunctionCall {
    /// Name of function
    pub fn_name: String,
    /// Parameter list
    pub parameters: Vec<Expression>
}

/// Block expressions (i.e. a block of statements terminated with an expression)
#[derive(Clone,PartialEq,Debug)]
pub struct BlockExpression {
    pub statements: Vec<Statement>,
    pub expression: Box<Expression>
}


/// Array indexing expressions
#[derive(Clone,PartialEq,Debug)]
pub struct AtIndex {
    pub array: String,
    pub index: CompileTimeConstant<usize>
}

#[derive(Clone,PartialEq,Debug)]
pub struct AdHocArrayExpression {
    pub elements: Vec<Expression>
}