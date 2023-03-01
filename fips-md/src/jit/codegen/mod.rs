mod const_value;
mod binop;
mod identifier;

use std::collections::HashMap;

use super::llwrap as ll;

use crate::parsing::{ASTNode};

pub struct CodegenContext {
    pub ir_builder: ll::IRBuilder,
    variables: HashMap<String, Vec<ll::Value>>
}

impl CodegenContext {
    pub fn new(builder: ll::IRBuilder) -> CodegenContext {
        CodegenContext {
            ir_builder: builder,
            variables: HashMap::new()
        }
    }

    pub fn push_var(&mut self, name: &str, val: ll::Value) {
        match self.variables.get_mut(name) {
            Some(vstack) => vstack.push(val),
            None => { 
                let v = vec!(val);
                self.variables.insert(String::from(name), v); 
            }
        }
    }

    // None option means that variable has never existed or is no longer valid
    // TODO: Differentiate these cases for better debug messages?
    pub fn pop_var(&mut self, name: &str) -> Option<ll::Value> {
        match self.variables.get_mut(name) {
            Some(v) => v.pop(),
            None => None
        }
    }
}

pub trait Codegen {
    fn codegen(&self, cctx: &mut CodegenContext) -> ll::Value;
}

impl Codegen for ASTNode {
    fn codegen(&self, cctx: &mut CodegenContext) -> ll::Value {
        match self {
            ASTNode::BinOp(binop) => binop.codegen(cctx),
            ASTNode::Literal(cv) => cv.codegen(cctx),
            _ => panic!("JITing not implemented for this ASTNode")
        }
    }
}