use super::super::llwrap as ll;
use crate::parsing::{Identifier, IdentType};
use super::{Codegen, CodegenContext};

impl Codegen for Identifier {
    fn codegen(&self, cctx: &mut CodegenContext) -> ll::Value {
        match self.typ {
            IdentType::Inferred => {
                // Check if variable exists in current context
                let var = cctx.pop_var(&self.name);
                match var {
                    Some(varaddr) => varaddr,
                    None => {
                        panic!("Undefined variable {}", self.name)
                    }
                }
            }
            _ => panic!("Unimplemented variable type")
        }
    }
}