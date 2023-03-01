use super::super::llwrap as ll;
use crate::parsing::ConstValue;
use super::{Codegen, CodegenContext};

impl Codegen for ConstValue {
    fn codegen(&self, cctx: &mut CodegenContext) -> ll::Value {
        match self {
            ConstValue::Float(val) => ll::Value::with_constant_fp(ll::Type::get_double(cctx.ir_builder.get_context()), *val),
            ConstValue::Int(val)   => ll::Value::with_constant_int(ll::Type::get_int64(cctx.ir_builder.get_context()), *val),
            ConstValue::UInt(_)  => panic!("Unsigned integers not yet implemented!")
        }
    }
}