use llvm_sys::prelude::*;
use llvm_sys::core::*;
use super::{Type, LLTRUE};

pub struct Value {
    pub val: LLVMValueRef
}

impl Value {
    pub fn new(val: LLVMValueRef) -> Value {
        Value { val: val }
    }

    pub fn get_type(&self) -> Type {
        Type::new(unsafe { LLVMTypeOf(self.val) } )
    }

    pub fn with_constant_fp(ty: Type, val: f64) -> Value {
        Value { val: unsafe { LLVMConstReal(ty.ty, val) } }
    }

    pub fn with_constant_int(ty: Type, val: i64) -> Value {
        Value { val: unsafe { LLVMConstInt(ty.ty, val as u64, LLTRUE) } }
    }
}