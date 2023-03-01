use std::ptr::null_mut;
use std::ffi::CString;

use llvm_sys::prelude::*;
use llvm_sys::core::*;

use super::{Context, Value};

pub struct BasicBlock {
    pub bb: LLVMBasicBlockRef,
    name: CString
}

impl BasicBlock {
    pub fn append_function(ctx: &Context, fun: &Value, name: &str) -> BasicBlock {
        let mut bb = BasicBlock {
            bb: null_mut(),
            name: CString::new(name).unwrap()
        };
        bb.bb = unsafe { LLVMAppendBasicBlockInContext(ctx.ctx, fun.val, bb.name.as_ptr()) };
        bb
    }
}