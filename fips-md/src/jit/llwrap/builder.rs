use llvm_sys::prelude::*;
use llvm_sys::core::*;

use super::{Context, BasicBlock, Value, Type};

pub struct IRBuilder {
    pub builder: LLVMBuilderRef,
    ctx: Context
}

impl IRBuilder {
    pub fn with_context(ctx: &Context) -> IRBuilder { 
        IRBuilder {
            builder: unsafe { LLVMCreateBuilderInContext(ctx.ctx)},
            ctx : ctx.clone()
        }
    }

    pub fn get_context(&self) -> &Context {
        &self.ctx
    }

    pub fn position_at_end(&self, bb: &BasicBlock) {
        unsafe { LLVMPositionBuilderAtEnd(self.builder, bb.bb) }
    }

    pub fn build_ret(&self, val: &Value) {
        unsafe { LLVMBuildRet(self.builder, val.val); }
    }

    // Conversions

    pub fn build_si_to_fp(&self, val: &Value, ty: &Type, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildSIToFP(self.builder, val.val, ty.ty, name) })
    }

    // -- Arithmetic operations --

    // Addition

    pub fn build_int_add(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildAdd(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_fp_add(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildFAdd(self.builder, lhs.val, rhs.val, name) })
    }

    // Subtraction

    pub fn build_int_sub(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildSub(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_fp_sub(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildFSub(self.builder, lhs.val, rhs.val, name) })
    }

    // Multiplication
    
    pub fn build_int_mul(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildMul(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_fp_mul(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildFMul(self.builder, lhs.val, rhs.val, name) })
    }

    // Division

    pub fn build_int_sdiv(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildSDiv(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_int_udiv(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildUDiv(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_fp_div(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildFDiv(self.builder, lhs.val, rhs.val, name) })
    }

    // Division remainder (modulo)

    pub fn build_int_srem(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildSRem(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_int_urem(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildURem(self.builder, lhs.val, rhs.val, name) })
    }    

    pub fn build_fp_rem(&self, lhs: &Value, rhs: &Value, name: &str) -> Value {
        let name = self.ctx.store_string(name);
        Value::new( unsafe { LLVMBuildFRem(self.builder, lhs.val, rhs.val, name) })
    }
}

impl Drop for IRBuilder {
    fn drop(&mut self) { unsafe {
        LLVMDisposeBuilder(self.builder);
    }}
}