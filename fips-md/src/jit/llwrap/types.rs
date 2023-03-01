use std::ptr::null_mut;

pub use llvm_sys::LLVMTypeKind as TypeKind;
use llvm_sys::prelude::*;
use llvm_sys::core::*;

use super::{Context, LLFALSE};

#[derive(Clone)]
pub struct Type {
    pub ty: LLVMTypeRef
}

impl Type {
    pub fn new(ty: LLVMTypeRef) -> Type {
        Type { ty: ty }
    }

    pub fn get_kind(&self) -> TypeKind {
        unsafe { LLVMGetTypeKind(self.ty) }
    }

    #[allow(dead_code)]
    pub fn get_void(ctx: &Context) -> Type {
        Type {
            ty: unsafe { LLVMVoidTypeInContext(ctx.ctx) }
        }
    }

    pub fn get_double(ctx: &Context) -> Type {
        Type {
            ty: unsafe { LLVMDoubleTypeInContext(ctx.ctx) }
        }
    }

    #[allow(dead_code)]
    pub fn get_int32(ctx: &Context) -> Type {
        Type {
            ty: unsafe { LLVMInt32TypeInContext(ctx.ctx) }
        }
    }

    pub fn get_int64(ctx: &Context) -> Type {
        Type {
            ty: unsafe { LLVMInt64TypeInContext(ctx.ctx) }
        }
    }
}

pub struct FunctionType {
    pub ty: LLVMTypeRef,
    arg_types: Vec<LLVMTypeRef>
}

impl FunctionType {
    pub fn new(return_type: &Type, arg_types: &mut [Type]) -> FunctionType {
        let mut ft = FunctionType {
            arg_types: arg_types.to_vec().iter().map(|x| x.ty).collect(),
            ty: null_mut()
        };
        unsafe { ft.ty = LLVMFunctionType(return_type.ty, ft.arg_types.as_mut_ptr(), ft.arg_types.len() as u32, LLFALSE) };
        ft
    }
}