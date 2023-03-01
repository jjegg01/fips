//! Builtin and external global functions

use std::ffi::CString;

use llvm_sys::{core::{LLVMAddFunction, LLVMAddGlobal, LLVMArrayType, LLVMBuildCall, LLVMBuildLoad, LLVMBuildStore, LLVMConstNull, LLVMDoubleTypeInContext, LLVMFunctionType, LLVMInt64TypeInContext, LLVMPointerType, LLVMSetInitializer, LLVMSetLinkage, LLVMVoidTypeInContext}, prelude::*};
use rand::distributions::Distribution;
use anyhow::Result;

use crate::{codegen::LLFunctionSymbolValue, cstr, cstring, parser::ExternFunctionDecl, runtime::{BuiltinFunction, FunctionID, FunctionIndexEntry}};

use super::{CallbackTarget, LLSymbolValue, llhelpers::get_llvm_type_dims};

// TODO: Consider this design for the rest of the LLVM helpers to clean up that mess

impl FunctionIndexEntry {
    /// Create the symbol value for this function (i.e. the function declaration and any necessary global pointers)
    pub(crate) unsafe fn create_symbol_value(&self, function_id: FunctionID, context: LLVMContextRef, module: LLVMModuleRef) -> Result<LLSymbolValue> {
        match self {
            FunctionIndexEntry::Extern(externfunc) => externfunc.create_symbol_value(function_id, context, module),
            FunctionIndexEntry::Builtin(builtin) => builtin.create_symbol_value(function_id, context, module)
        }
    }

    /// Build a call to this function and return the return ValueRef
    /// TODO: Do type checking on parameters
    pub(crate) unsafe fn build_call(&self, context: LLVMContextRef, builder: LLVMBuilderRef,
        symbol_value: &LLFunctionSymbolValue, parameter_vals: Vec<LLVMValueRef>) -> Result<LLVMValueRef>
    {
        match self {
            FunctionIndexEntry::Extern(externfunc) => externfunc.build_call(
                context, builder, symbol_value, parameter_vals, self.returns_array()),
            FunctionIndexEntry::Builtin(builtin) => builtin.build_call(
                context, builder, symbol_value, parameter_vals, self.returns_array())
        }
    }    
}

impl BuiltinFunction {
    pub(crate) unsafe fn create_symbol_value(&self, function_id: FunctionID, context: LLVMContextRef, module: LLVMModuleRef) -> Result<LLSymbolValue> {
        let double_type = LLVMDoubleTypeInContext(context);
        let int64_type = LLVMInt64TypeInContext(context);
        let math_double_func_type = LLVMFunctionType(double_type, [double_type].as_mut_ptr(), 1, 0);
        Ok(match self {
            BuiltinFunction::Sqrt => {
                LLSymbolValue::Function(LLFunctionSymbolValue {
                    function_id,
                    function: LLVMAddFunction(module, cstr!("llvm.sqrt.f64"), math_double_func_type),
                    global_parameter_ptrs: vec![None]
                })
                
            },
            BuiltinFunction::Sin => {
                LLSymbolValue::Function(LLFunctionSymbolValue {
                    function_id,
                    function: LLVMAddFunction(module, cstr!("llvm.sin.f64"), math_double_func_type),
                    global_parameter_ptrs: vec![None]
                })
                
            },
            BuiltinFunction::Cos => {
                LLSymbolValue::Function(LLFunctionSymbolValue {
                    function_id,
                    function: LLVMAddFunction(module, cstr!("llvm.cos.f64"), math_double_func_type),
                    global_parameter_ptrs: vec![None]
                })
                
            },
            BuiltinFunction::RandomNormal => {
                let random_func_type = LLVMFunctionType(double_type, [int64_type].as_mut_ptr(), 1, 0);
                LLSymbolValue::Function(LLFunctionSymbolValue {
                    function_id,
                    function: LLVMAddFunction(module, cstr!("_random_normal"), random_func_type),
                    global_parameter_ptrs: vec![None]
                })
            },
        })
    }

    pub(crate) unsafe fn build_call(&self, _: LLVMContextRef, builder: LLVMBuilderRef,
        symbol_value: &LLFunctionSymbolValue, mut parameter_vals: Vec<LLVMValueRef>,
        _: bool) 
    -> Result<LLVMValueRef> {
        Ok(match self {
            BuiltinFunction::Sin | BuiltinFunction::Cos | BuiltinFunction::Sqrt | BuiltinFunction::RandomNormal => {
                let name = format!("{}_ret", self.get_name());
                LLVMBuildCall(builder, symbol_value.function, parameter_vals.as_mut_ptr(),
                    parameter_vals.len() as u32, cstring!(name))
            }
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn _random_normal(callback_target: u64) -> f64 {
    let callback_target = &mut *(callback_target as usize as *mut CallbackTarget);
    let rng = &mut (callback_target.thread_context.rng);
    callback_target.thread_context.normal_dist.sample(rng)
}

impl ExternFunctionDecl {
    pub(crate) unsafe fn create_symbol_value(&self, function_id: FunctionID, context: LLVMContextRef, module: LLVMModuleRef) -> Result<LLSymbolValue> {
        // Generate parameter types and globals for passing array parameters
        let mut parameter_types = vec![];
        let mut global_parameter_ptrs = vec![];
        for (i,typ) in self.parameter_types.iter().enumerate() {
            let (base_type, dims) = get_llvm_type_dims(context, typ)?;
            if dims.is_empty() { 
                parameter_types.push(base_type);
                global_parameter_ptrs.push(None);
            }            
            else {
                match dims.len() {
                    0 => unreachable!(),
                    1 => {
                        let name = format!("{}_param_{}", self.name, i);
                        let parameter_type = LLVMArrayType(base_type, dims[0] as u32);
                        let global_parameter_ptr = LLVMAddGlobal(module, 
                            parameter_type, 
                            cstring!(name)
                        );
                        LLVMSetLinkage(global_parameter_ptr, llvm_sys::LLVMLinkage::LLVMCommonLinkage);
                        LLVMSetInitializer(global_parameter_ptr, LLVMConstNull(parameter_type));
                        global_parameter_ptrs.push(Some(global_parameter_ptr));
                        parameter_types.push(LLVMPointerType(parameter_type, 0));
                    }
                    _ => unimplemented!("Multidimensional parameters in extern functions not yet supported")
                } 
            }
        }
        // Check return type
        let (base_type, dims) = get_llvm_type_dims(context, &self.return_type)?;
        let return_type;
        match dims.len() {
            0 => {
                return_type = base_type;
            }
            1 => {
                return_type = LLVMVoidTypeInContext(context);
                let parameter_type = LLVMArrayType(base_type, dims[0] as u32);
                let name = format!("{}_return", self.name);
                let global_parameter_ptr = LLVMAddGlobal(module, 
                    parameter_type, 
                    cstring!(name)
                );
                LLVMSetLinkage(global_parameter_ptr, llvm_sys::LLVMLinkage::LLVMCommonLinkage);
                LLVMSetInitializer(global_parameter_ptr, LLVMConstNull(parameter_type));
                global_parameter_ptrs.push(Some(global_parameter_ptr));
                parameter_types.push(LLVMPointerType(parameter_type, 0));
            }
            _ => unimplemented!("Multidimensional return type in extern functions not yet supported")
        }

        let func_type = LLVMFunctionType(return_type, 
            parameter_types.as_mut_ptr(), parameter_types.len() as u32, 0);
        let function = LLVMAddFunction(module, cstring!(self.name.clone()), func_type);
        Ok(LLSymbolValue::Function(LLFunctionSymbolValue {
            function_id,
            function,
            global_parameter_ptrs,
        }))
    }

    pub(crate) unsafe fn build_call(&self, _: LLVMContextRef, builder: LLVMBuilderRef,
        symbol_value: &LLFunctionSymbolValue, mut parameter_vals: Vec<LLVMValueRef>,
        returns_array: bool) 
    -> Result<LLVMValueRef> {
        // Replace array parameters by pointers by storing them in the global helpers
        let global_parameter_ptrs = &symbol_value.global_parameter_ptrs;
        for (parameter_val, global_parameter_ptr) in 
            parameter_vals.iter_mut().zip(global_parameter_ptrs.iter()) 
        {
            if let Some(global_parameter_ptr) = global_parameter_ptr {
                LLVMBuildStore(builder, *parameter_val, *global_parameter_ptr);
                *parameter_val = *global_parameter_ptr;
            }
        }
        // If the functions returns an array, add an additional parameter
        if returns_array {
            parameter_vals.push(global_parameter_ptrs.last()
                .unwrap() // If the function returns an array, there must be at least 1 parameter
                .unwrap() // The last parameter is pass-via-pointer
            );
        }
        // Call the function
        let name = format!("{}_ret", self.name);
        let call_ret = LLVMBuildCall(builder, symbol_value.function, 
            parameter_vals.as_mut_ptr(), parameter_vals.len() as u32,
            if returns_array {cstr!("")} else { cstring!(name) });
        
        Ok(if returns_array {
            let name = format!("{}_ret", self.name);
            LLVMBuildLoad(builder, global_parameter_ptrs.last().unwrap().unwrap(), cstring!(name))
        }
        else {
            call_ret
        })
    }
}