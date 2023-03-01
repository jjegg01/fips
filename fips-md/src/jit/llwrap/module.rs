use std::ffi::CString;
use std::ptr::null_mut;
use std::collections::HashMap;

use llvm_sys::prelude::*;
use llvm_sys::core::*;

use super::{Context, FunctionType, Value};

pub struct Module {
    pub module: LLVMModuleRef,
    name: CString,
    _ctx: Context,
    func_names: HashMap<String, CString>
}

impl Module {
    /// Create new module in context with given name
    pub fn with_context(ctx: &Context, name: &str) -> Module { 
        let mut m = Module {
            name: CString::new(name).unwrap(), // Tie string lifetime to struct lifetime
            module: null_mut(),
            _ctx: ctx.clone(),
            func_names: HashMap::new()
        };
        unsafe { m.module = LLVMModuleCreateWithNameInContext(m.name.as_ptr(), ctx.ctx); }
        m
    }

    pub fn add_function(&mut self, name : &str, ftype: &FunctionType) -> Value {
        self.store_func_name(name);
        Value::new(unsafe { LLVMAddFunction(self.module, self.func_names.get(name).unwrap().as_ptr(), ftype.ty) } )
    }

    pub fn dump(&mut self) {
        unsafe { LLVMDumpModule(self.module); }
    }

    fn store_func_name(&mut self, name: &str) {
        self.func_names.insert(name.to_string(), CString::new(name).unwrap());
    }
}

impl Drop for Module {
    fn drop(&mut self) { 
        unsafe { LLVMDisposeModule(self.module); }
    }
}