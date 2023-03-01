use std::ffi::CString;
use std::rc::Rc;
use std::cell::RefCell;
use std::ptr::null_mut;

use llvm_sys::prelude::*;
use llvm_sys::core::*;

#[derive(Clone)]
pub struct Context {
    pub ctx: LLVMContextRef, // Convenience
    ctx_container: Rc<RefCell<ContextContainer>>
}

/// Helper struct for keeping strings allocated while they are referenced by LLVM
/// (Necessary because LLVM does not appear (?) to copy strings passed as names etc.)
struct ContextContainer {
    ctx: LLVMContextRef,
    strings: Vec<CString>
}

impl ContextContainer {
    fn new() -> ContextContainer {
        ContextContainer {
            ctx: unsafe { LLVMContextCreate() },
            strings: Vec::new()
        }
    }

    fn store_string(&mut self, string: &str) -> *const libc::c_char {
        self.strings.push(CString::new(string).unwrap());
        self.strings[self.strings.len()-1].as_ptr()
    }
}

impl Context {
    /// Create new context (this cannot not fail?)
    pub fn new() -> Context {
        let mut ctx = Context {
            ctx: null_mut(),
            ctx_container: Rc::new(RefCell::new(ContextContainer::new()))
        };
        ctx.ctx = ctx.ctx_container.borrow_mut().ctx; // ctx
        ctx
    }

    pub fn store_string(&self, string: &str) -> *const libc::c_char {
        self.ctx_container.borrow_mut().store_string(string)
    }
}

impl Drop for ContextContainer {
    fn drop(&mut self) {
        unsafe {
            LLVMContextDispose(self.ctx);
        }
    }
}