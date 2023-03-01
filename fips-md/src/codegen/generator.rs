// use std::mem::uninitialized;

use std::{collections::{HashMap, HashSet}, ffi::{CString, c_void}, mem::MaybeUninit, ptr};

use anyhow::{anyhow, Result};

use libc::{c_int};
use llvm_sys::{prelude::*, LLVMIntPredicate, LLVMRealPredicate, LLVMLinkage, LLVMTypeKind,
    analysis::{LLVMVerifierFailureAction, LLVMVerifyModule},
    initialization::LLVMInitializeCore,
    orc2::*, orc2::lljit::*,
    transforms::{pass_manager_builder::*}};
use llvm_sys::target::*;
use llvm_sys::core::*;
use slotmap::{Key, KeyData};

use crate::{codegen::{functions::_random_normal, util::unwrap_usize_constant}, cstr, cstring, parser::{self, AssignStatement, FipsType, Statement}, runtime::{BuiltinFunction, Domain, FunctionID, InteractionQuantityID, MemberData, OutOfBoundsBehavior}};

use super::{
    util::unwrap_f64_constant, 
    CallbackTarget, ThreadContext, 
    evaluate_expression, evaluate_binop, convert_to_scalar_or_array,
    analysis::{FipsSymbolKind, SymbolTable, SimulationNode, BarrierKind}, 
    llhelpers::*};

const WORKER_MAIN_NAME: &str = "worker_main";
const WORKER_MODULE_NAME: &str = "worker_module";
type WorkerMainFunc = unsafe extern "C" fn();

#[no_mangle]
pub unsafe extern "C" fn _call2rust_handler(callback_target: u64, barrier_data: u64) {
    let callback_target = callback_target as usize as *const CallbackTarget;
    let barrier = KeyData::from_ffi(barrier_data).into();
    callback_target.as_ref().unwrap().handle_call2rust(barrier);
}

#[no_mangle]
pub unsafe extern "C" fn _interaction_handler(callback_target: u64, barrier_data: u64,
    block_index: usize, neighbor_list_index_ret: *mut *const usize, neighbor_list_ret: *mut *const usize) 
-> () {
    let callback_target = callback_target as usize as *const CallbackTarget;
    let barrier = KeyData::from_ffi(barrier_data).into();
    let (neighbor_list_index, neighbor_list) = callback_target.as_ref().unwrap().handle_interaction(barrier, block_index);
    *neighbor_list_index_ret = neighbor_list_index;
    *neighbor_list_ret = neighbor_list;
}

#[no_mangle]
pub unsafe extern "C" fn _interaction_sync_handler(callback_target: u64, barrier_data: u64) 
-> () {
    let callback_target = callback_target as usize as *const CallbackTarget;
    let barrier = KeyData::from_ffi(barrier_data).into();
    callback_target.as_ref().unwrap().handle_interaction_sync(barrier);
}

#[no_mangle]
pub unsafe extern "C" fn _end_of_step(callback_target: u64) {
    let callback_target = callback_target as usize as *const CallbackTarget;
    callback_target.as_ref().unwrap().end_of_step();
}

#[no_mangle]
pub unsafe extern "C" fn print_u64(x: u64) {
    println!("{}", x);
}

#[no_mangle]
pub unsafe extern "C" fn print_f64(x: f64) {
    println!("{}", x);
}

// DO NOT FORGET TO ADD ALL FUNCTION NAMES FROM ABOVE INTO THIS LIST!
const FIPS_FUNCS: &'static[&str] = &[
    "_call2rust_handler",
    "_interaction_handler",
    "_interaction_sync_handler",
    "_end_of_step",
    "print_u64",
    "print_f64",
    "_random_normal", // Defined in functions.rs
];

const SYSTEM_FUNCS: &'static[&str] = &[
    "memset",
    "fmod",
    "sqrt",
    "sin",
    "cos",
    "sincos",
];

pub extern "C" fn allowed_symbol_filter(ctx: *mut c_void, sym: LLVMOrcSymbolStringPoolEntryRef) -> c_int {
    unsafe {
        if ctx.is_null() {
            panic!("Cannot call allowed_symbol_filter with a null context");
        }
      
        let allow_list: *mut LLVMOrcSymbolStringPoolEntryRef = std::mem::transmute_copy(&ctx);
      
        // If Sym appears in the allowed list then return true.
        let mut allowed_symbol = allow_list;
        while !(*allowed_symbol).is_null() {
            if sym == *allowed_symbol {
                return 1;
            }
            allowed_symbol = allowed_symbol.offset(1);
        }
      
        // otherwise return false.
        return 0;
    }
}

/// Value entries for symbol table
pub(crate) enum LLSymbolValue {
    /// Single value that is a *pointer* (used for global and local variables,
    /// for uniform particle members and for global functions)
    SimplePointer(LLVMValueRef),
    /// Per particle members need two pointers: the global base pointer and a
    /// pointer to the locally loaded value (all particle members are loaded at
    /// the start of each loop)
    ParticleMember {
        base_ptr: LLVMValueRef,
        local_ptr: Option<LLVMValueRef>
    },
    /// Functions may need some reserved global memory to pass array arguments
    /// (stupid C-FFI...)
    Function(LLFunctionSymbolValue)
}

pub(crate) struct LLFunctionSymbolValue {
    // We need the function ID to build calls to this function
    pub(crate) function_id: FunctionID,
    // Function declaration
    pub(crate) function: LLVMValueRef,
    // Associated global for each parameter
    pub(crate) global_parameter_ptrs: Vec<Option<LLVMValueRef>>
}

/// Set of all the codegen stuff we need for evaluate interactions
struct InteractionValues {
    own_pos_block_index: LLVMValueRef,
    interaction_func: LLVMValueRef
}

/// This struct contains the IR Builder
pub struct CodeGenerator {
    /// The LLVM module
    module_ts: LLVMOrcThreadSafeModuleRef,
    /// The callback target
    callback_target: Box<CallbackTarget>,
    /// List of all external symbols for the linker
    external_symbols: Vec<String>
}

impl CodeGenerator {
    pub(crate) fn new(thread_context: ThreadContext) -> Result<Self> {
        // Create callback target (and sneakily switch the thread_context to a reference)
        let callback_target = Box::new(CallbackTarget::new(thread_context));
        let callback_target_ptr = &*callback_target as *const CallbackTarget as u64; // Assume native pointer size to be 64 bit
        let thread_context = &callback_target.thread_context;
        // Extract all necessary information from the thread context
        let particle_id = thread_context.particle_id;
        let domain = &thread_context.executor_context.global_context.runtime.domain;
        let timeline = thread_context.executor_context.global_context.simgraph.timelines.get(&particle_id).unwrap();
        let particle_index = &thread_context.executor_context.global_context.runtime.particle_index;
        let particle_store = &thread_context.executor_context.global_context.runtime.particle_store;
        let particle_data = particle_store.get_particle(particle_id).unwrap();
        let particle_range = thread_context.particle_range.clone();
        let function_index = &thread_context.executor_context.global_context.runtime.function_index;
        // Create the un-namespaced symbol table for compilation
        let global_symbols: SymbolTable<LLSymbolValue> = thread_context
            .executor_context.global_context.global_symbols.clone().convert();
        let particle_symbols: SymbolTable<LLSymbolValue> = timeline.particle_symbols.clone().convert();
        let mut symbol_table = SymbolTable::new();
        symbol_table.push_table(global_symbols);
        symbol_table.push_table(particle_symbols);
        // Gather references to all symbol tables of other particle types
        let neighbor_lists = thread_context.executor_context.neighbor_lists.iter()
            .map(|(interaction_id, neighbor_list)| {
                let neighbor_list = neighbor_list.read().unwrap();
                (*interaction_id, neighbor_list)
            })
            .collect::<HashMap<_,_>>();

        unsafe {
            // Create context, module and builder
            let context_ts = LLVMOrcCreateNewThreadSafeContext();
            let context = LLVMOrcThreadSafeContextGetContext(context_ts);
            let module = LLVMModuleCreateWithNameInContext(cstring!(WORKER_MODULE_NAME), context);
            let builder = LLVMCreateBuilderInContext(context);

            // Type aliases
            let void_type = LLVMVoidTypeInContext(context);
            let int64_type = LLVMInt64TypeInContext(context);
            let int8_type = LLVMInt8TypeInContext(context);
            let double_type = LLVMDoubleTypeInContext(context);

            // Start and end index values
            let start_index = LLVMConstInt(int64_type, particle_range.start as u64, 0);
            let end_index = LLVMConstInt(int64_type, particle_range.end as u64, 0);

            // Math functions (TODO: implement function calls)
            // let math_double_func_type = LLVMFunctionType(double_type, [double_type].as_mut_ptr(), 1, 0);
            // let sqrt_func = LLVMAddFunction(module, cstr!("llvm.sqrt.f64"), math_double_func_type);

            // Declare utility functions
            let barrier_handler_type = LLVMFunctionType(void_type, [int64_type, int64_type].as_mut_ptr(), 2, 0);
            let call2rust_handler = LLVMAddFunction(module, cstr!("_call2rust_handler"), barrier_handler_type);
            let interaction_handler_type = LLVMFunctionType(void_type, [int64_type, int64_type, int64_type,
                LLVMPointerType(LLVMPointerType(int64_type, 0), 0),
                LLVMPointerType(LLVMPointerType(int64_type, 0), 0)].as_mut_ptr(), 5, 0);
            let interaction_handler = LLVMAddFunction(module, cstr!("_interaction_handler"), interaction_handler_type);
            let interaction_sync_handler = LLVMAddFunction(module, cstr!("_interaction_sync_handler"), barrier_handler_type);
            let end_of_step_handler_type = LLVMFunctionType(void_type, [int64_type].as_mut_ptr(), 1, 0);
            let end_of_step_handler = LLVMAddFunction(module, cstr!("_end_of_step"), end_of_step_handler_type);

            // Bake callback pointer into module for communication with Rust code
            let callback_target_ptrptr = LLVMAddGlobal(module, int64_type, cstr!("_callback_target_ptr"));
            let initializer = LLVMConstInt(int64_type, callback_target_ptr, 0);
            LLVMSetGlobalConstant(callback_target_ptrptr, 1);
            LLVMSetInitializer(callback_target_ptrptr, initializer);

            // For debugging
            let print_func_u64_type = LLVMFunctionType(void_type, [int64_type].as_mut_ptr(), 1, 0);
            #[allow(unused_variables)]
            let print_func_u64 = LLVMAddFunction(module, cstr!("print_u64"), print_func_u64_type);
            let print_func_f64_type = LLVMFunctionType(void_type, [double_type].as_mut_ptr(), 1, 0);
            #[allow(unused_variables)]
            let print_func_f64 = LLVMAddFunction(module, cstr!("print_f64"), print_func_f64_type);

            // Collect all external symbols
            let mut external_symbols = FIPS_FUNCS.iter()
                .chain(SYSTEM_FUNCS.iter())
                .map(|symbol_name| symbol_name.to_string())
                .collect::<Vec<_>>();
            for (_, function_def) in function_index.get_functions() {
                external_symbols.push(function_def.get_name().to_string());
            }

            // Set symbol values (no local variables so far, so everything is module-level)
            for (name, symbol) in symbol_table.iter_mut() {
                match &symbol.kind {
                    // Constants are just translated to global constants
                    FipsSymbolKind::Constant(const_val) => {
                        let llname = format!("constant_{}", name);
                        let llval = create_global_const(module, llname, const_val.clone());
                        symbol.set_value(LLSymbolValue::SimplePointer(llval));
                    }
                    // Particle members get translated differently depending on whether
                    // they are uniform or not:
                    // - Uniforms get translated to global constants
                    // - Per-particle members get translated to a global base address
                    //   as well as a load in the loop body
                    FipsSymbolKind::ParticleMember(member_id) => {
                        // Lookup member information in index and store
                        let member_definition = particle_index.get(particle_id).unwrap()
                            .get_member(&member_id).unwrap();
                        let member_data = particle_data.borrow_member(member_id).unwrap();
                        match &*member_data {
                            MemberData::Uniform(value) => {
                                let llname = format!("uniform_{}", name);
                                let llval = create_global_const(module, llname, value.clone());
                                symbol.set_value(LLSymbolValue::SimplePointer(llval));
                            },
                            MemberData::PerParticle{data, ..} => {
                                let llname = format!("base_addr_{}", name);
                                let llval = create_global_ptr(module, llname, member_definition.get_type(),
                                    data.as_ptr() as usize)?;
                                symbol.set_value(LLSymbolValue::ParticleMember{
                                    base_ptr: llval,
                                    local_ptr: None
                                });
                            }
                        }
                    }
                    FipsSymbolKind::Function(function_id) => {
                        let val = function_index.get(*function_id).unwrap().create_symbol_value(*function_id, context, module)?;
                        symbol.set_value(val);
                    }
                    _ => panic!("Faulty symbol table: global symbols must be either constants or particle members")
                }
            }

            // Extract the sqrt function (we need that one specifically)
            let sqrt_func = match &symbol_table.resolve_symbol(BuiltinFunction::Sqrt.get_name())
                .unwrap().value.as_ref().unwrap() 
            {
                LLSymbolValue::Function(LLFunctionSymbolValue { function, .. }) => *function,
                _ => panic!("Corrupted sqrt function."),
            };

            // Create neighbor stuff
            let interaction_values = neighbor_lists.iter()
                .filter_map(|(interaction_id, neighbor_list)| {
                    // Get all the members of particle types A and B of the interaction
                    let interaction = thread_context.executor_context.global_context.runtime.interaction_index.get(*interaction_id).unwrap();
                    let interaction_name = interaction.get_name();

                    let target_names_a = interaction.iter().map(|(_, quantity_def)| quantity_def.get_target_a())
                        .collect::<Vec<_>>();
                    let target_names_b = interaction.iter().map(|(_, quantity_def)| quantity_def.get_target_b())
                        .collect::<Vec<_>>();
                    let (type_a, type_a_def) = particle_index.get_particle_by_name(interaction.get_type_a()).unwrap();
                    let (type_b, type_b_def) = particle_index.get_particle_by_name(interaction.get_type_b()).unwrap();
                    let is_a = particle_id == type_a; // Are we type A? // TODO: Is this correct?

                    // TODO: Clean this up
                    if particle_id != type_a && particle_id != type_b {
                        return None;
                    }

                    // Get the namespaces for a and b
                    let namespace_a = interaction.get_name_a();
                    let namespace_b = interaction.get_name_b();
                    // Extract all non-quantity members
                    let members_a = type_a_def.get_members().map(|(_, member_def)| member_def.get_name())
                        .filter(|member_name| !target_names_a.contains(member_name)) // Exclude target names
                        .collect::<Vec<_>>();
                    let members_b = type_b_def.get_members().map(|(_, member_def)| member_def.get_name())
                        .filter(|member_name| !target_names_b.contains(member_name))
                        .collect::<Vec<_>>();
                    // Extract all quantity members
                    let quantity_members_a = type_a_def.get_members().map(|(_, member_def)| member_def.get_name())
                        .filter(|member_name| target_names_a.contains(member_name)) // Exclude target names
                        .collect::<Vec<_>>();
                    let quantity_members_b = type_b_def.get_members().map(|(_, member_def)| member_def.get_name())
                        .filter(|member_name| target_names_b.contains(member_name))
                        .collect::<Vec<_>>();
                    // Get the names of the position members
                    let position_member_a_name = type_a_def.get_position_member().unwrap().1.get_name();
                    let position_member_b_name = type_b_def.get_position_member().unwrap().1.get_name();
                    // For all members of A/B: try to resolve the member for each position block
                    let member_vals_a = create_neighbor_member_values(module, members_a, neighbor_list, particle_index, particle_store);
                    let member_vals_b = create_neighbor_member_values(module, members_b, neighbor_list, particle_index, particle_store);
                    let quantity_member_vals_a = create_neighbor_member_values(module, quantity_members_a, neighbor_list, particle_index, particle_store);
                    let quantity_member_vals_b = create_neighbor_member_values(module, quantity_members_b, neighbor_list, particle_index, particle_store);
                    // Select own position member name and other position member name
                    let ((own_namespace, own_position_name, own_member_vals, own_quantity_member_vals),
                        (other_namespace, other_position_name, other_member_vals)) = 
                        if is_a {
                            ((namespace_a, position_member_a_name, member_vals_a, quantity_member_vals_a),
                            (namespace_b, position_member_b_name, member_vals_b))
                        }
                        else {
                            ((namespace_b, position_member_b_name, member_vals_b, quantity_member_vals_b),
                            (namespace_a, position_member_a_name, member_vals_a))
                        };
                    // Create global arrays for member values
                    let vals_to_global_array = |mut llvals: Vec<LLVMValueRef>, name| {
                        let llelem_type = LLVMTypeOf(llvals[0]);
                        let llval = LLVMConstArray(llelem_type, llvals.as_mut_ptr(), llvals.len() as u32);
                        let llarray_type = LLVMTypeOf(llval);
                        let llglobal = LLVMAddGlobal(module, llarray_type, cstring!(name));
                        LLVMSetInitializer(llglobal, llval);
                        LLVMSetGlobalConstant(llglobal, 1);
                        llglobal
                    };
                    let own_members = own_member_vals.into_iter()
                        .map(|(member_name, llvals)| {
                            let name = format!("neigh_{}_own_{}", interaction_name, member_name);
                            (member_name.to_string(), vals_to_global_array(llvals, name))
                        }).collect::<HashMap<_,_>>();
                    let own_quantity_members = own_quantity_member_vals.into_iter()
                        .map(|(member_name, llvals)| {
                            let name = format!("neigh_{}_own_quantity_{}", interaction_name, member_name);
                            (member_name.to_string(), vals_to_global_array(llvals, name))
                        }).collect::<HashMap<_,_>>();
                    let other_members = other_member_vals.into_iter()
                        .map(|(member_name, llvals)| {
                            let name = format!("neigh_{}_other_{}", interaction_name, member_name);
                            (member_name.to_string(), vals_to_global_array(llvals, name))
                        }).collect::<HashMap<_,_>>();
                    // Create global values for other necessary information
                    // Cutoff length (needed for neighbor checking)
                    let cutoff = unwrap_f64_constant(&interaction.get_cutoff()).unwrap();
                    let cutoff_sqr = cutoff*cutoff;
                    let cutoff_sqr = LLVMConstReal(double_type, cutoff_sqr);
                    // Position block size (needed for index calculation)
                    let block_size_max = neighbor_list.pos_block_size;
                    let block_size_max = LLVMConstInt(int64_type, block_size_max as u64, 0);
                    // Position block index and block length of this worker (needed for counter-quantities)
                    let mut own_block_index = None;
                    let mut own_block_length = None;
                    for (i, (block_particle_id, block_particle_range)) in neighbor_list.pos_blocks.iter().enumerate() {
                        if particle_id == *block_particle_id && particle_range == *block_particle_range {
                            own_block_index = Some(i);
                            own_block_length = Some(block_particle_range.len());
                        }
                    }
                    let own_block_index = own_block_index.unwrap(); // This must be true if the neighbor list is correct
                    let own_block_length = own_block_length.unwrap();
                    let _own_block_index = own_block_index; // We need the numbers later as well
                    let _own_block_length = own_block_length;
                    let own_block_index = LLVMConstInt(int64_type, own_block_index as u64, 0);
                    let own_block_length = LLVMConstInt(int64_type, own_block_length as u64, 0);

                    // -- Generate interaction function --
                    /* General structure
                    interaction_func():
                        a = 0
                        for i in 0..block_length:
                            load all particle members for own from block own_block with offset i
                            b = neighbor_list_index[i]
                            for n in a..b:
                                j = neighbor_list[n]
                                j_block = j / block_size
                                j_offset = j % block_size
                                load all particle members for theirs from block j_block with offset j_offset
                                if distance_sqr < cutoff_sqr
                                    evaluate all interaction quantities
                            a = b
                    */

                    let name = format!("interaction_{}_func", interaction_name);
                    let interaction_func_type = LLVMFunctionType(void_type,
                        [LLVMPointerType(int64_type, 0), LLVMPointerType(int64_type, 0)].as_mut_ptr(), 2, 0);
                    let interaction_func = LLVMAddFunction(module, cstring!(name), interaction_func_type);
                    LLVMSetLinkage(interaction_func, LLVMLinkage::LLVMLinkerPrivateLinkage);
                    let interaction_func_entry = LLVMAppendBasicBlockInContext(context, interaction_func, cstr!("entry"));
                    LLVMPositionBuilderAtEnd(builder, interaction_func_entry);
                    // Parameters
                    let neighbor_list_index = LLVMGetParam(interaction_func, 0);
                    let neighbor_list = LLVMGetParam(interaction_func, 1);
                    // Local variables
                    let outer_index_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("i_ptr"));
                    let inner_index_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("n_ptr"));
                    let current_offset_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("a_ptr"));
                    let next_offset_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("b_ptr"));
                    let other_block_index_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("other_block_index_ptr"));
                    let other_offset_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("other_offset_ptr"));
                    let distance_sqr_ptr = LLVMBuildAlloca(builder, double_type, cstr!("dist_sqr_ptr"));
                    let alloca_members = |(member_name, llglobal), prefix: &str| {
                        let lltype = LLVMGetElementType(LLVMGetElementType(LLVMTypeOf(llglobal)));
                        let lltype = match LLVMGetTypeKind(lltype) {
                            LLVMTypeKind::LLVMPointerTypeKind => LLVMGetElementType(lltype),
                            _ => lltype
                        };
                        let llval = LLVMBuildAlloca(builder, lltype, cstring!(format!("{}_{}_ptr", prefix, member_name)));
                        (member_name, (llglobal, llval))
                    };
                    let extract_local_symbols = |statement_block: &Vec<Statement>| {
                        let mut local_symbols = SymbolTable::new();
                        for statement in statement_block {
                            match statement {
                                Statement::Let(let_stmt) => {
                                    let llval = LLSymbolValue::SimplePointer(create_local_ptr(module, builder,
                                        let_stmt.name.clone(), &let_stmt.typ).unwrap());
                                    local_symbols.add_local_symbol_with_value(let_stmt.name.clone(),
                                        let_stmt.typ.clone(), llval).unwrap();
                                },
                                _ => {}
                            }
                        }
                        local_symbols
                    };
                    let own_members = own_members.into_iter()
                        .map(|x| alloca_members(x, "own")).collect::<HashMap<_,_>>();
                    let other_members = other_members.into_iter()
                        .map(|x| alloca_members(x, "other")).collect::<HashMap<_,_>>();
                    let own_quantity_members = own_quantity_members.into_iter()
                        .map(|x| alloca_members(x, "own_quantity")).collect::<HashMap<_,_>>();
                    let common_local_symbols = match interaction.get_common_block() {
                        Some(statement_block) => extract_local_symbols(statement_block),
                        None => SymbolTable::new(),
                    };
                    let mut local_symbols = interaction.iter()
                        .map(|(quantity_id, quantity_def)| {
                            match quantity_def.get_expression() {
                                parser::Expression::Block(block) => {
                                    (quantity_id, extract_local_symbols(&block.statements))
                                }
                                _ => (quantity_id, SymbolTable::new())
                            }
                        }).collect::<HashMap<InteractionQuantityID, SymbolTable<_>>>();
                    let mut extra_symbols = SymbolTable::new();
                    let mut distance_ptr = None;
                    if let parser::Identifier::Named(distance_name) = interaction.get_distance_identifier() {
                        let llval = LLVMBuildAlloca(builder, double_type, cstr!("distance"));
                        distance_ptr = Some(llval);
                        extra_symbols.add_local_symbol_with_value(distance_name.clone(), FipsType::Double,
                            LLSymbolValue::SimplePointer(llval)).unwrap();
                    }
                    let mut distance_vec_ptr = None;
                    if let Some(distance_vec_name) = interaction.get_distance_vec() {
                        let lltyp = LLVMVectorType(double_type, domain.get_dim() as u32);
                        let llval = LLVMBuildAlloca(builder, lltyp, cstr!("distance"));
                        distance_vec_ptr = Some(llval);
                        extra_symbols.add_local_symbol_with_value(distance_vec_name.clone(),
                            FipsType::Array{ typ: Box::new(FipsType::Double), length: parser::CompileTimeConstant::Literal(domain.get_dim())},
                            LLSymbolValue::SimplePointer(llval)).unwrap();
                    }
                    // Initialize the loop variable
                    LLVMBuildStore(builder, LLVMConstInt(int64_type, 0, 0), outer_index_ptr);
                    LLVMBuildStore(builder, LLVMConstInt(int64_type, 0, 0), current_offset_ptr);
                    // Initialize all quantity fields
                    for (member_name, (llglobal, _)) in own_quantity_members.iter() {
                        // Load global
                        let loaded_global = LLVMBuildLoad(builder, *llglobal, cstr!(""));
                        // Get the correct pointer from the array
                        let llptr = LLVMBuildExtractValue(builder, loaded_global, _own_block_index as u32,
                            cstring!(format!("own_quantity_{}_block_ptr", member_name)));
                        // Make sure the quantity is not uniform (this should be caught way before anyway)
                        assert!(matches!(LLVMGetTypeKind(LLVMTypeOf(llptr)), LLVMTypeKind::LLVMPointerTypeKind));
                        // Get size of element type
                        let elem_size = LLVMSizeOf(LLVMGetElementType(LLVMTypeOf(llptr)));
                        let mem_size = LLVMBuildMul(builder, elem_size, own_block_length, cstr!("mem_size"));
                        // For SUM: memset quantity field to zero
                        // TODO: Fix the initial values for reduction methods other than SUM
                        let llptr = LLVMBuildBitCast(builder, llptr, LLVMPointerType(int8_type, 0), cstr!(""));
                        LLVMBuildMemSet(builder, llptr, LLVMConstInt(int8_type, 0, 0), mem_size, 1);
                    }

                    // Create loop
                    let block_exit = LLVMAppendBasicBlockInContext(context, interaction_func, cstr!("exit"));
                    let block_outer_loop_body = LLVMAppendBasicBlockInContext(context, interaction_func, cstr!("outer_loop_body"));
                    let block_inner_loop_body = LLVMAppendBasicBlockInContext(context, interaction_func, cstr!("inner_loop_body"));
                    let block_outer_loop_body_exit = LLVMAppendBasicBlockInContext(context, interaction_func, cstr!("outer_loop_body_exit"));
                    let block_outer_loop_increment = build_loop(context, builder, block_outer_loop_body,
                        block_exit, outer_index_ptr, own_block_length);

                    // Close outer loop
                    LLVMPositionBuilderAtEnd(builder, block_outer_loop_body_exit);
                    // Write all accumulators back to the global array
                    for (member_name, (llglobal, lllocal)) in own_quantity_members.iter() {
                        // Load global
                        let loaded_global = LLVMBuildLoad(builder, *llglobal, cstr!(""));
                        // Get the correct pointer from the array
                        let llptr = LLVMBuildExtractValue(builder, loaded_global, _own_block_index as u32,
                            cstring!(format!("own_quantity_{}_block_ptr", member_name)));

                        // Get writeback pointer
                        let outer_index = LLVMBuildLoad(builder, outer_index_ptr, cstr!("i"));
                        let llptr_writeback = LLVMBuildGEP(builder, llptr, [outer_index].as_mut_ptr(), 1,
                            cstring!(format!("writeback_ptr_{}", member_name)));

                        // Add local accumulator to global
                        let llglobalval = LLVMBuildLoad(builder, llptr_writeback, cstring!(format!("global_val_{}", member_name)));
                        let llacc = LLVMBuildLoad(builder, *lllocal, cstring!(format!("acc_{}", member_name)));
                        let llval = convert_to_scalar_or_array(context, builder, 
                            evaluate_binop(context, builder, llglobalval, llacc, parser::BinaryOperator::Add).unwrap());

                        // Store result
                        LLVMBuildStore(builder, llval, llptr_writeback);
                    }
                    let next_offset = LLVMBuildLoad(builder, next_offset_ptr, cstr!("next_offset"));
                    LLVMBuildStore(builder, next_offset, current_offset_ptr);
                    LLVMBuildBr(builder, block_outer_loop_increment);

                    LLVMPositionBuilderAtEnd(builder, block_outer_loop_body);
                    let outer_index = LLVMBuildLoad(builder, outer_index_ptr, cstr!("i"));
                    // Reset quantity accumulators
                    for (_, (_, lllocal)) in own_quantity_members.iter() {
                        // Get size of element type
                        let elem_size = LLVMSizeOf(LLVMGetElementType(LLVMTypeOf(*lllocal)));
                        // TODO: Fix the initial values for reduction methods other than SUM
                        // For SUM: memset quantity field to zero
                        let llptr = LLVMBuildBitCast(builder, *lllocal, LLVMPointerType(int8_type, 0), cstr!(""));
                        LLVMBuildMemSet(builder, llptr, LLVMConstInt(int8_type, 0, 0), elem_size, 1);
                    }
                    // Load all members from own member list
                    let load_members = |(member_name, (llglobal, lllocal)): (String, (LLVMValueRef, LLVMValueRef)),
                        block_index: LLVMValueRef, particle_index: LLVMValueRef, infix: &str| 
                    {
                        // Load global
                        let loaded_global = LLVMBuildLoad(builder, llglobal, cstr!(""));
                        // Cast array to pointer
                        let ptrtype = LLVMPointerType(LLVMGetElementType(LLVMTypeOf(loaded_global)), 0);
                        let global_ptr = LLVMBuildBitCast(builder, llglobal, ptrtype, cstr!("global_ptr"));
                        // Load correct entry from array
                        let llptr = LLVMBuildGEP(builder, global_ptr, [block_index].as_mut_ptr(), 1, cstr!(""));
                        let llval = LLVMBuildLoad(builder, llptr, cstring!(format!("loaded_{}_{}", infix, member_name)));
                        // Non uniforms are still pointers at this point
                        let llval = match  LLVMGetTypeKind(LLVMTypeOf(llval)) {
                            LLVMTypeKind::LLVMPointerTypeKind => {
                                let llptr = LLVMBuildGEP(builder, llval, [particle_index].as_mut_ptr(), 1, cstr!(""));
                                LLVMBuildLoad(builder, llptr, cstring!(format!("really_loaded_{}_{}", infix, member_name)))
                            },
                            _ => llval
                        };
                        LLVMBuildStore(builder, llval, lllocal);
                        (member_name, lllocal)
                    };
                    let own_members_loaded = own_members.into_iter()
                        .map(|x| load_members(x, own_block_index, outer_index, "own")).collect::<HashMap<_,_>>();
                    // Load next offset from neighbor list
                    let llptr = LLVMBuildGEP(builder, neighbor_list_index, [outer_index].as_mut_ptr(), 1, cstr!(""));
                    let next_offset = LLVMBuildLoad(builder, llptr, cstr!("next_offset"));
                    LLVMBuildStore(builder, next_offset, next_offset_ptr);
                    // Create inner loop
                    let current_offset = LLVMBuildLoad(builder, current_offset_ptr, cstr!("previous_offset"));
                    LLVMBuildStore(builder, current_offset, inner_index_ptr);
                    let block_inner_loop_increment = build_loop(context, builder, block_inner_loop_body,
                        block_outer_loop_body_exit, inner_index_ptr, next_offset_ptr);
                    // Inner loop body
                    LLVMPositionBuilderAtEnd(builder, block_inner_loop_body);
                    let inner_index = LLVMBuildLoad(builder, inner_index_ptr, cstr!("n"));
                    // Get other particle's block and local index
                    let llptr = LLVMBuildGEP(builder, neighbor_list, [inner_index].as_mut_ptr(), 1, cstr!("j_ptr"));
                    let other_particle_index = LLVMBuildLoad(builder, llptr, cstr!("j"));
                    let other_block_index = LLVMBuildUDiv(builder, other_particle_index, block_size_max, cstr!("other_block"));
                    LLVMBuildStore(builder, other_block_index, other_block_index_ptr);
                    let other_offset = LLVMBuildURem(builder, other_particle_index, block_size_max, cstr!("other_offset"));
                    LLVMBuildStore(builder, other_offset, other_offset_ptr);
                    // Load members of other particle
                    let other_members_loaded = other_members.into_iter()
                        .map(|x| load_members(x, other_block_index, other_offset, "other")).collect::<HashMap<_,_>>();
                    // Get both (raw) positions
                    let own_position = LLVMBuildLoad(builder, *own_members_loaded.get(own_position_name).unwrap(), cstr!("own_position"));
                    let other_position = LLVMBuildLoad(builder, *other_members_loaded.get(other_position_name).unwrap(), cstr!("other_position"));
                    // Correct the other position if necessary
                    let cutoff_skin = cutoff * thread_context.executor_context
                        .global_context.runtime.enabled_interactions.get(interaction_id).unwrap().skin_factor;
                    let other_position = correct_postion_vector(context, builder, own_position, other_position,
                        cutoff_skin, domain);
                    // Calculate the distance vector
                    let (distance_sqr, distance_vec) = if is_a {
                            calculate_distance_sqr_and_vec(context, builder, own_position, other_position)
                        }
                        else {
                            calculate_distance_sqr_and_vec(context, builder, other_position, own_position)
                        };
                    LLVMBuildStore(builder, distance_sqr, distance_sqr_ptr);
                    if let Some(distance_vec_ptr) = distance_vec_ptr {
                        LLVMBuildStore(builder, distance_vec, distance_vec_ptr);
                    }
                    // Test if distance is smaller than cutoff
                    let within_cutoff = LLVMBuildFCmp(builder, LLVMRealPredicate::LLVMRealOLT,
                        distance_sqr, cutoff_sqr, cstr!("within_cutoff"));
                    let block_process_interaction = LLVMInsertBasicBlockInContext(context, block_inner_loop_increment, cstr!("process_interaction"));
                    LLVMBuildCondBr(builder, within_cutoff, block_process_interaction, block_inner_loop_increment);
                    // Process the interaction
                    LLVMPositionBuilderAtEnd(builder, block_process_interaction);

                    // Prepare the symbol table
                    let particle_symbols = symbol_table.pop_table().unwrap();
                    // Calculate distance if needed
                    if let Some(distance_ptr) = distance_ptr {
                        let distance_sqr = LLVMBuildLoad(builder, distance_sqr_ptr, cstr!("dist_sqr"));
                        let distance = LLVMBuildCall(builder, sqrt_func, [distance_sqr].as_mut_ptr(), 1, cstr!("dist"));
                        LLVMBuildStore(builder, distance, distance_ptr);
                    }
                    symbol_table.push_table(extra_symbols);
                    symbol_table.push_table(common_local_symbols);
                    // Prepare the namespace maps
                    let mut namespace_symbols = HashMap::new();
                    if let parser::Identifier::Named(own_namespace) = own_namespace {
                        namespace_symbols.insert(own_namespace, own_members_loaded);
                    }
                    if let parser::Identifier::Named(other_namespace) = other_namespace {
                        namespace_symbols.insert(other_namespace, other_members_loaded);
                    }
                    // Statement processing in quantity blocks
                    let handle_statement = |statement: &Statement, symbol_table: &mut SymbolTable<_>| {
                        match statement {
                            Statement::Let(_) | Statement::Assign(_) => {
                                let (target_name, expression) = match statement{
                                    Statement::Let(statement) => (&statement.name, &statement.initial),
                                    Statement::Assign(statement) => {
                                        if let Some(_) = statement.index {
                                            unimplemented!("Indexed assignment in quantities not supported yet");
                                        }
                                        (&statement.assignee, &statement.value)
                                    },
                                    _ => unreachable!()
                                };
                                let target = symbol_table.resolve_symbol(target_name)
                                    .expect(&format!("Unresolved identifier {}", target_name));
                                let target = match target.kind {
                                    FipsSymbolKind::Constant(_) => { panic!("Cannot assign to constant.") }
                                    FipsSymbolKind::Function(_) => { panic!("Cannot assign to function.") }
                                    FipsSymbolKind::ParticleMember(_) => { panic!("Cannot assign to particle member during interaction.")  }
                                    FipsSymbolKind::LocalVariable(_) => {
                                        // The actual symbol resolution is the same for local variables
                                        // and particle members
                                        match target.value.as_ref().unwrap() {
                                            LLSymbolValue::SimplePointer(ptr) => { *ptr }
                                            LLSymbolValue::ParticleMember { .. } | LLSymbolValue::Function { .. } => {
                                                panic!("Malformed symbol table in interaction!")
                                            }
                                        }
                                    }
                                };
                                let value = evaluate_expression(context, builder, expression,
                                    &symbol_table, &namespace_symbols, function_index, callback_target_ptrptr)
                                        .expect("Cannot evaluate interaction expression");
                                LLVMBuildStore(builder, convert_to_scalar_or_array(context, builder, value), target);
                            }
                            Statement::Update(_) => { panic!("Update statements are not allowed in interactions!") }
                            Statement::Call(_) => { panic!("Call statements are not allowed in interactions!") }
                        }
                    };

                    // Process the common block first
                    if let Some(statement_block) = interaction.get_common_block() {
                        for statement in statement_block {
                            handle_statement(statement, &mut symbol_table);
                        }
                    }
                    // Then all the quantity blocks
                    for (quantity_id, quantity_def) in interaction.iter() {
                        symbol_table.push_table(local_symbols.remove(&quantity_id).unwrap());
                        match quantity_def.get_expression() {
                            parser::Expression::Block(block) => {
                                for statement in &block.statements {
                                    handle_statement(statement, &mut symbol_table);
                                }
                                // Evaluate the final expression
                                let quantity_value = evaluate_expression(context, builder, &block.expression, 
                                    &symbol_table, &namespace_symbols, function_index, callback_target_ptrptr)
                                        .expect("Cannot evaluate interaction expression");

                                // Choose value for own particle
                                let value = match quantity_def.get_symmetry() {
                                    // Symmetric? => own particle and other particle get same value
                                    parser::InteractionSymmetry::Symmetric => { quantity_value },
                                    // Antisymmetric? If we are not A, we might need to multiply by -1
                                    parser::InteractionSymmetry::Antisymmetric => {
                                        if !is_a {
                                            llmultiply_by_minus_one(context, builder, quantity_value)
                                        }
                                        else {
                                            quantity_value
                                        }
                                    },
                                    // Symmetric? Then we must split the value into two parts and take one
                                    parser::InteractionSymmetry::Asymmetric => {
                                        let quantity_value = convert_to_scalar_or_array(context, builder, quantity_value);
                                        // TODO: Do typechecking earlier
                                        let lltyp = LLVMTypeOf(quantity_value);
                                        assert!(matches!(LLVMGetTypeKind(lltyp), LLVMTypeKind::LLVMArrayTypeKind));
                                        assert_eq!(LLVMGetArrayLength(lltyp), 2);
                                        // Extract correct value
                                        let idx = if is_a {0} else {1};
                                        LLVMBuildExtractValue(builder, quantity_value, idx, cstr!("quantity_part_own"))
                                    },
                                };
                                // Get pointers for this quantity
                                // TODO: Support for other reduction methods
                                assert!(matches!(quantity_def.get_reduction_method(), parser::ReductionMethod::Sum));
                                let target_name = if is_a { quantity_def.get_target_a() } else { quantity_def.get_target_b() };
                                let (llglobal, lllocal) = own_quantity_members.get(target_name).unwrap();
                                // Add value to accumulator
                                let accval = LLVMBuildLoad(builder, *lllocal, cstring!(format!("acc_{}", quantity_def.get_name())));
                                let writeback_value = convert_to_scalar_or_array(context, builder,
                                    evaluate_binop(context, builder, accval, value, parser::BinaryOperator::Add).unwrap());
                                if LLVMTypeOf(writeback_value) != LLVMGetElementType(LLVMTypeOf(*lllocal)) {
                                    panic!("Mismatched type in return expression for interaction quantity {}", quantity_def.get_name());
                                }
                                LLVMBuildStore(builder, writeback_value, *lllocal);

                                // If the other particle is also in our block, we need to write the result to that
                                // particle too (but directly, since we have no accumulator)
                                let block_particle_store = LLVMInsertBasicBlockInContext(context, block_inner_loop_increment,
                                    cstring!(format!("partner_store_{}", quantity_def.get_name())));
                                let block_next_quantity = LLVMInsertBasicBlockInContext(context, block_inner_loop_increment,
                                    cstring!(format!("quantity_after_{}", quantity_def.get_name())));
                                let other_block_index = LLVMBuildLoad(builder, other_block_index_ptr, cstr!("other_block_index"));
                                let is_same_block = LLVMBuildICmp(builder, LLVMIntPredicate::LLVMIntEQ,
                                    own_block_index, other_block_index, cstr!("is_same_block"));
                                LLVMBuildCondBr(builder, is_same_block, block_particle_store, block_next_quantity);
                                // Add counter value to global pointer at other_particle_offset
                                LLVMPositionBuilderAtEnd(builder, block_particle_store);

                                let value = match quantity_def.get_symmetry() {
                                    // Symmetric? => own particle and other particle get same value
                                    parser::InteractionSymmetry::Symmetric => { quantity_value },
                                    // Antisymmetric? If we are A, we might need to multiply by -1
                                    parser::InteractionSymmetry::Antisymmetric => {
                                        if is_a {
                                            llmultiply_by_minus_one(context, builder, quantity_value)
                                        }
                                        else {
                                            quantity_value
                                        }
                                    },
                                    // Symmetric? Then we must split the value into two parts
                                    parser::InteractionSymmetry::Asymmetric => {
                                        let quantity_value = convert_to_scalar_or_array(context, builder, quantity_value);
                                        // No need to type check again...
                                        // Extract correct value
                                        let idx = if is_a {1} else {0};
                                        LLVMBuildExtractValue(builder, quantity_value, idx, cstr!("quantity_part_own"))
                                    }
                                };
                                // Load global
                                let loaded_global = LLVMBuildLoad(builder, *llglobal, cstr!(""));
                                // Get the correct pointer from the array
                                let llptr = LLVMBuildExtractValue(builder, loaded_global, _own_block_index as u32,
                                    cstring!(format!("own_quantity_{}_block_ptr", target_name)));
                                // Get writeback pointer
                                let other_offset = LLVMBuildLoad(builder, other_offset_ptr, cstr!("other_offset"));
                                let llptr_writeback = LLVMBuildGEP(builder, llptr, [other_offset].as_mut_ptr(), 1,
                                    cstring!(format!("writeback_ptr_{}", target_name)));
                                // Add value to global
                                let llglobalacc = LLVMBuildLoad(builder, llptr_writeback, cstring!(format!("global_val_{}", target_name)));
                                let llval = convert_to_scalar_or_array(context, builder, 
                                    evaluate_binop(context, builder, llglobalacc, value, parser::BinaryOperator::Add).unwrap());
                                LLVMBuildStore(builder, llval, llptr_writeback);

                                LLVMBuildBr(builder, block_next_quantity);
                                LLVMPositionBuilderAtEnd(builder, block_next_quantity);

                                // if thread_context.particle_range.start != 0 {
                                //     let other_block_index = LLVMBuildLoad(builder, other_block_index_ptr, cstr!("other_block_index"));
                                //     let foo = llmultiply_by_minus_one(context, builder, LLVMConstInt(int64_type, 1, 0));
                                //     LLVMBuildCall(builder, print_func_u64, [foo].as_mut_ptr(), 1, cstr!(""));
                                // }

                                // if thread_context.particle_range.start != 0 {
                                //     // let other_block_index = LLVMBuildLoad(builder, other_block_index_ptr, cstr!("other_block_index"));
                                //     // let outer_index = LLVMBuildLoad(builder, outer_index_ptr, cstr!("other_block_index"));
                                //     // // let foo = llmultiply_by_minus_one(context, builder, LLVMConstInt(int64_type, 1, 0));
                                //     //let val = LLVMBuildExtractValue(builder, value, 0, cstr!(""));
                                //     let val = LLVMBuildExtractElement(builder, value, LLVMConstInt(int64_type, 0, 0), cstr!(""));
                                //     // LLVMBuildCall(builder, print_func_u64, [outer_index].as_mut_ptr(), 1, cstr!(""));
                                //     // LLVMBuildCall(builder, print_func_u64, [other_block_index].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_f64, [val].as_mut_ptr(), 1, cstr!(""));
                                // }

                                // if thread_context.particle_range.start == 0 {
                                //     let outer_index = LLVMBuildLoad(builder, outer_index_ptr, cstr!("other_block_index"));
                                //     let valx = LLVMBuildExtractElement(builder, other_position, LLVMConstInt(int64_type, 0, 0), cstr!(""));
                                //     let valy = LLVMBuildExtractElement(builder, other_position, LLVMConstInt(int64_type, 1, 0), cstr!(""));
                                //     let valz = LLVMBuildExtractElement(builder, other_position, LLVMConstInt(int64_type, 2, 0), cstr!(""));
                                //     let foo = LLVMBuildZExt(builder, other_correction, int64_type, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_u64, [outer_index].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_u64, [other_particle_index].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_u64, [foo].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_f64, [valx].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_f64, [valy].as_mut_ptr(), 1, cstr!(""));
                                //     LLVMBuildCall(builder, print_func_f64, [valz].as_mut_ptr(), 1, cstr!(""));
                                // }

                                // LLVMPositionBuilderAtEnd(builder, block_else);
                            }
                            _ => todo!()
                        }
                        symbol_table.pop_table(); // local symbols
                    }
                    symbol_table.pop_table(); // common local symbols
                    symbol_table.pop_table(); // extra symbols

                    // Merge with inner loop again
                    LLVMBuildBr(builder, block_inner_loop_increment);
                    
                    // Restore the symbol table
                    symbol_table.push_table(particle_symbols);

                    LLVMPositionBuilderAtEnd(builder, block_exit);
                    LLVMBuildRetVoid(builder);

                    let code = InteractionValues {
                        own_pos_block_index: own_block_index,
                        interaction_func
                    };
                    Some((interaction_id, code))
                })
                .collect::<HashMap<_,_>>();

            // Build worker main function
            let worker_main_type = LLVMFunctionType(void_type, std::ptr::null_mut(), 0, 0);
            let main_function = LLVMAddFunction(module, cstring!(WORKER_MAIN_NAME), worker_main_type);
            let main_entry = LLVMAppendBasicBlockInContext(context, main_function, cstr!("entry"));
            
            // Position builder
            LLVMPositionBuilderAtEnd(builder, main_entry);

            // Allocate space for neighbor list pointers
            let neighbor_list_index_var = LLVMBuildAlloca(builder, 
                LLVMPointerType(int64_type, 0), cstr!("neighbor_list_index_var"));
            let neighbor_list_var = LLVMBuildAlloca(builder, 
                LLVMPointerType(int64_type, 0), cstr!("neighbor_list_var"));

            for node in &timeline.nodes {
                match node {
                    SimulationNode::StatementBlock(node) => {
                        // Create a new function
                        let node_func = LLVMAddFunction(module, cstr!("node_func"), worker_main_type);
                        LLVMSetLinkage(node_func, LLVMLinkage::LLVMLinkerPrivateLinkage);
                        let block_node_main_entry = LLVMAppendBasicBlockInContext(context, node_func, cstr!("entry"));

                        // Position builder at the end of the new function
                        LLVMPositionBuilderAtEnd(builder, block_node_main_entry);

                        // Create allocas for local symbols
                        let mut local_symbols: SymbolTable<LLSymbolValue> = node.local_symbols.clone().convert();
                        for (name, symbol) in local_symbols.iter_mut() {
                            match &symbol.kind {
                                FipsSymbolKind::LocalVariable(typ) => {
                                    match typ {
                                        FipsType::Double => {
                                            let typ = &typ.clone();
                                            symbol.set_value(LLSymbolValue::SimplePointer(
                                                create_local_ptr(module, builder, name.clone(), typ)?
                                            ));
                                        }
                                        _ => todo!() // TODO!
                                    }
                                },
                                _ => panic!("Faulty symbol table: found non-local-variable symbol in local symbols")
                            }
                        }
                        // Push local symbols onto global symbols
                        symbol_table.push_table(local_symbols);

                        // Allocate space for the particle members
                        for (name, symbol) in symbol_table.iter_mut() {
                            match &symbol.kind {
                                FipsSymbolKind::ParticleMember(member_id) => {
                                    let name = format!("current_{}", name);
                                    let member_definition = particle_index.get(particle_id).unwrap()
                                        .get_member(&member_id).unwrap();
                                    match symbol.value.as_mut().unwrap() {
                                        LLSymbolValue::ParticleMember { local_ptr, .. } => {
                                            *local_ptr = Some(create_local_ptr(module, builder, name, &member_definition.get_type())?)
                                        }
                                        // Ignore uniform members
                                        LLSymbolValue::SimplePointer(_) | LLSymbolValue::Function { .. } => {}
                                    };
                                }
                                FipsSymbolKind::Constant(_) => {}
                                FipsSymbolKind::LocalVariable(_) => {}
                                FipsSymbolKind::Function(_) => {},
                            }
                        }

                        // Create loop variable
                        let loop_index_ptr = LLVMBuildAlloca(builder, int64_type, cstr!("loop_var"));

                        // -- Create the loop structure --

                        // Initialize the loop variable
                        LLVMBuildStore(builder, start_index, loop_index_ptr);

                        // Create loop blocks
                        let block_loop_check = LLVMAppendBasicBlockInContext(context, node_func, cstr!("loop_check"));
                        let block_loop_body = LLVMAppendBasicBlockInContext(context, node_func, cstr!("loop_body"));
                        let block_loop_increment = LLVMAppendBasicBlockInContext(context, node_func, cstr!("loop_increment"));
                        let block_after_loop = LLVMAppendBasicBlockInContext(context, node_func, cstr!("after_loop"));

                        // Create loop check
                        LLVMBuildBr(builder, block_loop_check);
                        LLVMPositionBuilderAtEnd(builder, block_loop_check);
                        let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));
                        let comparison = LLVMBuildICmp(builder, LLVMIntPredicate::LLVMIntULT, loop_index, end_index, cstr!("loop_check"));
                        LLVMBuildCondBr(builder, comparison, block_loop_body, block_after_loop);

                        // Create loop increment
                        LLVMPositionBuilderAtEnd(builder, block_loop_increment);
                        let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));
                        let llone = LLVMConstInt(int64_type, 1, 0);
                        let incremented_index = LLVMBuildAdd(builder, loop_index, llone, cstr!("incremented_val"));
                        LLVMBuildStore(builder, incremented_index, loop_index_ptr);
                        LLVMBuildBr(builder, block_loop_check);

                        // Create loop body
                        LLVMPositionBuilderAtEnd(builder, block_loop_body);

                        // Load all particle members
                        let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));
                        for (name, symbol) in symbol_table.iter() {
                            match &symbol.kind {
                                FipsSymbolKind::ParticleMember(_) => {                                    
                                    match symbol.value.as_ref().unwrap() {
                                        LLSymbolValue::ParticleMember { base_ptr, local_ptr, .. } => {
                                            let llname = format!("base_addr_loaded_{}", &name);
                                            let base_ptr = LLVMBuildLoad(builder, *base_ptr, cstring!(llname));
                                            let llname = format!("current_ptr_{}", &name);
                                            let current_ptr = LLVMBuildGEP(builder, base_ptr, [loop_index].as_mut_ptr(), 1, cstring!(llname));

                                            let llname = format!("loaded_{}", &name);
                                            let llval = LLVMBuildLoad(builder, current_ptr, cstring!(llname));
                                            LLVMBuildStore(builder, llval, local_ptr.unwrap());
                                        }
                                        // Ignore uniform members
                                        LLSymbolValue::SimplePointer(_) | LLSymbolValue::Function { .. } => {}
                                    };
                                }
                                FipsSymbolKind::Constant(_) => {}
                                FipsSymbolKind::LocalVariable(_) => {}
                                FipsSymbolKind::Function(_) => {},
                            }
                        }

                        let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));

                        // Process all statements
                        let mut members_changed = HashSet::new(); // Keep track of particle members that are assigned to
                        for statement in &node.statements {
                            match statement {
                                // TODO: Validate let statements beforehand (right now we can use the variable before its definition)
                                Statement::Let(_) | Statement::Assign(_) => { 
                                    let (target_name, expression) = match statement{
                                        Statement::Let(statement) => (&statement.name, &statement.initial),
                                        Statement::Assign(statement) => (&statement.assignee, &statement.value),
                                        _ => unreachable!()
                                    };
                                    let target = symbol_table.resolve_symbol(target_name)
                                        .ok_or(anyhow!("Unresolved identifier {}", target_name))?;
                                    let target = match target.kind {
                                        FipsSymbolKind::Constant(_) => { return Err(anyhow!("Cannot assign to constant.")) }
                                        FipsSymbolKind::Function(_) => { return Err(anyhow!("Cannot assign to function.")) }
                                        FipsSymbolKind::LocalVariable(_) | FipsSymbolKind::ParticleMember(_) => {
                                            // Update changed members set
                                            match target.kind {
                                                FipsSymbolKind::ParticleMember(member_id) => {
                                                    members_changed.insert(member_id);
                                                }
                                                _ => {}
                                            }
                                            // The actual symbol resolution is the same for local variables
                                            // and particle members
                                            match target.value.as_ref().unwrap() {
                                                LLSymbolValue::SimplePointer(ptr) => { *ptr }
                                                LLSymbolValue::ParticleMember { local_ptr, .. } => {
                                                    local_ptr.unwrap()
                                                }
                                                LLSymbolValue::Function { .. } => panic!("Target of let or assign statement cannot be a function!")
                                            }
                                        }
                                    };
                                    let value = evaluate_expression(context, builder, expression, 
                                        &symbol_table, &HashMap::new(), function_index, callback_target_ptrptr)?;
                                    match statement {
                                        // Let statements and non-indexed assign statements just store to target
                                        Statement::Let(_) | Statement::Assign(AssignStatement {index: None, ..}) 
                                            => {
                                                LLVMBuildStore(builder, convert_to_scalar_or_array(context, builder, value), target);
                                            },
                                        // Indexed assign statements load, insert and store
                                        Statement::Assign(AssignStatement {index: Some(index), ..}) => {
                                            // Assert that assignment target is actually indexable 
                                            // (TODO: do this during typechecking)
                                            let lltyp = LLVMGetElementType(LLVMTypeOf(target));
                                            match LLVMGetTypeKind(lltyp) {
                                                LLVMTypeKind::LLVMArrayTypeKind => {
                                                    let elemtyp = LLVMGetElementType(lltyp);
                                                    match LLVMGetTypeKind(elemtyp) {
                                                        LLVMTypeKind::LLVMDoubleTypeKind | LLVMTypeKind::LLVMIntegerTypeKind => {}
                                                        _ => unimplemented!("Multidimensional assignment not supported")
                                                    }
                                                },
                                                _ => panic!("Trying to index non-array type (identifier {})!", target_name)
                                            }

                                            let name = format!("old_{}_assign", target_name);
                                            let llval = LLVMBuildLoad(builder, target, cstring!(name));
                                            let name = format!("new_{}_assign", target_name);
                                            let llval = LLVMBuildInsertValue(builder, llval, value, 
                                                unwrap_usize_constant(index)? as u32, cstring!(name));
                                            LLVMBuildStore(builder, llval, target);
                                        },
                                        _ => unreachable!(),
                                    }
                                    
                                }
                                Statement::Update(_) | Statement::Call(_) 
                                    => panic!("Update and call statements not eliminated in simgraph construction")
                            }
                        }

                        // Write all particle members back to memory
                        for (name, symbol) in symbol_table.iter() {
                            match &symbol.kind {
                                FipsSymbolKind::ParticleMember(member_id) => {
                                    // Help LLVM a bit: No need to store a particle member if it
                                    // was never assigned to
                                    if members_changed.contains(member_id) {
                                        match symbol.value.as_ref().unwrap() {
                                            LLSymbolValue::ParticleMember { base_ptr, local_ptr, .. } => {
                                                let llname = format!("base_addr_loaded_{}", &name);
                                                let base_ptr = LLVMBuildLoad(builder, *base_ptr, cstring!(llname));
                                                let llname = format!("current_ptr_{}", &name);
                                                let current_ptr = LLVMBuildGEP(builder, base_ptr, [loop_index].as_mut_ptr(), 1, cstring!(llname));
                                                let llname = format!("final_{}", &name);
                                                let mut llval = LLVMBuildLoad(builder, local_ptr.unwrap(), cstring!(llname));
                                                // Special treatment of positions: Periodic correction
                                                // TODO: make this more configurable
                                                if particle_index.get(particle_id).unwrap()
                                                    .get_member(member_id).unwrap()
                                                    .is_position() 
                                                {
                                                    match domain {
                                                        Domain::Dim2{x,y} => {
                                                            assert!(matches!(x.oob, OutOfBoundsBehavior::Periodic));
                                                            assert!(matches!(y.oob, OutOfBoundsBehavior::Periodic));
                                                        }
                                                        Domain::Dim3{x,y,z} => {
                                                            assert!(matches!(x.oob, OutOfBoundsBehavior::Periodic));
                                                            assert!(matches!(y.oob, OutOfBoundsBehavior::Periodic));
                                                            assert!(matches!(z.oob, OutOfBoundsBehavior::Periodic));
                                                        }
                                                    }
                                                    let lldomain_lo = match domain {
                                                        Domain::Dim2{x,y} => {
                                                            LLVMConstVector([
                                                                LLVMConstReal(double_type, x.low),
                                                                LLVMConstReal(double_type, y.low),
                                                            ].as_mut_ptr(), 2)
                                                        }
                                                        Domain::Dim3{x,y,z} => {
                                                            LLVMConstVector([
                                                                LLVMConstReal(double_type, x.low),
                                                                LLVMConstReal(double_type, y.low),
                                                                LLVMConstReal(double_type, z.low),
                                                            ].as_mut_ptr(), 3)
                                                        }
                                                    };
                                                    let lldomain_size = match domain {
                                                        Domain::Dim2{x,y} => {
                                                            LLVMConstVector([
                                                                LLVMConstReal(double_type, x.size()),
                                                                LLVMConstReal(double_type, y.size()),
                                                            ].as_mut_ptr(), 2)
                                                        }
                                                        Domain::Dim3{x,y,z} => {
                                                            LLVMConstVector([
                                                                LLVMConstReal(double_type, x.size()),
                                                                LLVMConstReal(double_type, y.size()),
                                                                LLVMConstReal(double_type, z.size()),
                                                            ].as_mut_ptr(), 3)
                                                        }
                                                    };
                                                    llval = evaluate_binop(context, builder, llval,
                                                        lldomain_lo, parser::BinaryOperator::Sub)?;
                                                    llval = LLVMBuildFRem(builder, llval, lldomain_size, cstr!(""));
                                                    llval = LLVMBuildFAdd(builder, llval, lldomain_size, cstr!(""));
                                                    llval = LLVMBuildFRem(builder, llval, lldomain_size, cstr!(""));
                                                    llval = LLVMBuildFAdd(builder, llval, lldomain_lo, cstr!(""));
                                                    llval = convert_to_scalar_or_array(context, builder, llval);
                                                }

                                                LLVMBuildStore(builder, llval, current_ptr);
                                            }
                                            // Ignore uniform members
                                            LLSymbolValue::SimplePointer(_) => {}
                                            LLSymbolValue::Function { .. } => panic!("Particle member symbol has function value")
                                        };
                                    }
                                }
                                FipsSymbolKind::Constant(_) => {}
                                FipsSymbolKind::LocalVariable(_) => {}
                                FipsSymbolKind::Function(_) => {},
                            }
                        }

                        LLVMBuildBr(builder, block_loop_increment);

                        // Return instruction
                        LLVMPositionBuilderAtEnd(builder, block_after_loop);
                        LLVMBuildRetVoid(builder);
                        
                        // Pop local symbol table
                        symbol_table.pop_table();

                        LLVMPositionBuilderAtEnd(builder, main_entry);
                        LLVMBuildCall(builder, node_func, std::ptr::null_mut(), 0, cstr!(""));
                    }
                    SimulationNode::CommonBarrier(barrier_id) => {
                        // Barrier "nodes"
                        match &thread_context.executor_context.global_context.simgraph.barriers.get(*barrier_id).unwrap().kind {
                            // Call2Rust barriers just wait for the callback thread to finish and continue
                            BarrierKind::CallBarrier(_) => {
                                let barrier_data = barrier_id.data().as_ffi();
                                let llbarrier_data = LLVMConstInt(int64_type, barrier_data, 0);
                                let callback_target_param = LLVMBuildLoad(builder, callback_target_ptrptr, cstr!("tmp"));
                                LLVMBuildCall(builder, call2rust_handler, [callback_target_param, llbarrier_data].as_mut_ptr(), 2, cstr!(""));
                            }
                            BarrierKind::InteractionBarrier(interaction_id, quantity_id) => {
                                if quantity_id.is_some() {
                                    unimplemented!();
                                }
                                let interaction_vals = interaction_values.get(interaction_id); // TODO: Do this better
                                if let Some(interaction_vals) = interaction_vals {
                                    let barrier_data = barrier_id.data().as_ffi();
                                    let llbarrier_data = LLVMConstInt(int64_type, barrier_data, 0);
                                    let callback_target_param = LLVMBuildLoad(builder, callback_target_ptrptr, cstr!("tmp"));
                                    let block_index = interaction_vals.own_pos_block_index;
                                    // Call Rust (for potential neighbor list update)
                                    LLVMBuildCall(builder, interaction_handler, [callback_target_param, llbarrier_data, block_index,
                                            neighbor_list_index_var, neighbor_list_var].as_mut_ptr(), 5, cstr!(""));
                                    let neighbor_list_index = LLVMBuildLoad(builder, neighbor_list_index_var, cstr!("neighbor_list_index"));
                                    let neighbor_list = LLVMBuildLoad(builder, neighbor_list_var, cstr!("neighbor_list"));
                                    LLVMBuildCall(builder, interaction_vals.interaction_func,
                                        [neighbor_list_index, neighbor_list].as_mut_ptr(), 2, cstr!(""));
                                    // We need to sync after the interaction function. Can you guess why?
                                    LLVMBuildCall(builder, interaction_sync_handler, [callback_target_param,
                                        llbarrier_data].as_mut_ptr(), 2, cstr!(""));
                                }
                                else {
                                    let name = thread_context.executor_context.global_context.runtime.interaction_index.get(*interaction_id)
                                        .unwrap().get_name();
                                    println!("Debug: Ignoring interaction barrier for disabled interaction {}", name);
                                }
                            }

                        }
                        
                    }
                }
            }

            // Finish worker main
            LLVMPositionBuilderAtEnd(builder, main_entry);
            let callback_target_param = LLVMBuildLoad(builder, callback_target_ptrptr, cstr!("tmp"));
            LLVMBuildCall(builder, end_of_step_handler, [callback_target_param].as_mut_ptr(), 1, cstr!(""));
            LLVMBuildRetVoid(builder);

            // TODO: Better logging and verification
            LLVMVerifyModule(module, LLVMVerifierFailureAction::LLVMPrintMessageAction, ptr::null_mut());

            // Optimization
            let pm_builder = LLVMPassManagerBuilderCreate();
            LLVMPassManagerBuilderUseInlinerWithThreshold(pm_builder, 255);
            LLVMPassManagerBuilderSetOptLevel(pm_builder, 2);
            let module_pass_manager = LLVMCreatePassManager();
            LLVMPassManagerBuilderPopulateModulePassManager(pm_builder, module_pass_manager);

            LLVMRunPassManager(module_pass_manager, module);

            #[cfg(debug_assertions)] {
                if thread_context.particle_range.start == 0 {
                    let module_cstr = LLVMPrintModuleToString(module);
                    let module_str = std::ffi::CStr::from_ptr(module_cstr).to_str()?;
                    println!("{}", module_str);
                    LLVMDisposeMessage(module_cstr);
                }
            }
            LLVMVerifyModule(module, LLVMVerifierFailureAction::LLVMPrintMessageAction, ptr::null_mut());

            // Create a thread-safe module and dispose of the other stuff used in code generation
            let module_ts = LLVMOrcCreateNewThreadSafeModule(module, context_ts);
            LLVMOrcDisposeThreadSafeContext(context_ts);
            //LLVMDisposeBuilder(builder);

            // Drop handle to particle store to remove borrow on thread_context
            std::mem::drop(particle_data);
            // Drop the read handle on neighbor lists
            std::mem::drop(neighbor_lists);
            Ok(Self {
                module_ts,
                callback_target,
                external_symbols
            })
        }
    }
}

/// This struct contains the actual JIT compiler
pub struct CodeExecutor {
    // OrcV2 JIT handle
    jit: LLVMOrcLLJITRef,
    // Callback target (THIS MUST BE KEPT ALIVE AT ALL COST)
    #[allow(dead_code)]
    callback_target: Box<CallbackTarget>,
    // Symbols exported to JIT from process space (THIS MUST BE KEPT ALIVE AT ALL COST)
    #[allow(dead_code)]
    allowed_syms: Box<[LLVMOrcSymbolStringPoolEntryRef]>,
    // Dummy vector of functions to sway the linker to not strip them from the binary
    #[allow(dead_code)]
    dummy_func_vec: Vec<*const extern "C" fn()>
}

impl CodeExecutor {
    pub(crate) fn new(codegen: CodeGenerator) -> Result<Self> {
        unsafe {
            // Make sure, the native target is initialized
            LLVMInitializeCore(LLVMGetGlobalPassRegistry());
            LLVM_InitializeNativeTarget();
            LLVM_InitializeNativeAsmPrinter();
            // Make sure the linker does not optimize out our callbacks
            let dummy_func_vec = vec![
                _call2rust_handler as _,
                _interaction_handler as _,
                _interaction_sync_handler as _,
                _end_of_step as _,
                print_u64 as _,
                print_f64 as _,
                _random_normal as _
            ];
            //std::mem::forget(funcs);

            // Create JIT
            let mut jit = MaybeUninit::uninit();
            let error = LLVMOrcCreateLLJIT(jit.as_mut_ptr(), ptr::null_mut());
            if !error.is_null() {
                return llvm_errorref_to_result("Failed to create LLJIT", error);
            };
            let jit = jit.assume_init();

            // Export a selected number of process functions to the JIT
            let mut allowed_syms = codegen.external_symbols.iter()
                .map(|symbol_name| LLVMOrcLLJITMangleAndIntern(jit, cstring!(symbol_name.clone())))
                .chain(std::iter::once(ptr::null_mut()))
                .collect::<Box<[_]>>();
        
            let mut process_symbols_generator = MaybeUninit::uninit();
            LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                process_symbols_generator.as_mut_ptr(), LLVMOrcLLJITGetGlobalPrefix(jit),
                Some(allowed_symbol_filter), allowed_syms.as_mut_ptr() as *mut c_void
            );
            let process_symbols_generator = process_symbols_generator.assume_init();
            LLVMOrcJITDylibAddGenerator(LLVMOrcLLJITGetMainJITDylib(jit), process_symbols_generator);

            // Add IR module to JIT
            let main_jitdylib = LLVMOrcLLJITGetMainJITDylib(jit);
            let error = LLVMOrcLLJITAddLLVMIRModule(jit, main_jitdylib, codegen.module_ts);
            if !error.is_null() {
                return llvm_errorref_to_result("Failed to add IR module", error);
            };

            // Done
            Ok(Self {
                jit,
                allowed_syms,
                dummy_func_vec,
                callback_target: codegen.callback_target
            })
        }
    }

    pub(crate) fn run(&mut self) {
        unsafe {
            let mut funcaddr = MaybeUninit::uninit();
            let error = LLVMOrcLLJITLookup(self.jit, funcaddr.as_mut_ptr(), cstring!(WORKER_MAIN_NAME));
            if !error.is_null() {
                panic!("Lookup of worker main function failed!");
            };
            let func: WorkerMainFunc = std::mem::transmute_copy(&funcaddr);
            func();
        }
    }
}