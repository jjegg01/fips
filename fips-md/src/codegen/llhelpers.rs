//! Macros and related things for making llvm-sys a little less painful


use std::{collections::HashMap, ffi::{CStr, CString}, sync::RwLockReadGuard, unimplemented};

use anyhow::{anyhow, Result};

use libc::{c_uint, c_ulonglong};
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::error::*;
use llvm_sys::{LLVMIntPredicate, LLVMRealPredicate, LLVMTypeKind};

use crate::{codegen::promote_to_vector, parser::{FipsType, CompileTimeConstant, BinaryOperator}, runtime::{MemberData, ParticleIndex, ParticleStore, Domain}};
use crate::utils::FipsValue;

use super::{NeighborList, evaluate_binop};

// Convert LLVMErrorRef to Result
pub(crate) unsafe fn llvm_errorref_to_result<T>(context: &str, error: LLVMErrorRef) -> Result<T> {
    let message = CStr::from_ptr(LLVMGetErrorMessage(error)).to_str()
            .expect("Error while decoding LLVM error (how ironic)");
    Err(anyhow!("{}: {}", context, message))
}

// Add \0 to string reference for C intercompatibility
#[macro_export]
macro_rules! cstr {
    ($string:expr) => {
        concat!($string, "\0").as_ptr() as *const _
    }
}

#[macro_export]
macro_rules! cstring {
    ($string:expr) => {
        CString::new($string).expect("String conversion failed")
            .as_bytes().as_ptr() as *const _
    }
}

/// Get (type, dimensions) tuple for LLVM type corresponding to FIPS type
pub(crate) unsafe fn get_llvm_type_dims(context: LLVMContextRef, typ: &FipsType) -> Result<(LLVMTypeRef, Vec<usize>)> {
    Ok(match typ {
        FipsType::Double => (LLVMDoubleTypeInContext(context), vec![]),
        FipsType::Int64 => (LLVMInt64TypeInContext(context), vec![]),
        FipsType::Array { typ, length } => {
            let length = match length {
                CompileTimeConstant::Literal(value) | CompileTimeConstant::Substituted(value, _) => value,
                CompileTimeConstant::Identifier(name) => return Err(anyhow!("Unresolved identifer {}", name))
            };
            let (subtype, mut subdims) = get_llvm_type_dims(context, typ)?;
            subdims.insert(0, *length);
            (subtype, subdims)
        } 
    })
}

pub(crate) unsafe fn create_global_const_double(module: LLVMModuleRef, name: String, value: f64) -> LLVMValueRef {
    let context = LLVMGetModuleContext(module);
    let typ = LLVMDoubleTypeInContext(context);
    let llval = LLVMAddGlobal(module, typ, cstring!(name));
    let initializer = LLVMConstReal(typ, value);
    LLVMSetGlobalConstant(llval, 1);
    LLVMSetInitializer(llval, initializer);
    llval
}

pub(crate) unsafe fn create_global_const_int64(module: LLVMModuleRef, name: String, value: i64) -> LLVMValueRef {
    let context = LLVMGetModuleContext(module);
    let typ = LLVMInt64TypeInContext(context);
    let llval = LLVMAddGlobal(module, typ, cstring!(name));
    let initializer = LLVMConstInt(typ, value as c_ulonglong, 0); // TODO: Sign extend?
    LLVMSetGlobalConstant(llval, 1);
    LLVMSetInitializer(llval, initializer);
    llval
}

pub(crate) unsafe fn create_global_const(module: LLVMModuleRef, name: String, value: FipsValue) -> LLVMValueRef {
    match value {
        FipsValue::Int64(value) => create_global_const_int64(module, name, value),
        FipsValue::Double(value) => create_global_const_double(module, name, value)
    }
}

unsafe fn __create_global_ptr(context: LLVMContextRef, module: LLVMModuleRef, name: String, scalar_type: LLVMTypeRef,
    ptr: usize, stride: u32) -> LLVMValueRef 
{
    let element_type = match stride {
        1 => scalar_type,
        n => LLVMArrayType(scalar_type, n as c_uint)
    };
    let typ = LLVMPointerType(element_type, 0);
    let llval = LLVMAddGlobal(module, typ, cstring!(name));
    // Here we just assume the native pointer size if 64 bit
    let initializer = LLVMConstIntToPtr(LLVMConstInt(LLVMInt64TypeInContext(context), ptr as c_ulonglong, 0), typ);
    LLVMSetGlobalConstant(llval, 1);
    LLVMSetInitializer(llval, initializer);
    llval
}

pub(crate) unsafe fn create_global_ptr(module: LLVMModuleRef, name: String, typ: &FipsType, ptr: usize) -> Result<LLVMValueRef> {
    let context = LLVMGetModuleContext(module);
    let (lltype, dims) = get_llvm_type_dims(context, typ)?;
    Ok(match dims.len() {
        0 => __create_global_ptr(context, module, name, lltype, ptr, 1),
        1 => __create_global_ptr(context, module, name, lltype, ptr, dims[0] as u32),
        _ => unimplemented!("No multidim support for now"),
    })
}

pub(crate) unsafe fn create_local_ptr(module: LLVMModuleRef, builder: LLVMBuilderRef, name: String, typ: &FipsType) -> Result<LLVMValueRef> {
    let context = LLVMGetModuleContext(module);
    let (lltyp, dims) = get_llvm_type_dims(context, typ)?;
    let typ = match dims.len() {
        0 => lltyp,
        1 => LLVMArrayType(lltyp, dims[0] as c_uint),
        _ => unimplemented!("No multidim support for now"),
    };
    Ok(LLVMBuildAlloca(builder, typ, cstring!(name)))
}

// TODO: Integrate these with the above
pub(crate) unsafe fn fips_value_2_llvm(module: LLVMModuleRef, value: &FipsValue) -> LLVMValueRef {
    match value {
        FipsValue::Int64(value) => {
            let context = LLVMGetModuleContext(module);
            let typ = LLVMInt64TypeInContext(context);
            LLVMConstInt(typ, *value as c_ulonglong, 0)
        },
        FipsValue::Double(value) => {
            let context = LLVMGetModuleContext(module);
            let typ = LLVMDoubleTypeInContext(context);
            LLVMConstReal(typ, *value)
        }
    }
}

unsafe fn __fips_ptr_2_llvm(context: LLVMContextRef, scalar_type: LLVMTypeRef,
    ptr: usize, stride: u32) -> LLVMValueRef 
{
    let element_type = match stride {
        1 => scalar_type,
        n => LLVMArrayType(scalar_type, n as c_uint)
    };
    let typ = LLVMPointerType(element_type, 0);
    // Here we just assume the native pointer size if 64 bit
    LLVMConstIntToPtr(LLVMConstInt(LLVMInt64TypeInContext(context), ptr as c_ulonglong, 0), typ)
}

pub(crate) unsafe fn fips_ptr_2_llvm(module: LLVMModuleRef, typ: &FipsType, ptr: usize) -> Result<LLVMValueRef> {
    let context = LLVMGetModuleContext(module);
    let (lltype, dims) = get_llvm_type_dims(context, typ)?;
    Ok(match dims.len() {
        0 => __fips_ptr_2_llvm(context, lltype, ptr, 1),
        1 => __fips_ptr_2_llvm(context, lltype, ptr, dims[0] as u32),
        _ => unimplemented!("No multidim support for now"),
    })
}
pub(crate) unsafe fn create_neighbor_member_values<'a>(module: LLVMModuleRef, members: Vec<&'a str>, 
    neighbor_list: &RwLockReadGuard<NeighborList>, particle_index: &ParticleIndex, particle_store: &ParticleStore) 
-> HashMap<&'a str, Vec<LLVMValueRef>> {
    members.iter().map(|member_name| {
        let llvals = neighbor_list.pos_blocks.iter()
            .map(move |(particle_id, index_range)| {
                let particle_def = particle_index.get(*particle_id).unwrap();
                let particle_data = particle_store.get_particle(*particle_id).unwrap();
                let (member_id, member_def) = match particle_def.get_member_by_name(member_name) {
                    None => { return None }
                    Some(x) => x
                };
                let member_data = particle_data.borrow_member(&member_id).unwrap();
                Some(match &*member_data {
                    MemberData::Uniform(value) => {
                        fips_value_2_llvm(module, value)
                    }
                    MemberData::PerParticle{data, ..} => {
                        // TODO: Less unwrap
                        let offset = index_range.start * member_def.get_member_size().unwrap();
                        let data_ptr = data.as_ptr() as usize + offset;
                        fips_ptr_2_llvm(module, member_def.get_type(), data_ptr).unwrap()
                    }
                })
            }).collect::<Vec<_>>();
        // TODO: For now, do no allow mixed allocation for neighbor members
        let mut lltyp_check = None;
        for llval in &llvals {
            if let Some(llval) = llval {
                lltyp_check = Some(LLVMTypeOf(*llval));
                break;
            }
        }
        let lltyp_check = lltyp_check.unwrap();
        // TODO: Clean this up; fun fact: on debug builds the unwrap_or value is constructed no matter what
        // so this crashes if we just try to generate NULL pointers for non-pointer types
        let llvals = llvals.into_iter().map(|llval| llval.unwrap())//llval.unwrap_or(LLVMConstPointerNull(lltyp_check)))
            .collect::<Vec<_>>();
        for llval in &llvals {
            assert_eq!(lltyp_check, LLVMTypeOf(*llval));
        }
        (*member_name, llvals)
    }).collect::<HashMap<_,_>>()
}

pub(crate) unsafe fn build_loop(context: LLVMContextRef, builder: LLVMBuilderRef, block_loop_body: LLVMBasicBlockRef,
    block_after_loop: LLVMBasicBlockRef, loop_index_ptr: LLVMValueRef, end_index: LLVMValueRef) -> LLVMBasicBlockRef {
    // Create loop blocks
    let block_loop_check = LLVMInsertBasicBlockInContext(context, block_after_loop, cstr!("loop_check"));
    // let block_loop_body = LLVMInsertBasicBlockInContext(context, block_after_loop, cstr!("loop_body"));
    let block_loop_increment = LLVMInsertBasicBlockInContext(context, block_after_loop, cstr!("loop_increment"));

    // Create loop check
    LLVMBuildBr(builder, block_loop_check);
    LLVMPositionBuilderAtEnd(builder, block_loop_check);
    let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));
    let end_index = match LLVMGetTypeKind(LLVMTypeOf(end_index)) {
        LLVMTypeKind::LLVMPointerTypeKind => { LLVMBuildLoad(builder, end_index, cstr!("end_index")) }
        _ => end_index
    };
    let comparison = LLVMBuildICmp(builder, LLVMIntPredicate::LLVMIntULT, loop_index, end_index, cstr!("loop_check"));
    LLVMBuildCondBr(builder, comparison, block_loop_body, block_after_loop);

    // Create loop increment
    LLVMPositionBuilderAtEnd(builder, block_loop_increment);
    let loop_index = LLVMBuildLoad(builder, loop_index_ptr, cstr!("loop_var_val"));
    let llone = LLVMConstInt(LLVMInt64TypeInContext(context), 1, 0);
    let incremented_index = LLVMBuildAdd(builder, loop_index, llone, cstr!("incremented_val"));
    LLVMBuildStore(builder, incremented_index, loop_index_ptr);
    LLVMBuildBr(builder, block_loop_check);

    block_loop_increment
}

pub(crate) unsafe fn llmultiply_by_minus_one(context: LLVMContextRef, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
    match LLVMGetTypeKind(LLVMTypeOf(value)) {
        LLVMTypeKind::LLVMArrayTypeKind => {
            let length = LLVMGetArrayLength(LLVMTypeOf(value));
            llmultiply_by_minus_one(context, builder, promote_to_vector(context, builder, value, length))
        }
        LLVMTypeKind::LLVMVectorTypeKind => {
            let scalar = scalar_minus_one(context, LLVMGetElementType(LLVMTypeOf(value)));
            let length = LLVMGetVectorSize(LLVMTypeOf(value));
            let llminus = LLVMConstVector(vec![scalar; length as usize].as_mut_ptr(), length);
            match LLVMGetTypeKind(LLVMGetElementType(LLVMTypeOf(value))) {
                LLVMTypeKind::LLVMDoubleTypeKind => {
                    LLVMBuildFMul(builder, llminus, value, cstr!("negated"))
                }
                LLVMTypeKind::LLVMIntegerTypeKind => {
                    LLVMBuildMul(builder, llminus, value, cstr!("negated"))
                }
                _ => panic!()
            }
        }
        LLVMTypeKind::LLVMDoubleTypeKind => {
            let llminus = scalar_minus_one(context, LLVMTypeOf(value));
            LLVMBuildFMul(builder, llminus, value, cstr!("negated"))
        }
        LLVMTypeKind::LLVMIntegerTypeKind => {
            let llminus = scalar_minus_one(context, LLVMTypeOf(value));
            LLVMBuildMul(builder, llminus, value, cstr!("negated"))
        }
        _ => panic!()
    }
}

unsafe fn scalar_minus_one(context: LLVMContextRef, typ: LLVMTypeRef) -> LLVMValueRef {
    let double_type = LLVMDoubleTypeInContext(context);
    let int64_type = LLVMInt64TypeInContext(context);
    match LLVMGetTypeKind(typ) {
        LLVMTypeKind::LLVMDoubleTypeKind => {
            LLVMConstReal(double_type, -1.0)
        }
        LLVMTypeKind::LLVMIntegerTypeKind => { 
            LLVMConstInt(int64_type, std::mem::transmute(-1i64), 1)
        }
        _ => panic!()
    }
}

pub(crate) unsafe fn calculate_distance_sqr_and_vec(context: LLVMContextRef, builder: LLVMBuilderRef,
    pos_1: LLVMValueRef, pos_2: LLVMValueRef) 
-> (LLVMValueRef, LLVMValueRef) {
    // Cheat a bit and create a pseudo parser expression for the difference
    let dist_vec = evaluate_binop(context, builder, pos_2, pos_1, BinaryOperator::Sub).unwrap();
    // Now square the elements of the dist vector
    let dist_vec_sqr = evaluate_binop(context, builder, dist_vec, dist_vec, BinaryOperator::Mul).unwrap();
    // Finally add all elements of the distance vector squared
    let mut dist = LLVMBuildExtractElement(builder, dist_vec_sqr, 
        LLVMConstInt(LLVMInt64TypeInContext(context), 0, 0), cstr!("dist_acc"));
    for i in 1..LLVMGetVectorSize(LLVMTypeOf(dist_vec_sqr)) {
        dist = LLVMBuildFAdd(builder, dist,
            LLVMBuildExtractElement(builder, dist_vec_sqr, 
                LLVMConstInt(LLVMInt64TypeInContext(context), i as u64, 0), cstr!("dist_elem")),
            cstr!("dist_acc")
        );
    }
    (dist, dist_vec)
}

pub(crate) unsafe fn correct_postion_vector(context: LLVMContextRef, builder: LLVMBuilderRef,
    position: LLVMValueRef, other_position: LLVMValueRef, cutoff_skin: f64, domain: &Domain)
-> LLVMValueRef {
    let double_type = LLVMDoubleTypeInContext(context);
    match domain {
        Domain::Dim2 { .. } => unimplemented!(),
        Domain::Dim3 { x, y, z } => {
            // Assume domain size > 2*cutoff
            assert!(x.size() > 2.*cutoff_skin);
            assert!(y.size() > 2.*cutoff_skin);
            assert!(z.size() > 2.*cutoff_skin);
            // If this condition holds, the raw distance vector must have a component larger than the cutoff length
            // or smaller than then negative cutoff length if it is incorrectly mirrored
            let raw_dist_vec = evaluate_binop(context, builder, position, other_position, BinaryOperator::Sub).unwrap();
            let cmp1 = LLVMBuildFCmp(builder, LLVMRealPredicate::LLVMRealOGT, raw_dist_vec, LLVMConstVector([
                LLVMConstReal(double_type, cutoff_skin),
                LLVMConstReal(double_type, cutoff_skin),
                LLVMConstReal(double_type, cutoff_skin),
            ].as_mut_ptr(), 3), cstr!("cmp_gt"));
            let cmp2 = LLVMBuildFCmp(builder, LLVMRealPredicate::LLVMRealOLT, raw_dist_vec, LLVMConstVector([
                LLVMConstReal(double_type, -cutoff_skin),
                LLVMConstReal(double_type, -cutoff_skin),
                LLVMConstReal(double_type, -cutoff_skin),
            ].as_mut_ptr(), 3), cstr!("cmp_lt"));

            // let valx = LLVMBuildExtractElement(builder, cmp1, LLVMConstInt(int8_type, 0, 0), cstr!(""));
            // let valy = LLVMBuildExtractElement(builder, cmp1, LLVMConstInt(int8_type, 1, 0), cstr!(""));
            // let valz = LLVMBuildExtractElement(builder, cmp1, LLVMConstInt(int8_type, 2, 0), cstr!(""));
            // LLVMBuildCall(builder, print_func_u64, [valx].as_mut_ptr(), 1, cstr!(""));
            // LLVMBuildCall(builder, print_func_u64, [valy].as_mut_ptr(), 1, cstr!(""));
            // LLVMBuildCall(builder, print_func_u64, [valz].as_mut_ptr(), 1, cstr!(""));

            // Cast to double
            let cmp1 = LLVMBuildUIToFP(builder, cmp1, LLVMVectorType(double_type, 3), cstr!("cmp_gt_dbl"));
            let cmp2 = LLVMBuildUIToFP(builder, cmp2, LLVMVectorType(double_type, 3), cstr!(""));
            let cmp2 = LLVMBuildFMul(builder, cmp2, LLVMConstVector([
                LLVMConstReal(double_type, -1.),
                LLVMConstReal(double_type, -1.),
                LLVMConstReal(double_type, -1.),
            ].as_mut_ptr(), 3), cstr!("cmp_lt_dbl"));
            // Add
            let cmp = LLVMBuildFAdd(builder, cmp1, cmp2, cstr!("cmp"));
            // Multiply domain length
            let corr_vec = LLVMBuildFMul(builder, cmp, LLVMConstVector([
                LLVMConstReal(double_type, x.size()),
                LLVMConstReal(double_type, y.size()),
                LLVMConstReal(double_type, z.size()),
            ].as_mut_ptr(), 3), cstr!("correction_vec"));
            evaluate_binop(context, builder, other_position, corr_vec, BinaryOperator::Add).unwrap()
        }
    }
}