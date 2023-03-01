//! Evaluation of expressions with LLVM

use std::{collections::HashMap, ffi::CString};

use anyhow::{anyhow, Result};

use crate::{codegen::util::unwrap_usize_constant, cstr, cstring, parser::{self, BinaryOperator}, runtime::FunctionIndex};
use super::{LLSymbolValue, analysis::{SymbolTable}};

use llvm_sys::{LLVMTypeKind, prelude::*};
use llvm_sys::core::*;

// Perform add, subtract, muliply or division on signed integer operands
unsafe fn int_asmd(builder: LLVMBuilderRef, op: parser::BinaryOperator, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
    match op {
        parser::BinaryOperator::Add => {
            LLVMBuildAdd(builder, lhs, rhs, cstr!("tmp_add"))
        }
        parser::BinaryOperator::Sub => {
            LLVMBuildSub(builder, lhs, rhs, cstr!("tmp_sub"))
        }
        parser::BinaryOperator::Mul => {
            LLVMBuildMul(builder, lhs, rhs, cstr!("tmp_mul"))
        }
        parser::BinaryOperator::Div => {
            LLVMBuildSDiv(builder, lhs, rhs, cstr!("tmp_div"))
        }
    }
}

// Perform add, subtract, muliply or division on floating point operands
unsafe fn float_asmd(builder: LLVMBuilderRef, op: parser::BinaryOperator, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
    match op {
        parser::BinaryOperator::Add => {
            LLVMBuildFAdd(builder, lhs, rhs, cstr!("tmp_add"))
        }
        parser::BinaryOperator::Sub => {
            LLVMBuildFSub(builder, lhs, rhs, cstr!("tmp_sub"))
        }
        parser::BinaryOperator::Mul => {
            LLVMBuildFMul(builder, lhs, rhs, cstr!("tmp_mul"))
        }
        parser::BinaryOperator::Div => {
            LLVMBuildFDiv(builder, lhs, rhs, cstr!("tmp_div"))
        }
    }
}

unsafe fn vector_asmd(builder: LLVMBuilderRef, op: parser::BinaryOperator, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
    let lhs_type = LLVMTypeOf(lhs);
    let rhs_type = LLVMTypeOf(rhs);
    let lhs_elem_type = LLVMGetElementType(lhs_type);
    let rhs_elem_type = LLVMGetElementType(rhs_type);
    match (LLVMGetTypeKind(lhs_elem_type), LLVMGetTypeKind(rhs_elem_type)) {
        (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMDoubleTypeKind) => {
            float_asmd(builder, op, lhs, rhs)
        }
        (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMIntegerTypeKind) => {
            int_asmd(builder, op, lhs, rhs)
        }
        (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMIntegerTypeKind) | 
        (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMDoubleTypeKind) => {
            unimplemented!("Cannot combine double and int vectors yet")
        }
        _ => panic!()
    }
}

pub(crate) unsafe fn promote_to_vector(context: LLVMContextRef, builder: LLVMBuilderRef, llval: LLVMValueRef, length: u32) -> LLVMValueRef {
    let llkind = LLVMGetTypeKind(LLVMTypeOf(llval));
    match llkind {
        // Already a vector? Nothing to do then (TODO: Multidim support)
        LLVMTypeKind::LLVMVectorTypeKind => {
            llval
        }
        // An array?
        LLVMTypeKind::LLVMArrayTypeKind => {
            assert_eq!(LLVMGetArrayLength(LLVMTypeOf(llval)), length);
            let elem_kind = LLVMGetTypeKind(LLVMGetElementType(LLVMTypeOf(llval)));
            let mut llvec = match elem_kind {
                LLVMTypeKind::LLVMDoubleTypeKind => {
                    LLVMGetUndef(
                        LLVMVectorType(LLVMDoubleTypeInContext(context), length)
                    )
                }
                LLVMTypeKind::LLVMIntegerTypeKind => {
                    LLVMGetUndef(
                        LLVMVectorType(LLVMInt64TypeInContext(context), length)
                    )
                }
                _ => unreachable!()
            };
            for i in 0..length {
                let llelement = LLVMBuildExtractValue(builder, llval, i, cstr!(""));
                llvec = LLVMBuildInsertElement(builder, llvec, llelement, 
                    LLVMConstInt(LLVMInt32TypeInContext(context), i as u64, 0),
                    cstring!("tmp_promoted_vec"));
            }
            llvec
        }
        // Not a vector? Then promote to one
        LLVMTypeKind::LLVMDoubleTypeKind | LLVMTypeKind::LLVMIntegerTypeKind => {
            // Start with undef
            let mut llvec = match llkind {
                LLVMTypeKind::LLVMDoubleTypeKind => {
                    LLVMGetUndef(
                        LLVMVectorType(LLVMDoubleTypeInContext(context), length)
                    )
                }
                LLVMTypeKind::LLVMIntegerTypeKind => {
                    LLVMGetUndef(
                        LLVMVectorType(LLVMInt64TypeInContext(context), length)
                    )
                }
                _ => unreachable!()
            };
            // Chain insertelement instructions to fill vector
            for i in 0..length {
                llvec = LLVMBuildInsertElement(builder, llvec, llval, 
                    LLVMConstInt(LLVMInt32TypeInContext(context), i as u64, 0),
                    cstring!("tmp_promoted_vec"));
            }
            llvec
        }
        _ => panic!()
    }
}


pub(crate) unsafe fn convert_to_scalar_or_array(context: LLVMContextRef, builder: LLVMBuilderRef, llval: LLVMValueRef) -> LLVMValueRef {
    let llkind = LLVMGetTypeKind(LLVMTypeOf(llval));
    match llkind {
        // Scalar or array? Ok then
        LLVMTypeKind::LLVMDoubleTypeKind | LLVMTypeKind::LLVMIntegerTypeKind | LLVMTypeKind::LLVMArrayTypeKind => {
            llval
        }
        LLVMTypeKind::LLVMVectorTypeKind => {
            let length = LLVMGetVectorSize(LLVMTypeOf(llval));
            let elem_type = LLVMGetElementType(LLVMTypeOf(llval));
            let mut llarray = LLVMGetUndef(LLVMArrayType(elem_type, length));
            for i in 0..length {
                let llelem = LLVMBuildExtractElement(builder, llval,
                    LLVMConstInt(LLVMInt32TypeInContext(context), i as u64, 0), cstr!(""));
                llarray = LLVMBuildInsertValue(builder, llarray, llelem, i, cstr!("tmp_vec2array"))
            }
            llarray
        }
        _ => panic!()
    }
}

pub(crate) unsafe fn evaluate_expression(context: LLVMContextRef, builder: LLVMBuilderRef, expr: &parser::Expression,
    symbol_table: &SymbolTable<LLSymbolValue>,
    namespace_symbols: &HashMap<&String, HashMap<String, LLVMValueRef>>,
    function_index: &FunctionIndex,
    callback_target_ptrptr: LLVMValueRef) -> Result<LLVMValueRef>
{
    match expr {
        parser::Expression::Atom(atom) => {
            match &atom {
                parser::Atom::Variable(name) => {
                    // Try to resolve name
                    let value = symbol_table.resolve_symbol(name)
                        .ok_or(anyhow!("Cannot resolve symbol {}", name))?;
                    let llname = format!("load_{}", name);
                    let llptr = match value.value.as_ref()
                        .expect(&format!("Incomplete symbol table: symbol for {} has no value", name)) 
                    {
                        LLSymbolValue::SimplePointer(llptr) => { *llptr }
                        LLSymbolValue::ParticleMember { local_ptr, .. } => 
                        {
                            local_ptr.expect(&format!("Local pointer not set for symbol {}", name))
                        }
                        LLSymbolValue::Function { .. } => panic!("Cannot evaluate function as atomic expression")
                    };
                    let llval = LLVMBuildLoad(builder, llptr, cstring!(llname));
                    Ok(llval)
                }
                parser::Atom::Literal(literal) => {
                    match literal {
                        parser::Literal::Double(val) => {
                            let typ = LLVMDoubleTypeInContext(context);
                            Ok(LLVMConstReal(typ, *val))
                        }
                        parser::Literal::Int64(val) => {
                            let typ = LLVMInt64TypeInContext(context);
                            Ok(LLVMConstInt(typ, *val as u64, 1))
                        }
                    }
                }
                parser::Atom::NamespaceVariable{ namespace, name } => {
                    let llptr = namespace_symbols.get(namespace)
                        .ok_or(anyhow!("Namespace {} not found", namespace))?
                        .get(name).map(|x| x)
                            .ok_or(anyhow!("Symbol {} not found in namespace {}", name, namespace))?;
                    Ok(LLVMBuildLoad(builder, *llptr, cstring!(format!("{}_{}", namespace, name))))
                }
            }
        }
        parser::Expression::BinaryOperation(binop) => {
            let lhs = evaluate_expression(context, builder, binop.lhs.as_ref(),
                symbol_table, namespace_symbols, function_index, callback_target_ptrptr)?;
            let rhs = evaluate_expression(context, builder, binop.rhs.as_ref(),
                symbol_table, namespace_symbols, function_index, callback_target_ptrptr)?;
            evaluate_binop(context, builder, lhs, rhs, binop.op)
        }
        parser::Expression::FunctionCall(call) => {
            // Resolve the function
            let symbol = symbol_table.resolve_symbol(&call.fn_name)
                .ok_or(anyhow!("Cannot find function {}", call.fn_name))?;
            let symbol_value = match symbol.value.as_ref()
                .ok_or(anyhow!("No value for symbol for function {}", call.fn_name))? 
            {
                LLSymbolValue::Function(val) => val,
                _ => return Err(anyhow!("Invalid symbol value for function {}", call.fn_name))
            };
            let func = function_index.get(symbol_value.function_id)
                .expect(&format!("Internal: cannot find function {} in index", call.fn_name));

            // Evaluate all parameters
            let mut parameter_vals = call.parameters.iter()
                .map(|expr| evaluate_expression(context, builder, expr,
                    symbol_table, namespace_symbols, function_index, callback_target_ptrptr))
                .collect::<Result<Vec<_>>>()?;

            // Check if function needs callback_target_ptr
            if func.needs_callback_target_ptr() {
                let callback_target_ptr = LLVMBuildLoad(builder, callback_target_ptrptr, cstr!("callback_target_ptr"));
                parameter_vals.insert(0, callback_target_ptr);
            }
            
            // Build call
            Ok(func.build_call(context, builder, symbol_value, parameter_vals)?)
        }
        parser::Expression::Block(_) => { unimplemented!() }
        parser::Expression::Indexing(indexop) => {
            // Kind of cheaty: Load the full array
            let array_val = evaluate_expression(context, builder, 
                &parser::Expression::Atom(parser::Atom::Variable(indexop.array.clone())),
                symbol_table, namespace_symbols, function_index, callback_target_ptrptr)?;
            // Extract a single value (TODO: Type checking)
            let index = unwrap_usize_constant(&indexop.index)? as u32;
            let name = format!("{}_indexed_at_{}", indexop.array, index);
            Ok(LLVMBuildExtractValue(builder, array_val, 
                index, cstring!(name)))
        },
        parser::Expression::AdHocArray(adhocarray) => {
            if adhocarray.elements.is_empty() {
                unimplemented!("Empty arrays are not supported yet");
            }
            // Evaluate all element expressions
            let elements = adhocarray.elements.iter()
                .map(|element_expr| evaluate_expression(
                    context, builder, element_expr, symbol_table, namespace_symbols, 
                    function_index, callback_target_ptrptr)
                ).collect::<Result<Vec<_>>>()?;
            let lltyp = LLVMTypeOf(elements[0]);
            let mut result = LLVMGetUndef(LLVMArrayType(lltyp, elements.len() as u32));
            for (i, element) in elements.into_iter().enumerate() {
                let element = convert_to_scalar_or_array(context, builder, element);
                let name = format!("adhoc_array_{}", i);
                result = LLVMBuildInsertValue(builder, result, element, i as u32, cstring!(name));
            }
            Ok(result)
        },
    }
}

pub(crate) unsafe fn evaluate_binop(context: LLVMContextRef, builder: LLVMBuilderRef, lhs: LLVMValueRef, rhs: LLVMValueRef, op: BinaryOperator) -> Result<LLVMValueRef> {
    let lhs_type = LLVMTypeOf(lhs);
    let rhs_type = LLVMTypeOf(rhs);
    let lhs_type_kind = LLVMGetTypeKind(lhs_type);
    let rhs_type_kind = LLVMGetTypeKind(rhs_type);
    // TODO: Do proper type checking beforehand
    Ok(match (lhs_type_kind, rhs_type_kind) {
        // Both operands real scalars
        (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMDoubleTypeKind) => {
            float_asmd(builder, op, lhs, rhs)
        }
        // Both operands integer scalars
        (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMIntegerTypeKind) => {
            int_asmd(builder, op, lhs, rhs)
        }
        // Both operands vectors
        (LLVMTypeKind::LLVMVectorTypeKind, LLVMTypeKind::LLVMVectorTypeKind) => {
            if LLVMGetVectorSize(lhs_type) != LLVMGetVectorSize(rhs_type) {
                return Err(anyhow!("Cannot combine vectors of different length in a binary operation"));
            }
            vector_asmd(builder, op, lhs, rhs)
        }
        // One operand is scalar, the other is aa vector
        (LLVMTypeKind::LLVMVectorTypeKind, _) | (_, LLVMTypeKind::LLVMVectorTypeKind) => {
            // Get length of vector operand
            let vec_length = match (lhs_type_kind, rhs_type_kind) {
                (LLVMTypeKind::LLVMVectorTypeKind, _) => {
                    // For safety: Assert rhs kind is sane
                    assert!(matches!(rhs_type_kind, LLVMTypeKind::LLVMDoubleTypeKind 
                        | LLVMTypeKind::LLVMIntegerTypeKind | LLVMTypeKind::LLVMArrayTypeKind));
                    LLVMGetVectorSize(lhs_type)
                }
                (_, LLVMTypeKind::LLVMVectorTypeKind) => {
                    // Same but reversed
                    assert!(matches!(lhs_type_kind, LLVMTypeKind::LLVMDoubleTypeKind 
                        | LLVMTypeKind::LLVMIntegerTypeKind | LLVMTypeKind::LLVMArrayTypeKind));
                    LLVMGetVectorSize(rhs_type)
                }
                _ => unreachable!()
            };
            // Promote operands (this does nothing for the vector operand)
            let lhs = promote_to_vector(context, builder, lhs, vec_length);
            let rhs = promote_to_vector(context, builder, rhs, vec_length);
            vector_asmd(builder, op, lhs, rhs)
        }
        // Array operands get promoted to vectors
        (LLVMTypeKind::LLVMArrayTypeKind, LLVMTypeKind::LLVMArrayTypeKind) => {
            let lhs = promote_to_vector(context, builder, lhs, LLVMGetArrayLength(lhs_type));
            let rhs = promote_to_vector(context, builder, rhs, LLVMGetArrayLength(rhs_type));
            evaluate_binop(context, builder, lhs, rhs, op)?
        }
        (LLVMTypeKind::LLVMArrayTypeKind, _) => {
            let lhs = promote_to_vector(context, builder, lhs, LLVMGetArrayLength(lhs_type));
            evaluate_binop(context, builder, lhs, rhs, op)?
        }
        (_ , LLVMTypeKind::LLVMArrayTypeKind) => {
            let rhs = promote_to_vector(context, builder, rhs, LLVMGetArrayLength(rhs_type));
            evaluate_binop(context, builder, lhs, rhs, op)?
        }
        // Unimplemented
        (LLVMTypeKind::LLVMDoubleTypeKind, LLVMTypeKind::LLVMIntegerTypeKind) | 
        (LLVMTypeKind::LLVMIntegerTypeKind, LLVMTypeKind::LLVMDoubleTypeKind) => {
            unimplemented!("Cannot combine double and int scalars yet")
        }
        _ => panic!()
    })
}