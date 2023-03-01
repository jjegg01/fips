use super::super::llwrap as ll; // TODO: Proper workspace
use crate::parsing::{BinOp, BinOpKind};
use super::{Codegen, CodegenContext};

impl Codegen for BinOp {
    fn codegen(&self, cctx: &mut CodegenContext) -> ll::Value {
        let lhs = self.left.codegen(cctx);
        let rhs = self.right.codegen(cctx);
        let (lhs, rhs) = promote_operands(&cctx.ir_builder, lhs, rhs);
        match self.kind {
            BinOpKind::Add => insert_add(&cctx.ir_builder, &lhs, &rhs),
            BinOpKind::Sub => insert_sub(&cctx.ir_builder, &lhs, &rhs),
            BinOpKind::Mul => insert_mul(&cctx.ir_builder, &lhs, &rhs),
            BinOpKind::Div => insert_div(&cctx.ir_builder, &lhs, &rhs, true), // TODO: Implement unsigned division
            BinOpKind::Mod => insert_mod(&cctx.ir_builder, &lhs, &rhs, true), // TODO: Implement unsigned modulo
            BinOpKind::Pow => panic!("Power operator not yet implemented!"),
            BinOpKind::Assign => panic!("Assignment not yet implemented!")
        }
    }
}

fn promote_operands(builder: &ll::IRBuilder, lhs: ll::Value, rhs: ll::Value) -> (ll::Value, ll::Value) {
    let lhs_type_kind = lhs.get_type().get_kind();
    let rhs_type_kind = rhs.get_type().get_kind();
    if lhs_type_kind == rhs_type_kind { (lhs, rhs) }
    else if lhs_type_kind == ll::TypeKind::LLVMDoubleTypeKind || rhs_type_kind == ll::TypeKind::LLVMDoubleTypeKind {
        (convert_to_double(&builder, lhs),
        convert_to_double(&builder, rhs))
    }
    else {
        panic!("Cannot promote operands!")
    }
}

fn convert_to_double(builder: &ll::IRBuilder, val: ll::Value) -> ll::Value {
    let kind = val.get_type().get_kind();
    if kind == ll::TypeKind::LLVMDoubleTypeKind {
        val
    }
    else if kind == ll::TypeKind::LLVMIntegerTypeKind {
        builder.build_si_to_fp(&val, &ll::Type::get_double(builder.get_context()), "cvt_op")
    }
    else {
        panic!("Cannot operand convert to double") // TODO: Better error message
    }
}

/// Insert appropriate add instruction for operands
/// LLVM type of both operands must be the same
fn insert_add(builder: &ll::IRBuilder, lhs: &ll::Value, rhs: &ll::Value) -> ll::Value {
    match lhs.get_type().get_kind() {
        ll::TypeKind::LLVMIntegerTypeKind => builder.build_int_add(&lhs, &rhs, "sum"),
        ll::TypeKind::LLVMDoubleTypeKind  => builder.build_fp_add(&lhs, &rhs, "sum"),
        _ => panic!("Encountered invalid type! ") // TODO: Better error message
    }
}

fn insert_sub(builder: &ll::IRBuilder, lhs: &ll::Value, rhs: &ll::Value) -> ll::Value {
    match lhs.get_type().get_kind() {
        ll::TypeKind::LLVMIntegerTypeKind => builder.build_int_sub(&lhs, &rhs, "diff"),
        ll::TypeKind::LLVMDoubleTypeKind  => builder.build_fp_sub(&lhs, &rhs, "diff"),
        _ => panic!("Encountered invalid type! ") // TODO: Better error message
    }
}

fn insert_mul(builder: &ll::IRBuilder, lhs: &ll::Value, rhs: &ll::Value) -> ll::Value {
    match lhs.get_type().get_kind() {
        ll::TypeKind::LLVMIntegerTypeKind => builder.build_int_mul(&lhs, &rhs, "prod"),
        ll::TypeKind::LLVMDoubleTypeKind  => builder.build_fp_mul(&lhs, &rhs, "prod"),
        _ => panic!("Encountered invalid type! ") // TODO: Better error message
    }
}

fn insert_div(builder: &ll::IRBuilder, lhs: &ll::Value, rhs: &ll::Value, signed: bool) -> ll::Value {
    match lhs.get_type().get_kind() {
        ll::TypeKind::LLVMIntegerTypeKind => if signed { builder.build_int_sdiv(&lhs, &rhs, "quot") }
                                             else      { builder.build_int_udiv(&lhs, &rhs, "quot") }
        ll::TypeKind::LLVMDoubleTypeKind  => builder.build_fp_div(&lhs, &rhs, "quot"),
        _ => panic!("Encountered invalid type! ") // TODO: Better error message
    }
}

fn insert_mod(builder: &ll::IRBuilder, lhs: &ll::Value, rhs: &ll::Value, signed: bool) -> ll::Value {
    match lhs.get_type().get_kind() {
        ll::TypeKind::LLVMIntegerTypeKind => if signed { builder.build_int_srem(&lhs, &rhs, "rem") }
                                             else      { builder.build_int_urem(&lhs, &rhs, "rem") }
        ll::TypeKind::LLVMDoubleTypeKind  => builder.build_fp_rem(&lhs, &rhs, "rem"),
        _ => panic!("Encountered invalid type! ") // TODO: Better error message
    }
}