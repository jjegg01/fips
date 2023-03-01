// mod codegen;
// mod llwrap;

//use crate::parsing::ast::ASTNode;
// use self::codegen::{Codegen, CodegenContext};

// pub fn test(ast: ASTNode) {
//     let context = llwrap::Context::new();
//     let mut module = llwrap::Module::with_context(&context, "mymodule");
//     let builder = llwrap::IRBuilder::with_context(&context);

//     let ftype = llwrap::FunctionType::new(&llwrap::Type::get_double(&context), &mut []);
//     let fun = module.add_function("myfunction", &ftype);
//     let bb = llwrap::BasicBlock::append_function(&context, &fun, "myfunctionbody");
//     builder.position_at_end(&bb);
//     let mut cctx = CodegenContext::new(builder);
//     let retval = ast.codegen(&mut cctx);
//     cctx.ir_builder.build_ret(&retval);
//     module.dump();
// }