//! Parser for the FIPS language

use super::*;
use crate::runtime::BUILTIN_CONST_NDIM;
use std::num::IntErrorKind;

// Helper function
fn make_binop(lhs: Expression, rhs: Expression, op: BinaryOperator) -> Expression {
    Expression::BinaryOperation(BinaryOperation {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op
    })
}

peg::parser!{
    pub grammar fips_parser() for str {

        /// Parse a unit (i.e. a source file)
        pub rule unit() -> Unit
            = _ members:unit_member() ** _ _ {
                let mut unit = Unit::new();
                for member in members {
                    match member {
                        UnitMember::GlobalStateMember(global_state_member) => unit.global_state_members.push(global_state_member),
                        UnitMember::Particle(particle) => unit.particles.push(particle),
                        UnitMember::Interaction(interaction) => unit.interactions.push(interaction),
                        UnitMember::Simulation(simulation) => unit.simulations.push(simulation),
                        UnitMember::ExternFunctionDecl(extern_functiondecl) => unit.extern_functiondecls.push(extern_functiondecl),
                    }
                }
                unit
            }

        pub(crate) rule unit_member() -> UnitMember
            = global_state_member:global_state_member() { UnitMember::GlobalStateMember(global_state_member) } 
            / particle:particle() { UnitMember::Particle(particle) }
            / interaction:interaction() { UnitMember::Interaction(interaction) }
            / simulation:simulation() { UnitMember::Simulation(simulation) }
            / extern_funcdecl:extern_functiondecl() { UnitMember::ExternFunctionDecl(extern_funcdecl) }

        /// Global state member
        rule global_state_member() -> GlobalStateMember
            = "global" __ name:identifier_not_ignored() _ ":" _ mutable:("mut" __)? _ typ:typ() ";"
            {
                let mutable = mutable.is_some();
                GlobalStateMember { name, mutable, typ}
            }

        /// Parse an `extern` function declaration
        pub rule extern_functiondecl() -> ExternFunctionDecl
            = "extern" __ "fn" __ name:identifier_not_ignored() _ 
                "(" _ parameter_types:typ() ** (_ "," _) _ ")" _ 
                "->" _ return_type:typ() 
            {
                ExternFunctionDecl { name, parameter_types, return_type }
            }

        /// Parse a `particle` primary block
        pub rule particle() -> Particle
            = "particle" __ name:identifier_not_ignored() _ 
                "{" _ members:(
                    _ "}" { Vec::new() } // Default to empty vector
                    / (m:particle_member() ** (_ "," _) _ "}" {m})
                )
            {
                Particle { name, members }
            }

        rule particle_member() -> ParticleMember 
            = name:identifier_not_ignored() _ ":" _ mutable:("mut" __)? _ typ_pos:typ_or_position() { 
                ParticleMember { name, mutable: !mutable.is_none(), typ: typ_pos.0, is_position: typ_pos.1 }
            }

        /// Parse an `interaction` primary block
        pub rule interaction() -> Interaction
            = "interaction" __ name:identifier_not_ignored() _
                "(" _ name_a:identifier() _ ":" _ type_a:identifier_not_ignored() _
                "," _ name_b:identifier() _ ":" _ type_b:identifier_not_ignored() _ ")" _
                "for" distance_vec:( __ "|" _ d:identifier_not_ignored() _ "|" _ "=" {d})? __ distance:identifier() _ "<" _ cutoff:float_ctc() _
                "{" _ common_block:("common" _ "{" _ b:statement() ** (_ ";" _) _ ";" _ "}" {b})? _ // TODO: Clean this up            
                    quantities:(
                    _ "}" { Vec::new() } // Default to empty vector
                    / (q:interaction_quantity() ** ( _ ) _ "}" {q})
                )
            {
                Interaction {
                    name,
                    name_a, type_a,
                    name_b, type_b,
                    distance, distance_vec, cutoff,
                    common_block, quantities
                }
            }

        rule interaction_quantity() -> InteractionQuantity
            = "quantity" _ name:identifier_not_ignored() _ 
                "-" _ "[" _ reduction_method:reduction_method()  _ "]" _ "->" _
                "(" _ target_a:identifier_not_ignored() _ "," _ symmetry:interaction_symmetry()? _ target_b:identifier_not_ignored() _ ")" _
                "{" _ expression:expression_block() _ "}"
            {
                let symmetry = symmetry.unwrap_or(InteractionSymmetry::Symmetric);
                InteractionQuantity {
                    name,
                    reduction_method,
                    target_a,
                    target_b,
                    symmetry,
                    expression
                }
            }

        rule reduction_method() -> ReductionMethod
            = "sum" { ReductionMethod::Sum }

        rule interaction_symmetry() -> InteractionSymmetry
            = "-" { InteractionSymmetry::Antisymmetric }
            / "!" { InteractionSymmetry::Asymmetric }

        /// Parse a simulation block
        pub rule simulation() -> Simulation
            = "simulation" _ name:identifier_not_ignored() _ "{" _
                default_particle:("default" _ "particle" _ dp:identifier_not_ignored() _ ";" {dp})? _
                blocks:simulation_block() ** _ _
                "}"
            { 
                Simulation {
                    name,default_particle,blocks
                }
            }

        rule simulation_block() -> SimulationBlock
            = "once" __ step:nat_ctc() _ "{" _ subblocks:simulation_sub_block()* _ "}" {
                SimulationBlock::Once(OnceBlock {
                    step, subblocks
                })
            }
            / "step" __ step_range:step_range()? _ "{" _ subblocks:simulation_sub_block() ** ( _ ) _ "}" {
                SimulationBlock::Step( StepBlock {
                    step_range: step_range.unwrap_or_default(), subblocks
                })
            }
        
        rule simulation_sub_block() -> SimulationSubBlock
            = "particle" __ particle:identifier_not_ignored() _ statements:statement_block()
            {
                SimulationSubBlock {
                    statements, particle: Some(particle)
                }
            }
            / &statement() _ statements:statement() ** ( _ ";" _ ) _ ";" {
                SimulationSubBlock {
                    statements, particle: None
                }
            }

        pub(crate) rule step_range() -> StepRange
            = start:nat_ctc()? _ ".." _ end:nat_ctc()? _ step:("," _ s:nat_ctc() {s})? {
                let mut range: StepRange = Default::default();
                range.start = start.unwrap_or(range.start);
                range.end   = end.unwrap_or(range.end);
                range.step  = step.unwrap_or(range.step);
                range
            }
            / step:nat_ctc() { StepRange {
                step, ..Default::default()
            }}

        // == Expressions and statements ==

        /// Parse a block expression
        rule expression_block() -> Expression
            = statements:(s:statement() ** ( _ ";" _ ) _ ";" {s})? _ expr:expression() {
                let statements = statements.unwrap_or(vec![]);
                Expression::Block(BlockExpression {
                    statements, expression: Box::new(expr)
                })
            }

        /// Parse a non-block expression
        pub rule expression() -> Expression = precedence!{
            lhs:(@) _ "+" _ rhs:@ { make_binop(lhs,rhs,BinaryOperator::Add) }
            lhs:(@) _ "-" _ rhs:@ { make_binop(lhs,rhs,BinaryOperator::Sub) }
            --
            lhs:(@) _ "*" _ rhs:@ { make_binop(lhs,rhs,BinaryOperator::Mul) }
            lhs:(@) _ "/" _ rhs:@ { make_binop(lhs,rhs,BinaryOperator::Div) }
            --
            fn_name:identifier_not_ignored() _ "(" _ parameters:expression() ** (_ "," _) _ ")"  {
                Expression::FunctionCall(FunctionCall {
                    fn_name,
                    parameters
                })
            }
            namespace:identifier_not_ignored() _ "." _ name:identifier_not_ignored() {
                Expression::Atom(Atom::NamespaceVariable{namespace, name})
            }
            // Currently only one-dimensional arrays are supported and
            // all arrays must be bound to an identifier first (TODO)
            array:identifier_not_ignored() _ "[" _ index:nat_ctc() _ "]" {
                Expression::Indexing(AtIndex {
                    array, index
                })
            }
            "[" _ elements:expression() ** (_ "," _) _ "]" {
                Expression::AdHocArray(AdHocArrayExpression{ elements })
            }
            atom:atom() {
                Expression::Atom(atom)
            }
            "(" _ expr:expression() _ ")" { expr }
        }

        /// Parse an atomic expression
        rule atom() -> Atom 
            = x:numeric_literal() { Atom::Literal(x) }
            / name:identifier_not_ignored() { Atom::Variable(name) }

        rule statement_block() -> Vec<Statement>
            = "{" _ statements:statement() ** ( _ ";" _ ) _ ";" _ "}" {
                statements
            }
            / "{" _ "}" { vec![] }

        /// Parse a statement
        pub rule statement() -> Statement
            = "let" __ name:identifier_not_ignored() _ ":" _ typ:typ() _ "=" _ initial:expression() {
                Statement::Let(LetStatement {
                    name, initial, typ
                })
            }
            / "update" __ interaction:identifier_not_ignored()
                quantity:(_ "." _ q:identifier_not_ignored() {q})?
            {
                Statement::Update(UpdateStatement {
                    interaction, quantity
                })
            }
            / "call" __ name:identifier_not_ignored() {
                Statement::Call(CallStatement {
                    name
                })
            }
            / assignee:identifier_not_ignored() _ index:("[" _ idx:nat_ctc() _ "]" {idx})? _ "=" _ value:expression() {
                Statement::Assign(AssignStatement {
                    assignee, value, index
                })
            }

        // == General syntax elements ==

        /// Any type specification
        /// (This is a tuple to denote the position flag)
        rule typ_or_position() -> (FipsType, bool)
            = "position" {
                (FipsType::Array {
                    typ: Box::new(FipsType::Double),
                    length: CompileTimeConstant::Identifier(BUILTIN_CONST_NDIM.into())
                }, true)
            }
            / typ:typ() { (typ, false) }

        rule typ() -> FipsType
            = "f64" { FipsType::Double }
            / "i64" { FipsType::Int64 }
            / "[" _ typ:typ() _ ";" _ length:nat_ctc() _ "]" { 
                FipsType::Array {
                    typ: Box::new(typ),
                    length
                }
            }

        /// Natural number (including 0)
        rule nat() -> usize
            = s:$("0" / (['1'..='9']['0'..='9']*))
            { ?
                s.parse::<usize>().or_else(|err| { 
                    match err.kind() {
                        IntErrorKind::Empty => unreachable!(),
                        IntErrorKind::InvalidDigit => unreachable!(),
                        IntErrorKind::PosOverflow => Err("Cannot parse integer number (positive overflow)"),
                        IntErrorKind::NegOverflow => unreachable!(),
                        IntErrorKind::Zero => unreachable!(),
                        _ => Err("Cannot parse integer number (unknown reason)")
                    }
                } )
            }

        /// Floating point or integer literal (always tries to parse as integer first)
        rule numeric_literal() -> Literal 
            = ['n'|'N'] ['a'|'A'] ['n'|'N'] { Literal::Double(f64::NAN) }
            / "+"? ['i'|'I'] ['n'|'N'] ['f'|'F'] { Literal::Double(f64::INFINITY) }
            / "-" ['i'|'I'] ['n'|'N'] ['f'|'F'] { Literal::Double(f64::NEG_INFINITY) }
            / s:$(['+'|'-']? ['0'..='9']+ ("." ['0'..='9']*)? ((['e'|'E'] ['+'|'-']? ['0'..='9']+)?)?)
            { ?
                // First try to parse as integer
                match s.parse::<i64>() {
                    Ok(n) => Ok(Literal::Int64(n)),
                    Err(e) => {
                        match e.kind() {
                            // Integer literals should not silently overflow to floats
                            IntErrorKind::PosOverflow => Err("Positive integer overflow"),
                            IntErrorKind::NegOverflow => Err("Negative integer overflow"),
                            // If overflow was not the problem, try to parse as float
                            _ => {
                                s.parse::<f64>()
                                    .or_else(|err| { Err("Cannot parse float number") } )
                                    .map(|x| Literal::Double(x))
                            },
                        }
                    },
                }
            }

        /// Floating point compile_time_constant
        rule float_ctc() -> CompileTimeConstant<f64>
            = x:numeric_literal() {
                // Unify type of literal to float
                let x = match x {
                    Literal::Double(x) => x,
                    Literal::Int64(n) => n as f64,
                };
                CompileTimeConstant::Literal(x)
            }
            / name:identifier_not_ignored() { CompileTimeConstant::Identifier(name) }

        /// Natural compile time constant
        rule nat_ctc() ->  CompileTimeConstant<usize>
            = n:nat() { CompileTimeConstant::Literal(n) } 
            / name:identifier_not_ignored() { CompileTimeConstant::Identifier(name) }

        rule identifier() -> Identifier
            = !keyword() s:$(['a'..='z'|'A'..='Z']['a'..='z'|'A'..='Z'|'0'..='9'|'_']*) {
                match s {
                    "_" => Identifier::Ignored,
                    _ => Identifier::Named(String::from(s))
                }
            }

        rule identifier_not_ignored() -> String
            = ident:identifier() { ?
                match ident {
                    Identifier::Ignored => Err("Cannot use _ here"),
                    Identifier::Named(name) => Ok(name)
                }
            }

        rule keyword() -> ()
            = "particle" / "once" / "step" / "call" / "let" / "update"

        rule _() -> ()
            = quiet!{(
                [' '|'\n'|'\t'|'\r'] _) // 0 or more whitespaces
                / ("//" (!['\n'][_])* ['\n'] _) // Comment to EOL
                / ""}

        rule __() -> ()
            = quiet!{[' '|'\n'|'\t'|'\r'] _}
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn global_state() {
        // Invalid cases
        assert!(fips_parser::unit_member("global foo: i64").is_err(),
            "Missing semicolon in global state member declaration not caught");
        assert!(fips_parser::unit_member("global foo;").is_err(),
            "Missing type specification in global state member declaration not caught");
        assert!(fips_parser::unit_member("global : i64;").is_err(),
            "Missing member name in global state member declaration not caught");
        assert!(fips_parser::unit_member("global foo : i64 = 42;").is_err(),
            "Invalid assignment in global state member declaration not caught");
        // Valid cases
        assert_eq!(fips_parser::unit_member("global foo: f64;")
            .expect("Cannot parse immutable global state member"),
            UnitMember::GlobalStateMember(GlobalStateMember {
                name: "foo".into(),
                mutable: false,
                typ: FipsType::Double
            })
        );
        assert_eq!(fips_parser::unit_member("global foo: mut f64;")
        .expect("Cannot parse mutable global state member"),
        UnitMember::GlobalStateMember(GlobalStateMember {
            name: "foo".into(),
            mutable: true,
            typ: FipsType::Double
        })
    );
    }

    #[test]
    fn statements_let() {
        assert!(fips_parser::statement("let foo = 42").is_err(),
            "Let statement without type was not caught");
        assert!(fips_parser::statement("let : f64 = 42").is_err(),
            "Let statement without name was not caught");
        assert!(fips_parser::statement("let _: f64 = 42").is_err(),
            "Let statement with omitted name was not caught");
        assert!(fips_parser::statement("let _: f64").is_err(),
            "Let statement without initial value not caught");
        let parser_result = fips_parser::statement("let foo: f64 = 42")
            .expect("Cannot parse valid let statement");
        assert_eq!(parser_result, Statement::Let(LetStatement{
            name: "foo".into(),
            typ: FipsType::Double,
            initial: Expression::Atom(
                Atom::Literal(
                    Literal::Int64(42)
                )
            ) 
        }));
    }

    #[test]
    fn statements_assign() {
        assert!(fips_parser::statement(" = 1337").is_err(),
            "Assign statement without assignee not caught");
        assert!(fips_parser::statement("foo[] = 1337").is_err(),
            "Indexed assign statement with empty index not caught");
        assert!(fips_parser::statement("foo[1.2] = 1337").is_err(),
            "Indexed assign statement with invalid index not caught");
        assert!(fips_parser::statement("foo = ").is_err(),
            "Assign statement without value not caught");
        assert!(fips_parser::statement("foo[123] = ").is_err(),
            "Indexed assign statement without value not caught");
        let parser_result = fips_parser::statement("foo = 1337")
            .expect("Cannot parse valid assign statement");
        assert_eq!(parser_result, Statement::Assign(AssignStatement { 
            assignee: "foo".into(),
            value: Expression::Atom(Atom::Literal(Literal::Int64(1337))),
            index: None
        }));
        let parser_result = fips_parser::statement("foo[123] = 1337")
            .expect("Cannot parse valid assign statement");
        assert_eq!(parser_result, Statement::Assign(AssignStatement { 
            assignee: "foo".into(),
            value: Expression::Atom(Atom::Literal(Literal::Int64(1337))),
            index: Some(CompileTimeConstant::Literal(123))
        }));
    }

    #[test]
    fn statements_update() {
        assert!(fips_parser::statement("update").is_err(),
            "Update statement without interaction name not caught");
        let parser_result = fips_parser::statement("update myinteraction")
            .expect("Cannot parse valid update statement without quantity");
        assert_eq!(parser_result, Statement::Update(UpdateStatement {
            interaction: "myinteraction".into(),
            quantity: None
        }));
        let parser_result = fips_parser::statement("update myinteraction.myquantity")
            .expect("Cannot parse valid update statement with quantity");
        assert_eq!(parser_result, Statement::Update(UpdateStatement {
            interaction: "myinteraction".into(),
            quantity: Some("myquantity".into())
        }));

    }

    #[test]
    fn statements_call() {
        assert!(fips_parser::statement("call").is_err(),
            "Call statement without callback name not caught");
        let parser_result = fips_parser::statement("call mycallback")
            .expect("Cannot parse valid call statement");
        assert_eq!(parser_result, Statement::Call(CallStatement {
            name: "mycallback".into()
        }));

    }

    #[test]
    fn expression_atom() {
        // Test invalid literals
        assert!(fips_parser::expression("41foobar").is_err(),
            "Invalid identifier atom not caught");
        assert!(fips_parser::expression("41.2.3").is_err(),
            "Invalid numeric literal not caught");
        assert!(fips_parser::expression("9223372036854775808").is_err(),
            "Overflowing integer literal not caught");
        // Test integer literals
        let integer_test_set: Vec<(_, i64)> = vec![
            ("0", 0),
            ("+0", 0),
            ("-0", 0),
            ("123", 123),
            ("-123", -123),
            ("(123)", 123),
            ("(-123)", -123),
            ("9223372036854775807", 9223372036854775807),
            ("-9223372036854775808", -9223372036854775808)
        ];
        for (s,n) in integer_test_set {
            assert_eq!(fips_parser::expression(s)
                .expect(&format!("Cannot parse integer literal {} as expression", s)),
                Expression::Atom(Atom::Literal(Literal::Int64(n)))
            );
        }
        // Test floating point literals
        let float_test_set = vec![
            ("0.0", 0.0),
            ("+0.0", 0.0),
            ("-0.0", -0.0),
            ("123.456", 123.456),
            ("-123.456", -123.456),
            ("123456e-3", 123456e-3),
            ("123456.789e-3", 123456.789e-3),
            ("inf", f64::INFINITY),
            ("+inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
            ("nan", f64::NAN),
        ];
        for (s,x) in float_test_set {
            let parser_result = fips_parser::expression(s)
                .expect(&format!("Cannot parse float literal {} as expression", s));
            if x.is_nan() {
                match parser_result {
                    Expression::Atom(Atom::Literal(Literal::Double(y))) => {
                        assert!(y.is_nan(), "Expected NaN but got {}", y);
                    },
                    _ => panic!("Expected Atom(Literal(Double(NaN))) but got {:?}", parser_result)
                }
            }
            else {
                assert_eq!(parser_result,
                    Expression::Atom(Atom::Literal(Literal::Double(x)))
                );
            }
        }
        // Test identifiers
        assert_eq!(fips_parser::expression("foobar")
            .expect(&format!("Cannot parse identifier as expression")),
            Expression::Atom(Atom::Variable("foobar".into()))
        );
        assert_eq!(fips_parser::expression("foo.bar")
            .expect(&format!("Cannot parse identifier as expression")),
            Expression::Atom(Atom::NamespaceVariable { namespace: "foo".into(), name: "bar".into() })
        );
    }

    #[test]
    fn expression_arrays() {
        // Helper functions
        let make_int_atom = |i| {
            Expression::Atom(Atom::Literal(Literal::Int64(i)))
        };
        let make_ident_atom = |name: &str| {
            Expression::Atom(Atom::Variable(name.into()))
        };
        // Test invalid ad-hoc arrays
        assert!(fips_parser::expression("[1,2,3").is_err(),
            "Unbalanced array brackets not caught");
        assert!(fips_parser::expression("[,1,2,3]").is_err(),
            "Missing array element at the start not caught");
        assert!(fips_parser::expression("[1,2,,3]").is_err(),
            "Missing array element inbetween not caught");
        assert!(fips_parser::expression("[1,2,3,]").is_err(),
            "Missing array element at the end not caught");
        // Test valid ad-hoc arrays
        assert_eq!(fips_parser::expression("[]")
            .expect(&format!("Cannot parse empty array")),
            Expression::AdHocArray(AdHocArrayExpression { elements: vec![] })
        );
        assert_eq!(fips_parser::expression("[1]")
            .expect(&format!("Cannot parse one-element array")),
            Expression::AdHocArray(AdHocArrayExpression {
                elements: vec![make_int_atom(1)] 
            })
        );
        assert_eq!(fips_parser::expression("[1,2,a,b]")
        .expect(&format!("Cannot parse multi-element array")),
        Expression::AdHocArray(AdHocArrayExpression {
            elements: vec![
                make_int_atom(1), make_int_atom(2),
                make_ident_atom("a"), make_ident_atom("b")
            ]})
        );
    }

    #[test]
    fn expression_indexing() {
        // Test invalid array index access
        assert!(fips_parser::expression("myarr[]").is_err(),
            "Empty array index not caught");
        assert!(fips_parser::expression("myarr[-1]").is_err(),
            "Negative array index not caught");
        // Test valid indexing expressions
        assert_eq!(fips_parser::expression("foo[0]")
            .expect(&format!("Cannot parse valid array indexing")),
            Expression::Indexing(AtIndex {
                array: "foo".into(),
                index: CompileTimeConstant::Literal(0)
            })
        );
    }

    #[test]
    fn expression_identifier_namespace() {
        // Test invalid identifier namespacing
        assert!(fips_parser::expression("myparticle.").is_err(),
            "Missing identifier not caught");
        assert!(fips_parser::expression(".myident").is_err(),
            "Missing namespace not caught");
        assert!(fips_parser::expression(".").is_err(),
            "Missing identifier and namespace not caught");
        // Test valid namespacing
        assert_eq!(fips_parser::expression("myparticle.myident")
            .expect(&format!("Cannot parse empty array")),
            Expression::Atom(Atom::NamespaceVariable {
                namespace: "myparticle".into(),
                name: "myident".into()
            })
        );
    }

    #[test]
    fn expression_function_call() {
        // Test invalid function calls
        assert!(fips_parser::expression("foo(").is_err(),
            "Missing closing parenthesis not caught");
        assert!(fips_parser::expression("foo(bar,").is_err(),
            "Missing closing parenthesis with parameters not caught");
        assert!(fips_parser::expression("foo(bar,)").is_err(),
            "Missing function argument not caught");
        assert!(fips_parser::expression("foo)").is_err(),
            "Missing opening parenthesis not caught");
        // Test valid function call
        assert_eq!(fips_parser::expression("foo(bar)")
            .expect(&format!("Cannot parse empty array")),
            Expression::FunctionCall(FunctionCall {
                fn_name: "foo".into(),
                parameters: vec![
                    Expression::Atom(Atom::Variable("bar".into()))
                ]
            })
        );
    }

    #[ignore]
    #[test]
    fn euler() {
        let input = r#"// Simple Euler integrator for point-like particles

// Format for a particle:
particle PointLike {
    // Position is special and cannot be redefined, but aliased (array specifier is not allowed,
    // as position is always NDIM dimensional)
    x : mut position,
    v : mut [f64; NDIM], // NDIM is dimensionality of problem
    F : mut [f64; NDIM], // Quantities bound to interaction calculations later also need to be declared
    mass: f64            // Constants have to be defined in Rust code (can be either per-particle or
                         // per-type)
}

// Particles can also inherit members
particle Orientable extends PointLike {
    phi : mut [f64; ROTDIM], // ROTDIM is user defined (1 for 2D, 3 for 3D)
    omega : mut [f64; ROTDIM],
    torque : mut [f64; ROTDIM],
    inertia : f64
}

// Format for an interaction block:
// The "dyn" marks an interaction between the first type and all types extending the second
// (similar to the meaning of dyn in Rust)
// Here we also bind names to the two interacting particles so we can access their members
// (i.e. the carge for E/M interactions)
interaction myinteraction (p1: PointLike, p2: dyn PointLike) for r < CUTOFF {
    // Sole argument is the distance (commonly used and already calculated in neighbour checking)
    // (p1 and p2 can be used too, obviously)
    // Syntax: quantity <name> -[<reduction method>]-> <member in first>, <member in second> { <body> }
    quantity myforce -[sum]-> (F, F) {
        // Rust like expression syntax
        4.0*EPSILON*((SIGMA/r))
    }
}
// Interactions are always mutual (this is enforced in syntax to avoid accidental violations of e.g.
// Newton's third law)!

// Types of value reduction:
// * sum: (Vector) sum all contributions together

// Simulation block
simulation MySim {
    // Indicate that PointLike is the default particle for step blocks
    default particle PointLike;
    // Once blocks are executed once in the given timestep
    once 0 {

    }
    // Step blocks contain a single time step
    // Multiple step blocks can be defined in one simulation, so they are interleaved as necessary
    // Step blocks are executed every x timesteps (default is 1, i.e. every timestep):
    // step x {...}
    // If multiple step blocks are to be executed in the same timestep, they are executed in the order of definition
    step { 
        // This block can be omitted, since PointLike is the default particle type
        particle PointLike {
            update myinteraction; // This causes all quantities of myinteraction to be updated across all processors
            // All members of the particle are accessible without extra scoping
            // Members bound in interactions are technically write-accessible, but this should issue a warning
            // since their value will be overwritten by any update command
            v = v + F / m * DT; // Euler step for velocity (DT is the timestep constant)
            x = x + v * DT;
        }
    }
}"#;
        
        fips_parser::unit(input).expect("Cannot parse input");
    }
}