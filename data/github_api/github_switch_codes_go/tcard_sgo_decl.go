// Repository: tcard/sgo
// File: sgo/types/decl.go

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"

	"github.com/tcard/sgo/sgo/ast"
	"github.com/tcard/sgo/sgo/constant"
	"github.com/tcard/sgo/sgo/token"
)

func (check *Checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", obj.Name()) // secondary error, \t indented
	}
}

func (check *Checker) declare(scope *Scope, id *ast.Ident, obj Object, pos token.Pos) {
	// spec: "The blank identifier, represented by the underscore
	// character _, may be used in a declaration like any other
	// identifier but the declaration does not introduce a new
	// binding."
	if obj.Name() != "_" {
		if alt := scope.Insert(obj); alt != nil {
			check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
			check.reportAltDecl(alt)
			return
		}
		obj.setScopePos(pos)
	}
	if id != nil {
		check.recordDef(id, obj)
	}
}

// objDecl type-checks the declaration of obj in its respective (file) context.
// See check.typ for the details on def and path.
func (check *Checker) objDecl(obj Object, def *Named, path []*TypeName) {
	if obj.Type() != nil {
		return // already checked - nothing to do
	}

	if trace {
		check.trace(obj.Pos(), "-- declaring %s", obj.Name())
		check.indent++
		defer func() {
			check.indent--
			check.trace(obj.Pos(), "=> %s", obj)
		}()
	}

	d := check.objMap[obj]
	if d == nil {
		check.dump("%s: %s should have been declared", obj.Pos(), obj.Name())
		// unreachable()
		switch obj := obj.(type) {
		case *Const:
			obj.typ = Typ[Invalid]
		case *Var:
			obj.typ = Typ[Invalid]
		case *TypeName:
			obj.typ = Typ[Invalid]
		case *Func:
			obj.typ = Typ[Invalid]
		default:
			unreachable()
		}
		return
	}

	// save/restore current context and setup object context
	defer func(ctxt context) {
		check.context = ctxt
	}(check.context)
	check.context = context{
		scope: d.file,
	}

	// Const and var declarations must not have initialization
	// cycles. We track them by remembering the current declaration
	// in check.decl. Initialization expressions depending on other
	// consts, vars, or functions, add dependencies to the current
	// check.decl.
	switch obj := obj.(type) {
	case *Const:
		check.decl = d // new package-level const decl
		check.constDecl(obj, d.typ, d.init)
	case *Var:
		check.decl = d // new package-level var decl
		check.varDecl(obj, d.lhs, d.entangledLhs, d.typ, d.init)
	case *TypeName:
		// invalid recursive types are detected via path
		check.typeDecl(obj, d.typ, def, path, d.alias)
	case *Func:
		// functions may be recursive - no need to track dependencies
		check.funcDecl(obj, d)
	default:
		unreachable()
	}
}

func (check *Checker) constDecl(obj *Const, typ, init ast.Expr) {
	assert(obj.typ == nil)

	if obj.visited {
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// use the correct value of iota
	assert(check.iota == nil)
	check.iota = obj.val
	defer func() { check.iota = nil }()

	// provide valid constant value under all circumstances
	obj.val = constant.MakeUnknown()

	// determine type, if any
	if typ != nil {
		t := check.typ(typ)
		if !isConstType(t) {
			// don't report an error if the type is an invalid C (defined) type
			// (issue #22090)
			if t.Underlying() != Typ[Invalid] {
				check.errorf(typ.Pos(), "invalid constant type %s", t)
			}
			obj.typ = Typ[Invalid]
			return
		}
		obj.typ = t
	}

	// check initialization
	var x operand
	if init != nil {
		check.expr(&x, init)
	}
	check.initConst(obj, &x)
}

func (check *Checker) varDecl(obj *Var, lhs []*Var, entangledLhs *Var, typ, init ast.Expr) {
	assert(obj.typ == nil)

	if obj.visited {
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// var declarations cannot use iota
	assert(check.iota == nil)

	// determine type, if any
	if typ != nil {
		obj.setType(check.typ(typ))
		// We cannot spread the type to all lhs variables if there
		// are more than one since that would mark them as checked
		// (see Checker.objDecl) and the assignment of init exprs,
		// if any, would not be checked.
		//
		// TODO(gri) If we have no init expr, we should distribute
		// a given type otherwise we need to re-evalate the type
		// expr for each lhs variable, leading to duplicate work.
	}

	// check initialization
	if init == nil {
		if typ == nil {
			// error reported before by arityMatch
			obj.typ = Typ[Invalid]
		}
		if has, _ := check.hasZeroValue(obj.typ); !has {
			obj.usable = false
		}
		return
	}

	obj.usable = true
	if debugUsable {
		fmt.Println("USABLE varDecl:", obj.name, fmt.Sprintf("%p", obj), obj.usable)
	}

	shouldCheckRhs := !check.conf.IgnoreTopLevelVarValues || check.scope.parent.parent != Universe
	if obj.typ == nil && !shouldCheckRhs {
		obj.setType(Typ[Invalid])
	}

	if lhs == nil || (len(lhs) == 1 && entangledLhs == nil) {
		assert(lhs == nil || lhs[0] == obj)
		if shouldCheckRhs {
			var x operand
			check.expr(&x, init)
			check.initVar(obj, &x, "variable declaration")
		}
		return
	}

	if debug {
		// obj must be one of lhs
		found := false
		for _, lhs := range lhs {
			if obj == lhs {
				found = true
				break
			}
		}
		if !found {
			panic("inconsistent lhs")
		}
	}
	if shouldCheckRhs {
		// We have multiple variables on the lhs and one init expr.
		// Make sure all variables have been given the same type if
		// one was specified, otherwise they assume the type of the
		// init expression values (was issue #15755).
		if typ != nil {
			for _, lhs := range append(lhs, entangledLhs) {
				if lhs != nil {
					lhs.typ = obj.typ
				}
			}
		}

		check.initVars(lhs, &ast.ExprList{List: []ast.Expr{init}}, token.NoPos, entangledLhs)
	}

	if entangledLhs != nil {
		entangledLhs.collapses = lhs
		for _, v := range lhs {
			v.usable = false
			if debugUsable {
				fmt.Println("USABLE varDecl2:", v.name, fmt.Sprintf("%p", v), v.usable)
			}
		}
	}
}

// underlying returns the underlying type of typ; possibly by following
// forward chains of named types. Such chains only exist while named types
// are incomplete.
func underlying(typ Type) Type {
	for {
		n, _ := typ.(*Named)
		if n == nil {
			break
		}
		typ = n.underlying
	}
	return typ
}

func (n *Named) setUnderlying(typ Type) {
	if n != nil {
		n.underlying = typ
	}
}

func (check *Checker) typeDecl(obj *TypeName, typ ast.Expr, def *Named, path []*TypeName, alias bool) {
	assert(obj.typ == nil)

	// type declarations cannot use iota
	assert(check.iota == nil)

	if alias {

		obj.typ = Typ[Invalid]
		obj.typ = check.typExpr(typ, nil, append(path, obj))

	} else {

		named := &Named{obj: obj}
		def.setUnderlying(named)
		obj.typ = named // make sure recursive type declarations terminate

		// determine underlying type of named
		check.typExpr(typ, named, append(path, obj))

		// The underlying type of named may be itself a named type that is
		// incomplete:
		//
		//	type (
		//		A B
		//		B *C
		//		C A
		//	)
		//
		// The type of C is the (named) type of A which is incomplete,
		// and which has as its underlying type the named type B.
		// Determine the (final, unnamed) underlying type by resolving
		// any forward chain (they always end in an unnamed type).
		named.underlying = underlying(named.underlying)

	}

	// check and add associated methods
	// TODO(gri) It's easy to create pathological cases where the
	// current approach is incorrect: In general we need to know
	// and add all methods _before_ type-checking the type.
	// See https://play.golang.org/p/WMpE0q2wK8
	check.addMethodDecls(obj)
}

func (check *Checker) addMethodDecls(obj *TypeName) {
	// get associated methods
	methods := check.methods[obj.name]
	if len(methods) == 0 {
		return // no methods
	}
	delete(check.methods, obj.name)

	// use an objset to check for name conflicts
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	base, _ := obj.typ.(*Named) // nil if receiver base type is type alias
	if base != nil {
		if t, _ := base.underlying.(*Struct); t != nil {
			for _, fld := range t.fields {
				if fld.name != "_" {
					assert(mset.insert(fld) == nil)
				}
			}
		}

		// Checker.Files may be called multiple times; additional package files
		// may add methods to already type-checked types. Add pre-existing methods
		// so that we can detect redeclarations.
		for _, m := range base.methods {
			assert(m.name != "_")
			assert(mset.insert(m) == nil)
		}
	}

	// type-check methods
	for _, m := range methods {
		// spec: "For a base type, the non-blank names of methods bound
		// to it must be unique."
		if m.name != "_" {
			if alt := mset.insert(m); alt != nil {
				switch alt.(type) {
				case *Var:
					check.errorf(m.pos, "field and method with the same name %s", m.name)
				case *Func:
					check.errorf(m.pos, "method %s already declared for %s", m.name, obj)
				default:
					unreachable()
				}
				check.reportAltDecl(alt)
				continue
			}
		}

		// type-check
		check.objDecl(m, nil, nil)

		// methods with blank _ names cannot be found - don't keep them
		if base != nil && m.name != "_" {
			base.methods = append(base.methods, m)
		}
	}
}

func (check *Checker) funcDecl(obj *Func, decl *declInfo) {
	assert(obj.typ == nil)

	// func declarations cannot use iota
	assert(check.iota == nil)

	sig := new(Signature)
	obj.typ = sig // guard against cycles
	fdecl := decl.fdecl
	check.funcType(sig, fdecl.Recv, fdecl.Type)
	if sig.recv == nil && obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
		check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
		// ok to continue
	}

	// function body must be type-checked after global declarations
	// (functions implemented elsewhere have no body)
	if !check.conf.IgnoreFuncBodies && fdecl.Body != nil {
		check.later(obj.name, decl, sig, fdecl.Body)
	}
}

func (check *Checker) declStmt(decl ast.Decl) {
	pkg := check.pkg

	switch d := decl.(type) {
	case *ast.BadDecl:
		// ignore

	case *ast.GenDecl:
		var last *ast.ValueSpec // last ValueSpec with type or init exprs seen
		for iota, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ValueSpec:
				switch d.Tok {
				case token.CONST:
					// determine which init exprs to use
					switch {
					case s.Type != nil || len(s.Values.List) > 0:
						last = s
					case last == nil:
						last = new(ast.ValueSpec) // make sure last exists
					}

					// declare all constants
					lhs := make([]*Const, len(s.Names.List))
					for i, name := range s.Names.List {
						obj := NewConst(name.Pos(), pkg, name.Name, nil, constant.MakeInt64(int64(iota)))
						lhs[i] = obj

						var init ast.Expr
						if last.Values != nil && i < len(last.Values.List) {
							init = last.Values.List[i]
						}

						check.constDecl(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					// spec: "The scope of a constant or variable identifier declared
					// inside a function begins at the end of the ConstSpec or VarSpec
					// (ShortVarDecl for short variable declarations) and ends at the
					// end of the innermost containing block."
					scopePos := s.End()
					for i, name := range s.Names.List {
						check.declare(check.scope, name, lhs[i], scopePos)
					}

				case token.VAR:
					lhs0 := make([]*Var, 0, len(s.Names.List))
					var entangledLhs *Var
					for i, name := range s.Names.List {
						v := NewVar(name.Pos(), pkg, name.Name, nil)
						if s.Names.EntangledPos > 0 && i == s.Names.EntangledPos-1 {
							entangledLhs = v
						} else {
							lhs0 = append(lhs0, v)
						}
					}

					// initialize all variables
					for i, obj := range lhs0 {
						var lhs []*Var
						var init ast.Expr
						switch len(s.Values.List) {
						case len(s.Names.List):
							// lhs and rhs match
							init = s.Values.List[i]
						case 1:
							// rhs is expected to be a multi-valued expression
							lhs = lhs0
							init = s.Values.List[0]
						default:
							if i < len(s.Values.List) {
								init = s.Values.List[i]
							}
						}
						check.varDecl(obj, lhs, entangledLhs, s.Type, init)
						if len(s.Values.List) == 1 {
							// If we have a single lhs variable we are done either way.
							// If we have a single rhs expression, it must be a multi-
							// valued expression, in which case handling the first lhs
							// variable will cause all lhs variables to have a type
							// assigned, and we are done as well.
							if debug {
								for _, obj := range lhs0 {
									assert(obj.typ != nil)
								}
							}
							break
						}
					}

					check.arityMatch(s, nil)

					// declare all variables
					// (only at this point are the variable scopes (parents) set)
					scopePos := s.End() // see constant declarations
					for i, name := range s.Names.List {
						var v *Var
						if s.Names.EntangledPos > 0 && i == s.Names.EntangledPos-1 {
							v = entangledLhs
						} else {
							v = lhs0[i]
						}
						// see constant declarations
						check.declare(check.scope, name, v, scopePos)
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				// spec: "The scope of a type identifier declared inside a function
				// begins at the identifier in the TypeSpec and ends at the end of
				// the innermost containing block."
				scopePos := s.Name.Pos()
				check.declare(check.scope, s.Name, obj, scopePos)
				check.typeDecl(obj, s.Type, nil, nil, s.Assign.IsValid())

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
