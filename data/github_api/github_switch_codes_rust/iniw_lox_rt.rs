// Repository: iniw/lox
// File: src/rt.rs

use mucow::MuCow;
use std::{collections::hash_map::Entry, fmt::Display, mem::replace};

use crate::{
    syntax::{AssignmentOp, BinaryOp, Block, Expr, Literal, Stmt, UnaryOp},
    LoxNumber,
};

#[derive(Debug, Clone)]
pub struct TreeWalker<'ast> {
    env: EnvManager<'ast>,
}

impl<'ast> TreeWalker<'ast> {
    pub fn new() -> Self {
        Self {
            env: EnvManager::new(),
        }
    }

    pub fn execute(&mut self, statements: Vec<Stmt<'ast>>) -> Execution<'ast> {
        let mut last_statement = Stated::Nothing;
        for statement in statements {
            last_statement = self.execute_statement(statement)?;
        }
        println!("Finishing with {} envs", self.env.list.len());
        Ok(last_statement)
    }

    fn execute_statement(&mut self, statement: Stmt<'ast>) -> Execution<'ast> {
        match statement {
            Stmt::Block { statements } => self.execute_block(statements),
            Stmt::Break => self.execute_break(),
            Stmt::ClassDecl {
                identifier,
                methods,
                superclass,
            } => self.execute_class_decl(identifier, methods, superclass),
            Stmt::Continue => self.execute_continue(),
            Stmt::Empty => self.execute_empty(),
            Stmt::Expr { expr } => self.execute_expression(expr),
            Stmt::FunDecl {
                identifier,
                parameters,
                body,
            } => self.execute_fun_decl(identifier, parameters, body),
            Stmt::If {
                condition,
                branch,
                else_branch,
            } => self.execute_if(condition, *branch, else_branch.map(|s| *s)),
            Stmt::Print { expr } => self.execute_print(expr),
            Stmt::Return { expr } => self.execute_return(expr),
            Stmt::VarDecl { identifier, expr } => self.execute_var_decl(identifier, expr),
            Stmt::While { condition, body } => self.execute_while(condition, *body),
        }
    }

    fn execute_block(&mut self, statements: Vec<Stmt<'ast>>) -> Execution<'ast> {
        self.env.push_scope();
        let result = self.perform_block(statements);
        self.env.pop_scope();
        result
    }

    fn execute_break(&self) -> Execution<'ast> {
        Ok(Stated::ControlFlow(ControlFlow::Break))
    }

    fn execute_class_decl(
        &mut self,
        identifier: &'ast str,
        methods: Vec<(&'ast str, Vec<&'ast str>, Block<'ast>)>,
        _superclass: Option<&'ast str>,
    ) -> Execution<'ast> {
        let scope = {
            let mut scope = Scope::with_capacity_and_hasher(methods.len(), Default::default());
            for (identifier, parameters, body) in methods {
                let function = Function {
                    identifier: Some(identifier),
                    parameters,
                    body,
                    parent_env: self.env.snapshot(),
                };
                scope.insert(identifier, Value::Callable(function));
            }
            scope
        };

        let class = Class {
            identifier,
            scope,
            superclass: None,
        };

        self.env.declare_symbol(identifier, Value::Class(class))?;

        Ok(Stated::Nothing)
    }

    fn execute_continue(&self) -> Execution<'ast> {
        Ok(Stated::ControlFlow(ControlFlow::Continue))
    }

    fn execute_empty(&self) -> Execution<'ast> {
        // No-op
        Ok(Stated::Nothing)
    }

    fn execute_expression(&mut self, expr: Expr<'ast>) -> Execution<'ast> {
        self.evaluate_expression(expr).map(Stated::Value)
    }

    fn execute_fun_decl(
        &mut self,
        identifier: &'ast str,
        parameters: Vec<&'ast str>,
        body: Block<'ast>,
    ) -> Execution<'ast> {
        let function = Function {
            identifier: Some(identifier),
            parameters,
            body,
            parent_env: self.env.snapshot(),
        };

        self.env
            .declare_symbol(identifier, Value::Callable(function))?;

        Ok(Stated::Nothing)
    }

    fn execute_if(
        &mut self,
        condition: Expr<'ast>,
        branch: Stmt<'ast>,
        else_branch: Option<Stmt<'ast>>,
    ) -> Execution<'ast> {
        if self.evaluate_expression(condition)?.is_truthy() {
            self.execute_statement(branch)
        } else if let Some(else_branch) = else_branch {
            self.execute_statement(else_branch)
        } else {
            Ok(Stated::Nothing)
        }
    }

    fn perform_block(&mut self, statements: Vec<Stmt<'ast>>) -> Execution<'ast> {
        for statement in statements {
            if let flow @ Stated::ControlFlow(_) = self.execute_statement(statement)? {
                return Ok(flow);
            }
        }
        Ok(Stated::Nothing)
    }

    fn execute_print(&mut self, expr: Expr<'ast>) -> Execution<'ast> {
        println!("{}", self.evaluate_expression(expr)?);
        Ok(Stated::Nothing)
    }

    fn execute_return(&mut self, expr: Option<Expr<'ast>>) -> Execution<'ast> {
        let value = if let Some(expr) = expr {
            self.evaluate_expression(expr)?
        } else {
            Value::Undefined
        };

        Ok(Stated::ControlFlow(ControlFlow::Return(value)))
    }

    fn execute_var_decl(
        &mut self,
        identifier: &'ast str,
        expr: Option<Expr<'ast>>,
    ) -> Execution<'ast> {
        let value = expr
            .map(|expr| self.evaluate_expression(expr))
            .transpose()?;

        let value = value.unwrap_or(Value::Nil);

        self.env.declare_symbol(identifier, value)?;

        Ok(Stated::Nothing)
    }

    fn execute_while(&mut self, condition: Expr<'ast>, body: Stmt<'ast>) -> Execution<'ast> {
        while self.evaluate_expression(condition.clone())?.is_truthy() {
            let result = self.execute_statement(body.clone())?;
            if let Stated::ControlFlow(control_flow) = result {
                match control_flow {
                    ControlFlow::Break => break,
                    ControlFlow::Continue => continue,
                    ret @ ControlFlow::Return(_) => return Ok(Stated::ControlFlow(ret)),
                }
            }
        }

        Ok(Stated::Nothing)
    }

    fn evaluate_expression(&mut self, expr: Expr<'ast>) -> Evaluation<'ast> {
        match expr {
            Expr::Assignment {
                op,
                identifier,
                expr,
            } => self.evaluate_assignment(op, identifier, *expr),
            Expr::Binary { left, op, right } => self.evaluate_binary(*left, op, *right),
            Expr::FunctionCall { expr, arguments } => self.evaluate_function_call(*expr, arguments),
            Expr::Grouping { expr } => self.evaluate_grouping(*expr),
            Expr::Lambda { parameters, body } => self.evaluate_lambda(parameters, body),
            Expr::Literal(literal) => self.evaluate_literal(literal),
            Expr::PropertyAccess { expr, property } => {
                self.evaluate_property_access(*expr, property)
            }
            Expr::PropertyAssignment {
                object,
                property,
                op,
                value,
            } => self.evaluate_property_assignment(*object, property, op, *value),
            Expr::Symbol { identifier } => self.evaluate_symbol(identifier),
            Expr::Unary { op, expr } => self.evaluate_unary(op, *expr),
        }
    }

    fn evaluate_assignment(
        &mut self,
        op: AssignmentOp,
        identifier: &'ast str,
        expr: Expr<'ast>,
    ) -> Evaluation<'ast> {
        let value = self.evaluate_expression(expr)?;
        if let Some(var) = self.env.find_symbol(identifier) {
            match op {
                AssignmentOp::Set => *var = value,
                AssignmentOp::Add => var.add_assign(value)?,
                AssignmentOp::Sub => var.sub_assign(value)?,
                AssignmentOp::Mul => var.mul_assign(value)?,
                AssignmentOp::Div => var.div_assign(value)?,
            }
            Ok(var.clone())
        } else {
            Err(RuntimeError::UndefinedVariable(identifier))
        }
    }

    fn evaluate_binary(
        &mut self,
        left: Expr<'ast>,
        op: BinaryOp,
        right: Expr<'ast>,
    ) -> Evaluation<'ast> {
        match op {
            // Logical (handled separately to implement short-circuiting)
            BinaryOp::And => {
                let left = self.evaluate_expression(left)?;
                if left.is_truthy() {
                    let right = self.evaluate_expression(right)?;
                    return Ok(Value::Bool(right.is_truthy()));
                } else {
                    return Ok(Value::Bool(false));
                }
            }
            BinaryOp::Or => {
                let left = self.evaluate_expression(left)?;
                if left.is_truthy() {
                    return Ok(Value::Bool(true));
                } else {
                    let right = self.evaluate_expression(right)?;
                    return Ok(Value::Bool(right.is_truthy()));
                }
            }
            op => {
                let left = self.evaluate_expression(left)?;
                let right = self.evaluate_expression(right)?;
                match op {
                    // Relational
                    BinaryOp::Equal => Ok(Value::Bool(left.eq(&right))),
                    BinaryOp::NotEqual => Ok(Value::Bool(left.neq(&right))),
                    BinaryOp::GreaterThan => left.gt(&right).map(Value::Bool),
                    BinaryOp::GreaterThanEqual => left.gte(&right).map(Value::Bool),
                    BinaryOp::LessThan => left.lt(&right).map(Value::Bool),
                    BinaryOp::LessThanEqual => left.lte(&right).map(Value::Bool),

                    // Arithmetic
                    BinaryOp::Add => left.add(right),
                    BinaryOp::Sub => left.sub(right),
                    BinaryOp::Mul => left.mul(right),
                    BinaryOp::Div => left.div(right),

                    // Logical operators were handled already
                    _ => unreachable!(),
                }
            }
        }
    }

    fn evaluate_function_call(
        &mut self,
        expr: Expr<'ast>,
        arguments: Vec<Expr<'ast>>,
    ) -> Evaluation<'ast> {
        let value = self.evaluate_expression(expr)?;
        if let Value::Callable(callee) = value {
            if arguments.len() != callee.parameters.len() {
                return Err(RuntimeError::MismatchedArity {
                    expected: callee.parameters.len(),
                    got: arguments.len(),
                });
            }

            // Evaluate the arguments before replacing our environment
            let arguments = arguments
                .into_iter()
                .map(|expr| self.evaluate_expression(expr))
                .collect::<Result<Vec<Value<'ast>>, RuntimeError>>()?;

            // Allocate a new environment for the call
            let new_env = self.env.push(callee.parent_env);

            // Replace ours with it and store the previous
            let old_env = self.env.switch(new_env);

            // Inject the arguments
            for (identifier, value) in callee.parameters.into_iter().zip(arguments.into_iter()) {
                self.env.declare_symbol(identifier, value)?;
            }

            // NOTE: `perform_block` instead of `execute_block` to avoid pushing a useless scope
            let result = self.perform_block(callee.body);

            // Restore the active environment
            self.env.active = old_env;

            // If no edge was connected to the newly created environment we can safely dispose of it
            // This avoids pilling up "weak" environments
            if !self.env.get(new_env).is_parent {
                self.env.remove(new_env);
            }

            match result {
                Ok(Stated::ControlFlow(ControlFlow::Return(value))) => Ok(value),
                Ok(_) => Ok(Value::Nil),
                Err(e) => Err(e),
            }
        } else if let Value::Class(class) = value {
            Ok(Value::Instance(class))
        } else {
            Err(RuntimeError::InvalidCallee(value))
        }
    }

    fn evaluate_grouping(&mut self, expr: Expr<'ast>) -> Evaluation<'ast> {
        self.evaluate_expression(expr)
    }

    fn evaluate_lambda(
        &mut self,
        parameters: Vec<&'ast str>,
        body: Block<'ast>,
    ) -> Evaluation<'ast> {
        Ok(Value::Callable(Function {
            identifier: None,
            parameters,
            body,
            parent_env: self.env.snapshot(),
        }))
    }

    fn evaluate_literal(&self, literal: Literal<'ast>) -> Evaluation<'ast> {
        match literal {
            Literal::Nil => Ok(Value::Nil),
            Literal::False => Ok(Value::Bool(false)),
            Literal::True => Ok(Value::Bool(true)),
            Literal::Number(n) => Ok(Value::Number(n)),
            Literal::String(s) => Ok(Value::String(s.to_owned())),
        }
    }

    fn evaluate_property_access(
        &mut self,
        expr: Expr<'ast>,
        property: &'ast str,
    ) -> Evaluation<'ast> {
        let value = self.evaluate_expression(expr)?;
        if let Value::Instance(object) = value {
            object
                .scope
                .get(property)
                .cloned()
                .ok_or(RuntimeError::UndefinedProperty(property))
        } else {
            Err(RuntimeError::InvalidObject(value))
        }
    }

    fn evaluate_property_assignment(
        &mut self,
        expr: Expr<'ast>,
        property: &'ast str,
        op: AssignmentOp,
        value: Expr<'ast>,
    ) -> Evaluation<'ast> {
        let value = self.evaluate_expression(value)?;

        let mut object = self.get_property(expr)?;
        match op {
            AssignmentOp::Set => {
                object.scope.insert(property, value.clone());
                Ok(value)
            }
            op => {
                if let Some(var) = object.scope.get_mut(property) {
                    match op {
                        AssignmentOp::Add => var.add_assign(value)?,
                        AssignmentOp::Sub => var.sub_assign(value)?,
                        AssignmentOp::Mul => var.mul_assign(value)?,
                        AssignmentOp::Div => var.div_assign(value)?,
                        _ => unreachable!(),
                    };
                    Ok(var.clone())
                } else {
                    Err(RuntimeError::UndefinedProperty(property))
                }
            }
        }
    }

    fn evaluate_symbol(&mut self, identifier: &'ast str) -> Evaluation<'ast> {
        self.env
            .find_symbol(identifier)
            .cloned()
            .ok_or(RuntimeError::UndefinedVariable(identifier))
    }

    fn evaluate_unary(&mut self, op: UnaryOp, expr: Expr<'ast>) -> Evaluation<'ast> {
        let expr = self.evaluate_expression(expr)?;
        match op {
            UnaryOp::Minus => match expr {
                Value::Number(n) => Ok(Value::Number(-n)),
                expr => Err(RuntimeError::InvalidOperand(expr, op)),
            },
            UnaryOp::LogicalNot => Ok(Value::Bool(!expr.is_truthy())),
        }
    }

    fn get_property(&mut self, expr: Expr<'ast>) -> Result<MuCow<Class<'ast>>, RuntimeError<'ast>> {
        match expr {
            Expr::Symbol { identifier } => match self.env.find_symbol(identifier) {
                Some(Value::Instance(object)) => Ok(MuCow::Borrowed(object)),
                Some(value) => Err(RuntimeError::InvalidObject(value.clone())),
                None => Err(RuntimeError::UndefinedVariable(identifier)),
            },
            Expr::PropertyAccess { expr, property } => {
                let class = self.get_property(*expr)?;
                match class {
                    MuCow::Borrowed(class) => match class.scope.get_mut(property) {
                        Some(Value::Instance(object)) => Ok(MuCow::Borrowed(object)),
                        Some(value) => Err(RuntimeError::InvalidObject(value.clone())),
                        None => Err(RuntimeError::UndefinedProperty(property)),
                    },
                    MuCow::Owned(mut class) => match class.scope.remove(property) {
                        Some(Value::Instance(object)) => Ok(MuCow::Owned(object)),
                        Some(value) => Err(RuntimeError::InvalidObject(value)),
                        None => Err(RuntimeError::UndefinedProperty(property)),
                    },
                }
            }
            expr => match self.evaluate_expression(expr)? {
                Value::Instance(object) => Ok(MuCow::Owned(object)),
                value => Err(RuntimeError::InvalidObject(value)),
            },
        }
    }
}

impl<'ast> Default for TreeWalker<'ast> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum Value<'ast> {
    Number(LoxNumber),
    String(String),
    Bool(bool),
    Callable(Function<'ast>),
    Class(Class<'ast>),
    Instance(Class<'ast>),
    Nil,
    Undefined,
}

impl<'ast> Value<'ast> {
    fn eq(&self, right: &Value<'ast>) -> bool {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => a == b,

            (String(a), String(b)) => a == b,

            (Bool(a), Bool(b)) => a == b,

            (Nil, Nil) => true,

            _ => false,
        }
    }

    fn neq(&self, right: &Value<'ast>) -> bool {
        use Value::*;
        match (self, right) {
            (Number(a), Number(b)) => a != b,

            (String(a), String(b)) => a != b,

            (Bool(a), Bool(b)) => a != b,

            (Nil, Nil) => false,

            _ => true,
        }
    }

    fn gt(&self, right: &Value<'ast>) -> Result<bool, RuntimeError<'ast>> {
        use Value::*;
        let value = match (self, right) {
            (Number(a), Number(b)) => a > b,
            (Number(a), Bool(b)) => *a > LoxNumber::from(*b),

            (String(a), String(b)) => a > b,

            (Bool(a), Bool(b)) => a > b,
            (Bool(a), Number(b)) => *b < LoxNumber::from(*a),

            (left, right) => {
                return Err(RuntimeError::InvalidOperands(
                    left.clone(),
                    right.clone(),
                    BinaryOp::GreaterThan,
                ))
            }
        };

        Ok(value)
    }

    fn gte(&self, right: &Value<'ast>) -> Result<bool, RuntimeError<'ast>> {
        use Value::*;
        let value = match (self, right) {
            (Number(a), Number(b)) => a >= b,
            (Number(a), Bool(b)) => *a >= LoxNumber::from(*b),

            (String(a), String(b)) => a >= b,

            (Bool(a), Bool(b)) => a >= b,
            (Bool(a), Number(b)) => *b <= LoxNumber::from(*a),

            (left, right) => {
                return Err(RuntimeError::InvalidOperands(
                    left.clone(),
                    right.clone(),
                    BinaryOp::GreaterThanEqual,
                ))
            }
        };

        Ok(value)
    }

    fn lt(&self, right: &Value<'ast>) -> Result<bool, RuntimeError<'ast>> {
        use Value::*;
        let value = match (self, right) {
            (Number(a), Number(b)) => a < b,
            (Number(a), Bool(b)) => *a < LoxNumber::from(*b),

            (String(a), String(b)) => a < b,

            (Bool(a), Bool(b)) => a < b,
            (Bool(a), Number(b)) => *b < LoxNumber::from(*a),

            (left, right) => {
                return Err(RuntimeError::InvalidOperands(
                    left.clone(),
                    right.clone(),
                    BinaryOp::LessThan,
                ))
            }
        };

        Ok(value)
    }

    fn lte(&self, right: &Value<'ast>) -> Result<bool, RuntimeError<'ast>> {
        use Value::*;
        let value = match (self, right) {
            (Number(a), Number(b)) => a <= b,
            (Number(a), Bool(b)) => *a <= LoxNumber::from(*b),

            (String(a), String(b)) => a <= b,

            (Bool(a), Bool(b)) => a <= b,
            (Bool(a), Number(b)) => *b <= LoxNumber::from(*a),

            (left, right) => {
                return Err(RuntimeError::InvalidOperands(
                    left.clone(),
                    right.clone(),
                    BinaryOp::LessThanEqual,
                ))
            }
        };

        Ok(value)
    }

    fn add(mut self, right: Value<'ast>) -> Evaluation<'ast> {
        self.add_assign(right)?;
        Ok(self)
    }

    fn add_assign(&mut self, right: Value<'ast>) -> Result<(), RuntimeError<'ast>> {
        use Value::*;
        match (&mut *self, right) {
            (Number(a), Number(b)) => *a += b,
            (Number(a), Bool(b)) => *a += LoxNumber::from(b),
            (Bool(a), Number(b)) => *self = Number(LoxNumber::from(*a) + b),
            (Bool(a), Bool(b)) => *self = Number(LoxNumber::from(*a) + LoxNumber::from(b)),

            (String(a), String(b)) => *a += &b,
            (String(a), Number(b)) => *a += &b.to_string(),
            (String(a), Bool(b)) => *a += if b { "true" } else { "false" },
            (Number(a), String(b)) => *self = String(a.to_string() + &b),
            (Bool(a), String(b)) => *self = String(a.to_string() + &b),

            (a, b) => {
                return Err(RuntimeError::InvalidOperands(
                    a.clone(),
                    b.clone(),
                    BinaryOp::Add,
                ))
            }
        };
        Ok(())
    }

    fn sub(mut self, right: Value<'ast>) -> Evaluation<'ast> {
        self.sub_assign(right)?;
        Ok(self)
    }

    fn sub_assign(&mut self, right: Value<'ast>) -> Result<(), RuntimeError<'ast>> {
        use Value::*;
        match (&mut *self, right) {
            (Number(a), Number(b)) => *a -= b,
            (Number(a), Bool(b)) => *a -= LoxNumber::from(b),

            (Bool(a), Number(b)) => *self = Number(LoxNumber::from(*a) - b),
            (Bool(a), Bool(b)) => *self = Number(LoxNumber::from(*a) - LoxNumber::from(b)),

            (a, b) => {
                return Err(RuntimeError::InvalidOperands(
                    a.clone(),
                    b.clone(),
                    BinaryOp::Sub,
                ))
            }
        };
        Ok(())
    }

    fn mul(mut self, right: Value<'ast>) -> Evaluation<'ast> {
        self.mul_assign(right)?;
        Ok(self)
    }

    fn mul_assign(&mut self, right: Value<'ast>) -> Result<(), RuntimeError<'ast>> {
        use Value::*;
        match (&mut *self, right) {
            (Number(a), Number(b)) => *a *= b,
            (Number(a), Bool(b)) => *a *= LoxNumber::from(b),

            (Bool(a), Number(b)) => *self = Number(LoxNumber::from(*a) * b),
            (Bool(a), Bool(b)) => *self = Number(LoxNumber::from(*a) * LoxNumber::from(b)),

            (String(a), Number(b)) => *self = String(a.repeat(b as usize)),
            (Number(a), String(b)) => *self = String(b.repeat(*a as usize)),

            (a, b) => {
                return Err(RuntimeError::InvalidOperands(
                    a.clone(),
                    b.clone(),
                    BinaryOp::Mul,
                ))
            }
        };
        Ok(())
    }

    fn div(mut self, right: Value<'ast>) -> Evaluation<'ast> {
        self.div_assign(right)?;
        Ok(self)
    }

    fn div_assign(&mut self, right: Value<'ast>) -> Result<(), RuntimeError<'ast>> {
        use Value::*;
        match (&mut *self, right) {
            (Number(a), Number(b)) => *a /= b,
            (Number(a), Bool(b)) => *a /= LoxNumber::from(b),

            (Bool(a), Number(b)) => *self = Number(LoxNumber::from(*a) / b),
            (Bool(a), Bool(b)) => *self = Number(LoxNumber::from(*a) / LoxNumber::from(b)),

            (a, b) => {
                return Err(RuntimeError::InvalidOperands(
                    a.clone(),
                    b.clone(),
                    BinaryOp::Div,
                ))
            }
        };
        Ok(())
    }

    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Nil => false,
            _ => true,
        }
    }
}

impl<'ast> Display for Value<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Callable(c) => write!(f, "function \"{}\"", c.identifier.unwrap_or("<anon>")),
            Value::Class(c) => write!(f, "class \"{}\"", c.identifier),
            Value::Instance(o) => write!(f, "object {}", o.identifier),
            Value::Nil => write!(f, "nil"),
            Value::Undefined => write!(f, "undefined"),
        }
    }
}

#[allow(dead_code)]
enum ValueHandle<'ast> {
    Ref(&'ast str, EnvHandle),
    Own(Value<'ast>),
}

#[derive(Debug, Clone)]
pub struct Function<'ast> {
    identifier: Option<&'ast str>,
    parameters: Vec<&'ast str>,
    body: Block<'ast>,
    parent_env: usize,
}

#[derive(Debug, Clone)]
pub struct Class<'ast> {
    identifier: &'ast str,
    scope: Scope<'ast>,
    #[allow(dead_code)]
    superclass: Option<Box<Class<'ast>>>,
}

/// Maps identifiers to `Value`s
type Scope<'ast> = fnv::FnvHashMap<&'ast str, Value<'ast>>;

/// A Lox "environment", containing a list of scopes and a handle to the parent it "inherits" (closes) from.
#[derive(Debug, Clone)]
struct Env<'ast> {
    /// The FILO stack of scopes that gets pushed/popped by blocks and functions.
    /// There is always at least one scope and the active scope is always the last one.
    scopes: Vec<Scope<'ast>>,

    /// The parent environment, if any.
    /// Optional because the global environment has no parent.
    parent: Option<EnvHandle>,

    /// Whether this environment is a parent to another environment.
    /// This is an optimization to avoid walking the entire tree when deciding weather an environment has SCCs.
    is_parent: bool,

    /// Whether this environment is part of the "lexical" global environment.
    is_global: bool,
}

impl<'ast> Env<'ast> {
    /// Creates a new environment with a single scope as a child of the given parent.
    fn with_parent(parent: usize) -> Self {
        Self {
            scopes: vec![Scope::with_hasher(Default::default())],
            parent: Some(parent),
            is_parent: false,
            is_global: false,
        }
    }

    fn find_symbol(&mut self, identifier: &'ast str) -> Option<&mut Value<'ast>> {
        self.scopes
            .iter_mut()
            .rev()
            .find_map(|scope| scope.get_mut(identifier))
    }

    fn current_scope(&mut self) -> &mut Scope<'ast> {
        // SAFETY: We always have at least one scope.
        unsafe { self.scopes.last_mut().unwrap_unchecked() }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::with_hasher(Default::default()));
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn is_empty(&self) -> bool {
        self.scopes.iter().all(Scope::is_empty)
    }
}

/// Manages the "tree" of environments and provides methods to interact with them.
/// This is a tree-like structure but is implemented as a flat list where parent nodes are an index into said list instead of a direct pointer/reference,
/// which makes the borrow checker happy.
#[derive(Debug, Clone)]
struct EnvManager<'ast> {
    list: Vec<Env<'ast>>,
    active: EnvHandle,
}

type EnvHandle = usize;

impl<'ast> EnvManager<'ast> {
    /// Creates a new environment manager with a single global environment.
    fn new() -> Self {
        let global = Env {
            scopes: vec![Scope::with_hasher(Default::default())],
            parent: None,
            is_parent: false,
            is_global: true,
        };

        Self {
            list: vec![global],
            active: 0,
        }
    }

    #[inline]
    fn find_symbol(&mut self, identifier: &'ast str) -> Option<&mut Value<'ast>> {
        self.find_symbol_in(self.active, identifier)
    }

    #[inline]
    fn declare_symbol(
        &mut self,
        identifier: &'ast str,
        value: Value<'ast>,
    ) -> Result<(), RuntimeError<'ast>> {
        let allow_overriding = self.active_env().is_global && self.active_env().scopes.len() == 1;
        match self.active_env().current_scope().entry(identifier) {
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
            Entry::Occupied(mut entry) => {
                if allow_overriding {
                    entry.insert(value);
                    Ok(())
                } else {
                    Err(RuntimeError::NameCollision(identifier))
                }
            }
        }
    }

    #[inline]
    fn snapshot(&mut self) -> EnvHandle {
        if self.active_env().is_empty() {
            return self.active;
        }

        let new_env = self.push(self.active);
        self.get(new_env).is_global = self.get(self.active).is_global;
        self.switch(new_env)
    }

    #[inline]
    fn push(&mut self, parent: EnvHandle) -> EnvHandle {
        self.list.push(Env::with_parent(parent));
        self.list[parent].is_parent = true;
        self.list.len() - 1
    }

    #[inline]
    fn remove(&mut self, env: EnvHandle) {
        self.list.remove(env);
    }

    #[inline]
    fn switch(&mut self, env: EnvHandle) -> EnvHandle {
        replace(&mut self.active, env)
    }

    #[inline]
    fn push_scope(&mut self) {
        self.active_env().push_scope();
    }

    #[inline]
    fn pop_scope(&mut self) {
        self.active_env().pop_scope();
    }

    /// Walks backwards from the given environment to it's parents until it finds the first instance of the given identifier.
    fn find_symbol_in(
        &mut self,
        env: EnvHandle,
        identifier: &'ast str,
    ) -> Option<&mut Value<'ast>> {
        let mut env = self.get(env);
        loop {
            if let Some(value) = env.find_symbol(identifier) {
                // SAFETY:
                // The output lifetime is bound to `self`, so this doesn't extend the lifetime.
                // FIXME: remove once polonius (or another new borrow checker) sees that it's fine.
                return unsafe {
                    Some(std::mem::transmute::<&mut Value<'ast>, &mut Value<'ast>>(
                        value,
                    ))
                };
            } else if let Some(parent) = env.parent {
                env = self.get(parent);
            } else {
                return None;
            }
        }
    }

    #[inline]
    fn active_env(&mut self) -> &mut Env<'ast> {
        self.get(self.active)
    }

    #[inline]
    fn get(&mut self, env: EnvHandle) -> &mut Env<'ast> {
        &mut self.list[env]
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError<'ast> {
    #[error("Invalid operand \"{0}\" for unary operator \"{1:?}\".")]
    InvalidOperand(Value<'ast>, UnaryOp),

    #[error("Invalid operands \"{0}\" and \"{1}\" for binary operator \"{2:?}\".")]
    InvalidOperands(Value<'ast>, Value<'ast>, BinaryOp),

    #[error("Value \"{0}\" is not callable.")]
    InvalidCallee(Value<'ast>),

    #[error("Value \"{0}\" is not a class object.")]
    InvalidObject(Value<'ast>),

    #[error("Undefined property \"{0}\".")]
    UndefinedProperty(&'ast str),

    #[error("Undefined variable \"{0}\".")]
    UndefinedVariable(&'ast str),

    #[error("Symbol named \"{0}\" already exists in the current scope.")]
    NameCollision(&'ast str),

    #[error("Mismatched argument count, expected ({expected}) but got ({got})")]
    MismatchedArity { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub enum ControlFlow<'ast> {
    Break,
    Continue,
    Return(Value<'ast>),
}

impl<'ast> Display for ControlFlow<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Break => write!(f, "break"),
            Self::Continue => write!(f, "continue"),
            Self::Return(v) => write!(f, "{}", v),
        }
    }
}

/// The result of executing a statement.
#[derive(Debug, Clone)]
pub enum Stated<'ast> {
    Nothing,
    #[allow(dead_code)]
    Value(Value<'ast>),
    ControlFlow(ControlFlow<'ast>),
}

impl<'ast> Display for Stated<'ast> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nothing => Ok(()),
            Self::Value(v) => write!(f, "{}", v),
            Self::ControlFlow(c) => write!(f, "{}", c),
        }
    }
}

type Execution<'ast> = Result<Stated<'ast>, RuntimeError<'ast>>;
type Evaluation<'ast> = Result<Value<'ast>, RuntimeError<'ast>>;
