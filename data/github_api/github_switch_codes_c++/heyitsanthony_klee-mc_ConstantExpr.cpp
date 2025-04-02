#include <stdio.h>
#include "klee/Expr.h"

#include "ExprAlloc.h"

#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/CommandLine.h>
// FIXME: We shouldn't need this once fast constant support moves into
// Core. If we need to do arithmetic, we probably want to use APInt.
#include "klee/Internal/Support/IntEvaluation.h"

#include "klee/util/ExprPPrinter.h"

#include <iostream>
#include <sstream>

using namespace klee;

unsigned ConstantExpr::constexpr_count = 0;

ref<Expr> ConstantExpr::fromMemory(void *address, Width width) {
  switch (width) {
  default: assert(0 && "invalid type");
  case  Expr::Bool: return ConstantExpr::create(*(( uint8_t*) address), width);
  case  Expr::Int8: return ConstantExpr::create(*(( uint8_t*) address), width);
  case Expr::Int16: return ConstantExpr::create(*((uint16_t*) address), width);
  case Expr::Int32: return ConstantExpr::create(*((uint32_t*) address), width);
  case Expr::Int64: return ConstantExpr::create(*((uint64_t*) address), width);
  // FIXME: what about machines without x87 support?
  case Expr::Fl80:
    return ConstantExpr::alloc(llvm::APInt(width,
      (width+llvm::integerPartWidth-1)/llvm::integerPartWidth,
      (const uint64_t*)address));
  }
}

void ConstantExpr::toMemory(void *address) {
  switch (getWidth()) {
  default: assert(0 && "invalid type");
  case  Expr::Bool: *(( uint8_t*) address) = getZExtValue(1); break;
  case  Expr::Int8: *(( uint8_t*) address) = getZExtValue(8); break;
  case Expr::Int16: *((uint16_t*) address) = getZExtValue(16); break;
  case Expr::Int32: *((uint32_t*) address) = getZExtValue(32); break;
  case Expr::Int64: *((uint64_t*) address) = getZExtValue(64); break;
  // FIXME: what about machines without x87 support?
  case Expr::Fl80:
    *((long double*) address) = *(long double*) value.getRawData();
    break;
  }
}

ref<ConstantExpr> ConstantExpr::alloc(const llvm::APInt &v)
{ return cast<ConstantExpr>(theExprAllocator->Constant(v)); }

ref<ConstantExpr> ConstantExpr::alloc(uint64_t v, Width w)
{ return cast<ConstantExpr>(theExprAllocator->Constant(v, w)); }

ref<ConstantExpr> ConstantExpr::createVector(const llvm::ConstantVector* v)
{
	unsigned int	elem_count;

	elem_count = v->getNumOperands();

	ref<ConstantExpr>	cur_v;
	for (unsigned int i = 0; i < elem_count; i++) {
		llvm::Value		*op;
		llvm::APInt		api;

		op = v->getOperand(i);

		if (llvm::ConstantInt* ci = dyn_cast<llvm::ConstantInt>(op)) {
			api = ci->getValue();
		} else if (llvm::ConstantFP* cf=dyn_cast<llvm::ConstantFP>(op)){
			api = cf->getValueAPF().bitcastToAPInt();
		} else if (llvm::UndefValue* cu=dyn_cast<llvm::UndefValue>(op)){
			/* 0 will probably lead to the least confusion */
			/* 1 so that it can be used in shuffle vector */
			api = llvm::APInt(
				cu->getType()->getPrimitiveSizeInBits(),
				0x1);
	//		api = llvm::APInt::getAllOnesValue(
	//			cu->getType()->getPrimitiveSizeInBits());
		} else {
			std::cerr << "v: ";
			v->dump();
			std::cerr << "DUMPING OPERAND FOR V\n";
			op->dump();
			std::cerr << "TYPE::::::\n";
			op->getType()->dump();
			std::cerr << '\n';
			assert (0 == 1 && "Weird type??");
		}

		cur_v = ref<ConstantExpr>((i == 0)
			? alloc(api)
			: dyn_cast<ConstantExpr>(MK_CONCAT(alloc(api), cur_v)));
	}

	return cur_v;
}

ref<ConstantExpr> ConstantExpr::createSeqData(const llvm::ConstantDataSequential* v)
{
	unsigned		bytes_per_elem, elem_c;
	ref<ConstantExpr>	cur_v;
	const llvm::Type	*t;
	bool			is_int;

	bytes_per_elem = v->getElementByteSize();
	elem_c = v->getNumElements();
	t = v->getElementType();

	assert (bytes_per_elem*elem_c <= 64);

	is_int = t->isIntegerTy();
	assert (is_int || (t->isFloatTy() || t->isDoubleTy()));

	for (unsigned i = 0; i < elem_c; i++) {
		ref<ConstantExpr>	ce;

		if (is_int) {
			ce = MK_CONST(
				v->getElementAsInteger(i), bytes_per_elem*8);
		} else if (t->isFloatTy()) {
			float	f = v->getElementAsFloat(i);
			ce = MK_CONST(*((uint32_t*)&f), bytes_per_elem*8);
		} else if (t->isDoubleTy()) {
			double d = v->getElementAsDouble(i);
			ce = MK_CONST(*((uint64_t*)&d), bytes_per_elem*8);
		}

		if (i == 0) cur_v = ce;
#ifdef BROKEN_OSDI
		else cur_v = cur_v->Concat(ce);
#else
		else cur_v = ce->Concat(cur_v);
#endif
	}


	return cur_v;
}

ref<ConstantExpr> ConstantExpr::Concat(const ref<ConstantExpr> &RHS)
{
  Expr::Width W = getWidth() + RHS->getWidth();
  llvm::APInt Tmp(value);
  Tmp = Tmp.zext(W);
  Tmp <<= RHS->getWidth();
  Tmp |= llvm::APInt(RHS->value).zext(W);

  return ConstantExpr::alloc(Tmp);
}

ref<ConstantExpr> ConstantExpr::Extract(unsigned Offset, Width W) const
{ return ConstantExpr::alloc(llvm::APInt(value.ashr(Offset)).zextOrTrunc(W)); }

ref<ConstantExpr> ConstantExpr::ZExt(Width W)
{ return ConstantExpr::alloc(llvm::APInt(value).zextOrTrunc(W)); }

ref<ConstantExpr> ConstantExpr::SExt(Width W)
{ return ConstantExpr::alloc(llvm::APInt(value).sextOrTrunc(W)); }

#define DECL_CE_OP(FNAME, OP)	\
ref<ConstantExpr> ConstantExpr::FNAME(const ref<ConstantExpr> &in_rhs) \
{ return ConstantExpr::alloc(value OP in_rhs->value); }

#define DECL_CE_OP_CHK(FNAME, OP)	\
ref<ConstantExpr> ConstantExpr::FNAME(const ref<ConstantExpr> &in_rhs) \
{	llvm::APInt temp(value OP in_rhs->value);	\
	return (temp == value)				\
		? this					\
		: ConstantExpr::alloc(temp); }

#define DECL_CE_FUNCOP(FNAME, OP)	\
ref<ConstantExpr> ConstantExpr::FNAME(const ref<ConstantExpr> &in_rhs) \
{ return ConstantExpr::alloc(value.OP(in_rhs->value)); }

#define DECL_CE_NONZERO_FUNCOP(FNAME, OP)	\
ref<ConstantExpr> ConstantExpr::FNAME(const ref<ConstantExpr> &in_rhs) \
{ \
	if (in_rhs->isZero()) {	\
		if (!Expr::errors)	\
			std::cerr << "[Expr] 0 as RHS on restricted op\n"; \
		Expr::errors++;	\
		return in_rhs;	\
	} \
	return ConstantExpr::alloc(value.OP(in_rhs->value));	\
}


#define DECL_CE_CMPOP(FNAME, OP)	\
ref<ConstantExpr> ConstantExpr::FNAME(const ref<ConstantExpr> &in_rhs)	\
{ return ConstantExpr::alloc(value.OP(in_rhs->value), Expr::Bool); }

DECL_CE_OP(Add, +)
DECL_CE_OP(Sub, -)
DECL_CE_OP(Mul, *)
DECL_CE_OP_CHK(And, &)
DECL_CE_OP_CHK(Or, |)
DECL_CE_OP_CHK(Xor, ^)
DECL_CE_NONZERO_FUNCOP(UDiv, udiv)
DECL_CE_NONZERO_FUNCOP(SDiv, sdiv)
DECL_CE_NONZERO_FUNCOP(URem, urem)
DECL_CE_NONZERO_FUNCOP(SRem, srem)
DECL_CE_FUNCOP(Shl, shl)
DECL_CE_FUNCOP(LShr, lshr)

ref<ConstantExpr> ConstantExpr::AShr(const ref<ConstantExpr> &in_rhs)
{
	if (getWidth() == 1) return this;
	return ConstantExpr::alloc(value.ashr(in_rhs->value));
}

ref<ConstantExpr> ConstantExpr::Neg() { return ConstantExpr::alloc(-value); }
ref<ConstantExpr> ConstantExpr::Not() { return ConstantExpr::alloc(~value); }

ref<ConstantExpr> ConstantExpr::Eq(const ref<ConstantExpr> &RHS)
{ return ConstantExpr::alloc(value == RHS->value, Expr::Bool); }

ref<ConstantExpr> ConstantExpr::Ne(const ref<ConstantExpr> &RHS)
{ return ConstantExpr::alloc(value != RHS->value, Expr::Bool); }

DECL_CE_CMPOP(Ult, ult)
DECL_CE_CMPOP(Ule, ule)
DECL_CE_CMPOP(Ugt, ugt)
DECL_CE_CMPOP(Uge, uge)
DECL_CE_CMPOP(Slt, slt)
DECL_CE_CMPOP(Sle, sle)
DECL_CE_CMPOP(Sgt, sgt)
DECL_CE_CMPOP(Sge, sge)

#include <llvm/ADT/Hashing.h>

Expr::Hash ConstantExpr::computeHash(void)
{
	skeletonHash = getWidth() * MAGIC_HASH_CONSTANT;
	hashValue =
		//value.hash_value(value).size_t() 
		hash_value(value)
		^ (getWidth() * MAGIC_HASH_CONSTANT);
	return hashValue;
}
