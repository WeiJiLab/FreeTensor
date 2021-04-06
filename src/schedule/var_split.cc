#include <schedule/var_split.h>

namespace ir {

Stmt VarSplit::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        found_ = true;

        if (dim_ >= (int)_op->buffer_->tensor().shape().size()) {
            throw InvalidSchedule("There is no dimension " +
                                  std::to_string(dim_) + " in variable " +
                                  _op->name_);
        }
        if (factor_ != -1) {
            dynFactor_ = makeIntConst(factor_);
        } else {
            ASSERT(nparts_ != -1);
            dynFactor_ = makeCeilDiv(_op->buffer_->tensor().shape()[dim_],
                                     makeIntConst(nparts_));
        }

        var_ = _op->name_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        var_.clear();

        if (fixedSize_) {
            if (!op->sizeLim_.isValid()) {
                Expr size;
                for (auto &&dim : op->buffer_->tensor().shape()) {
                    size = size.isValid() ? makeMul(size, dim) : dim;
                }
                op->sizeLim_ = size;
            }
        } else {
            if (op->buffer_->atype() != AccessType::Cache) {
                throw InvalidSchedule(
                    "Using RelaxedSize mode in an I/O variable is not allowed");
            }
        }

        auto &shape = op->buffer_->tensor().shape();
        if (factor_ != -1) {
            shape[dim_] = makeCeilDiv(shape[dim_], dynFactor_);
            shape.insert(shape.begin() + dim_ + 1, dynFactor_);
        } else {
            ASSERT(nparts_ != -1);
            shape[dim_] = dynFactor_;
            shape.insert(shape.begin() + dim_, makeIntConst(nparts_));
        }
        return op;
    } else {
        return Mutator::visit(_op);
    }
}

Stmt VarSplit::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return splitMemAcc(op);
}

Stmt VarSplit::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return splitMemAcc(op);
}

Expr VarSplit::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return splitMemAcc(op);
}

} // namespace ir
