#include <schedule/fission.h>

namespace ir {

Stmt HoistVar::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            for (auto &&def : defStack_) {
                xLoops_[def->name_].emplace_back(op->id());
            }
        }
        return ret;
    } else {
        inside_ = true, isAfter_ = false;
        auto ret = Mutator::visit(op);
        inside_ = false;
        for (auto &&def : defStack_) {
            xLoops_[def->name_].emplace_back(op->id());
        }
        for (auto i = defStack_.rbegin(); i != defStack_.rend(); i++) {
            ret = makeVarDef((*i)->id(), std::move((*i)->name_),
                             std::move(*((*i)->buffer_)), ret);
        }
        return ret;
    }
}

Stmt HoistVar::visit(const StmtSeq &op) {
    if (!inside_) {
        return Mutator::visit(op);
    } else {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&_stmt : op->stmts_) {
            auto stmt = (*this)(_stmt);
            stmts.emplace_back(stmt);
            isAfter_ |= stmt->id() == after_;
        }
        return makeStmtSeq(std::move(stmts));
    }
}

Stmt HoistVar::visit(const VarDef &_op) {
    if (!inside_) {
        return Mutator::visit(_op);
    } else {
        part0Vars_.erase(_op->name_);
        part1Vars_.erase(_op->name_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (part0Vars_.count(op->name_) && part1Vars_.count(op->name_)) {
            defStack_.emplace_back(op);
            return op->body_;
        } else {
            return op;
        }
    }
}

Stmt HoistVar::visit(const Store &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Expr HoistVar::visit(const Load &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Stmt HoistVar::visit(const AddTo &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Stmt AddDimToVar::visit(const For &op) {
    forMap_[op->id()] = op;
    return Mutator::visit(op);
}

Stmt AddDimToVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (toAdd_.count(op->name_)) {
        auto &&loops = toAdd_.at(op->name_);
        auto shape = op->buffer_->tensor().shape();
        for (auto it = loops.rbegin(); it != loops.rend(); it++) {
            auto len = makeSub(forMap_.at(*it)->end_, forMap_.at(*it)->begin_);
            shape.emplace_back(len);
        }
        op->buffer_->tensor().setShape(std::move(shape));
    }
    return op;
}

Stmt AddDimToVar::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    return doAdd(__op.as<StoreNode>());
}

Expr AddDimToVar::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    return doAdd(__op.as<LoadNode>());
}

Stmt AddDimToVar::visit(const AddTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::AddTo);
    return doAdd(__op.as<AddToNode>());
}

Stmt FissionFor::visit(const For &op) {
    if (op->id() != loop_) {
        return Mutator::visit(op);
    } else {
        auto begin = (*this)(op->begin_);
        auto end = (*this)(op->end_);
        inside_ = true;
        isPart0_ = true, inPart_ = true;
        auto part0 = (*this)(op->body_);
        isPart0_ = false, inPart_ = false;
        auto part1 = (*this)(op->body_);
        inside_ = false;
        auto for0 = makeFor(id0_, op->iter_, begin, end, part0);
        auto for1 = makeFor(id1_, op->iter_, begin, end, part1);
        return makeStmtSeq({for0, for1});
    }
}

Stmt FissionFor::visit(const StmtSeq &op) {
    if (!inside_) {
        return Mutator::visit(op);
    } else {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&_stmt : op->stmts_) {
            bool beforeInPart = inPart_;
            auto stmt = (*this)(_stmt);
            bool afterInPart = inPart_;
            if (beforeInPart || afterInPart) {
                stmts.emplace_back(stmt);
            }
            if (_stmt->id() == after_) {
                inPart_ = !isPart0_;
            }
        }
        if (stmts.size() == 1) {
            return stmts[0];
        } else {
            return makeStmtSeq(std::move(stmts));
        }
    }
}

Stmt FissionFor::visit(const VarDef &_op) {
    if (!inside_) {
        return Mutator::visit(_op);
    } else {
        varUses_.erase(_op->name_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (varUses_.count(op->name_)) {
            return op;
        } else {
            return op->body_;
        }
    }
}

Stmt FissionFor::visit(const Store &op) {
    if (inPart_) {
        varUses_.insert(op->var_);
    }
    return Mutator::visit(op);
}

Expr FissionFor::visit(const Load &op) {
    if (inPart_) {
        varUses_.insert(op->var_);
    }
    return Mutator::visit(op);
}

Stmt FissionFor::visit(const AddTo &op) {
    if (inPart_) {
        varUses_.insert(op->var_);
    }
    return Mutator::visit(op);
}

} // namespace ir

