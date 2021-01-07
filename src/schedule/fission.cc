#include <schedule/fission.h>

namespace ir {

Stmt HoistVar::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            for (auto &&def : defStack_) {
                xLoops_[def->name_].emplace_back(op->id());
            }
            innerLoops_.emplace_back(op->id());
        } else {
            outerScopes_.emplace_back(op->id());
        }
        return ret;
    } else {
        inside_ = true, isAfter_ = false;
        auto ret = Mutator::visit(op);
        inside_ = false;
        for (auto &&def : defStack_) {
            xLoops_[def->name_].emplace_back(op->id());
        }
        innerLoops_.emplace_back(op->id());
        for (auto i = defStack_.rbegin(); i != defStack_.rend(); i++) {
            ret = makeVarDef((*i)->id(), std::move((*i)->name_),
                             std::move(*((*i)->buffer_)), ret);
        }
        return ret;
    }
}

Stmt HoistVar::visit(const StmtSeq &op) {
    if (!inside_) {
        outerScopes_.emplace_back(op->id());
        return Mutator::visit(op);
    } else {
        std::vector<Stmt> before, after;
        for (auto &&_stmt : op->stmts_) {
            auto stmt = (*this)(_stmt);
            (isAfter_ ? after : before).emplace_back(stmt);
            isAfter_ |= stmt->id() == after_;
        }
        Stmt ret;
        if (after.empty()) {
            ret = makeStmtSeq(op->id(), std::move(before));
        } else if (before.empty()) {
            ret = makeStmtSeq(op->id(), std::move(after));
        } else {
            auto beforeNode = before.size() > 1
                                  ? makeStmtSeq("", std::move(before))
                                  : before[0];
            auto afterNode =
                after.size() > 1 ? makeStmtSeq("", std::move(after)) : after[0];
            ret = makeStmtSeq(op->id(), {beforeNode, afterNode});
        }
        if (isAfter_ && seqId_.empty()) {
            seqId_ = ret->id();
        }
        return ret;
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

void FissionFor::markNewId(const Stmt &op, bool isPart0) {
    std::string oldId = op->id(), newId;
    if (isPart0) {
        op->setId(newId = oldId + ".a");
        ids0_.emplace(oldId, newId);
    } else {
        op->setId(newId = oldId + ".b");
        ids1_.emplace(oldId, newId);
    }
}

Stmt FissionFor::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            markNewId(ret, isPart0_);
        }
        return ret;
    } else {
        auto begin = (*this)(op->begin_);
        auto end = (*this)(op->end_);
        inside_ = true;
        isPart0_ = true, inPart_ = true;
        auto part0 = (*this)(op->body_);
        isPart0_ = false, inPart_ = false;
        auto part1 = (*this)(op->body_);
        inside_ = false;
        auto for0 = makeFor(op->id(), op->iter_, begin, end, part0);
        auto for1 = makeFor(op->id(), op->iter_, begin, end, part1);
        markNewId(for0, true);
        markNewId(for1, false);
        return makeStmtSeq("", {for0, for1});
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
            return stmts[0]; // id already modified
        } else {
            Stmt ret = makeStmtSeq(op->id(), std::move(stmts));
            markNewId(ret, isPart0_);
            return ret;
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
        Stmt ret = varUses_.count(op->name_) ? __op : op->body_;
        markNewId(ret, isPart0_);
        return ret;
    }
}

Stmt FissionFor::visit(const Store &op) {
    if (inPart_) {
        varUses_.insert(op->var_);
    }
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
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
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const If &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const Assert &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

} // namespace ir
