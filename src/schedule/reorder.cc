#include <analyze/hash.h>
#include <except.h>
#include <schedule/reorder.h>

namespace ir {

bool MakeReduction::isSameElem(const Store &s, const Load &l) {
    if (s->var_ != l->var_) {
        return false;
    }
    ASSERT(s->indices_.size() == l->indices_.size());
    for (size_t i = 0, iEnd = s->indices_.size(); i < iEnd; i++) {
        if (getHash(s->indices_[i]) != getHash(l->indices_[i])) {
            return false;
        }
    }
    return true;
}

Stmt MakeReduction::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (op->expr_->nodeType() == ASTNodeType::Add) {
        auto expr = op->expr_.as<AddNode>();
        if (expr->lhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(op, expr->lhs_.as<LoadNode>())) {
            return makeAddTo(op->id(), op->var_, op->indices_, expr->rhs_);
        }
        if (expr->rhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(op, expr->rhs_.as<LoadNode>())) {
            return makeAddTo(op->id(), op->var_, op->indices_, expr->lhs_);
        }
    }
    return op;
}

Stmt SwapFor::visit(const For &_op) {
    if (_op->id() == oldOuter_->id()) {
        insideOuter_ = true;
        auto body = Mutator::visit(_op);
        insideOuter_ = false;
        return makeFor(oldInner_->id(), oldInner_->iter_, oldInner_->begin_,
                       oldInner_->end_, body);
    } else if (_op->id() == oldInner_->id()) {
        insideInner_ = true;
        auto __op = Mutator::visit(_op);
        insideInner_ = false;
        visitedInner_ = true;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        return op->body_;
    } else {
        return Mutator::visit(_op);
    }
}

Stmt SwapFor::visit(const StmtSeq &_op) {
    if (insideOuter_) {
        if (insideInner_) {
            return Mutator::visit(_op);
        }

        Stmt before, inner, after;
        std::vector<Stmt> beforeStmts, afterStmts;
        for (auto &&_stmt : _op->stmts_) {
            bool beforeInner = !visitedInner_;
            auto stmt = (*this)(_stmt);
            bool afterInner = visitedInner_;
            bool isInner = beforeInner && afterInner;
            if (isInner) {
                inner = stmt;
            } else if (beforeInner) {
                beforeStmts.emplace_back(stmt);
            } else {
                ASSERT(afterInner);
                afterStmts.emplace_back(stmt);
            }
        }

        if (!beforeStmts.empty()) {
            before =
                makeIf("", makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
                       beforeStmts.size() == 1 ? beforeStmts[0]
                                               : makeStmtSeq("", beforeStmts));
        }
        if (!afterStmts.empty()) {
            after =
                makeIf("", makeEQ(makeVar(oldInner_->iter_), oldInner_->begin_),
                       afterStmts.size() == 1 ? afterStmts[0]
                                              : makeStmtSeq("", afterStmts));
        }

        std::vector<Stmt> stmts;
        if (before.isValid()) {
            stmts.emplace_back(before);
        }
        if (inner.isValid()) {
            stmts.emplace_back(inner);
        }
        if (after.isValid()) {
            stmts.emplace_back(after);
        }
        return stmts.size() == 1 ? stmts[0] : makeStmtSeq(_op->id(), stmts);
    } else {
        return Mutator::visit(_op);
    }
}

} // namespace ir
