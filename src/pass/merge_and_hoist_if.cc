#include <algorithm>

#include <analyze/all_reads.h>
#include <analyze/all_writes.h>
#include <analyze/check_all_defined.h>
#include <analyze/hash.h>
#include <pass/merge_and_hoist_if.h>

namespace ir {

Stmt MergeAndHoistIf::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    for (auto &&stmt : op->stmts_) {
        if (!stmts.empty() && stmts.back()->nodeType() == ASTNodeType::If &&
            stmt->nodeType() == ASTNodeType::If) {
            auto if1 = stmts.back().as<IfNode>();
            auto if2 = stmt.as<IfNode>();
            if (getHash(if1->cond_) == getHash(if2->cond_)) {
                auto writes = allWrites(if1);
                auto reads = allReads(if2->cond_);
                if (std::none_of(reads.begin(), reads.end(),
                                 [&writes](const std::string &var) -> bool {
                                     return writes.count(var);
                                 })) {
                    auto thenCase =
                        makeStmtSeq("", {if1->thenCase_, if2->thenCase_});
                    Stmt elseCase;
                    if (if1->elseCase_.isValid() && if2->elseCase_.isValid()) {
                        elseCase =
                            makeStmtSeq("", {if1->elseCase_, if2->elseCase_});
                    } else if (if1->elseCase_.isValid() &&
                               !if2->elseCase_.isValid()) {
                        elseCase = if1->elseCase_;
                    } else if (!if1->elseCase_.isValid() &&
                               if2->elseCase_.isValid()) {
                        elseCase = if2->elseCase_;
                    }
                    stmts.pop_back();
                    stmts.emplace_back(makeIf("", if1->cond_,
                                              std::move(thenCase),
                                              std::move(elseCase)));
                    continue;
                }
            }
        }
        stmts.emplace_back(stmt);
    }
    op->stmts_ = std::vector<SubTree<StmtNode>>(stmts.begin(), stmts.end());
    if (op->stmts_.size() == 1) {
        return op->stmts_[0];
    }
    return op;
}

Stmt MergeAndHoistIf::visit(const VarDef &_op) {
    def_.insert(_op->name_);
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    def_.erase(_op->name_);

    if (op->body_->nodeType() == ASTNodeType::If) {
        auto branch = op->body_.as<IfNode>();
        if (!branch->elseCase_.isValid() &&
            checkAllDefined(def_, branch->cond_)) {
            return makeIf(branch->id(), branch->cond_,
                          makeVarDef(op->id(), op->name_,
                                     std::move(*op->buffer_), op->sizeLim_,
                                     branch->thenCase_, op->pinned_));
        }
    }
    return op;
}

Stmt MergeAndHoistIf::visit(const For &_op) {
    def_.insert(_op->iter_);
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    def_.erase(op->iter_);

    if (op->body_->nodeType() == ASTNodeType::If) {
        auto branch = op->body_.as<IfNode>();
        if (!branch->elseCase_.isValid() &&
            checkAllDefined(def_, branch->cond_)) {
            auto writes = allWrites(branch);
            auto reads = allReads(branch->cond_);
            if (std::none_of(reads.begin(), reads.end(),
                             [&writes](const std::string &var) -> bool {
                                 return writes.count(var);
                             })) {
                return makeIf(branch->id(), branch->cond_,
                              makeFor(op->id(), op->iter_, op->begin_, op->end_,
                                      op->len_, op->parallel_, op->unroll_,
                                      op->vectorize_, branch->thenCase_));
            }
        }
    }
    return op;
}

Stmt mergeAndHoistIf(const Stmt &op) { return MergeAndHoistIf()(op); }

} // namespace ir
