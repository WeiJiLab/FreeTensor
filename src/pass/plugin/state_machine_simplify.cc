#include <pass/plugin/state_machine_simplify.h>
#include <pass/scalar_prop_const.h>
#include <pass/z3_simplify.h>

#include <analyze/all_iters.h>
#include <analyze/all_reads.h>

namespace ir {

namespace {
class Z3Condition : public Z3Simplify {
  public:
    Z3Condition(const SymbolTableInterface &symbolTable)
        : Z3Simplify(symbolTable) {}
    Expr solve(const Expr &expr) {
        auto all_iters = allIters(expr);
        int cond_cnt = 0;
        for (const auto &iter : all_iters) {
            auto loop = symbolTable_.loop(iter);
            if (loop->step_->nodeType() == ASTNodeType::IntConst &&
                loop->step_.as<IntConstNode>()->val_ == 1) {
                auto iter_var = makeVar(iter);
                push((*this)(makeGE(iter_var, loop->begin_)));
                push((*this)(makeLT(iter_var, loop->end_)));
                cond_cnt += 2;
            }
        }

        auto cond = (*this)(expr);
        auto notCond = (*this)(makeLNot(expr));
        std::optional<bool> result;
        if (prove(cond))
            result = true;
        else if (prove(notCond))
            result = false;

        for (int i = 0; i < cond_cnt; ++i)
            pop();

        if (result.has_value())
            return makeBoolConst(*result);
        else
            return cond;
    }
};

class Simplify : public ScalarPropConst {
    std::optional<Expr> obstacle_;
    ASTHashMap<Expr, bool> assumptions_;
    std::string postfix_;

  protected:
    Stmt visitStmt(const Stmt &stmt) override {
        auto ret = ScalarPropConst::visitStmt(stmt);
        ret->setId(ret->id().strId() + postfix_);
        return ret;
    }

    Stmt visit(const If &op) override {
        if (obstacle_)
            return ScalarPropConst::visit(op);

        auto cond = visitExpr(op->cond_);
        if (cond->nodeType() == ASTNodeType::BoolConst) {
            // constant branch, eliminate one
            if (cond.as<BoolConstNode>()->val_)
                return visitStmt(op->thenCase_);
            else
                return op->elseCase_.isValid() ? visitStmt(op->elseCase_)
                                               : makeStmtSeq("", {});
        }

        if (assumptions_.count(cond)) {
            if (assumptions_[cond])
                return visitStmt(op->thenCase_);
            else
                return op->elseCase_.isValid() ? visitStmt(op->elseCase_)
                                               : makeStmtSeq("", {});
        }

        auto z3_cond = Z3Condition(this->symbolTableSnapshot()).solve(cond);
        if (!z3_cond->isConst() && allReads(z3_cond).empty())
            obstacle_ = z3_cond;

        return ScalarPropConst::visit(
            COPY_DEBUG_INFO(
                makeIf(op->id(), z3_cond, op->thenCase_, op->elseCase_), op)
                .as<IfNode>());
    }

    Stmt visit(const For &op) override {
        std::function<Stmt(Stmt)> dfs = [&](Stmt body) {
            auto backup_state_begin = backup_state();
            body = (*this)(body);

            if (!obstacle_)
                return body;

            auto obstacle = *obstacle_;

            auto itersInObstacle = allIters(obstacle);
            if (!itersInObstacle.count(op->iter_))
                return body;

            obstacle_ = std::nullopt;
            auto backup_state_end = backup_state();

            postfix_ = ".assume_true";
            restore_state(backup_state_begin);
            assumptions_[obstacle] = true;
            auto thenCase = dfs(body);

            postfix_ = ".assume_false";
            restore_state(backup_state_begin);
            assumptions_[obstacle] = false;
            auto elseCase = dfs(body);

            assumptions_.erase(obstacle);
            restore_state(backup_state_end);

            return makeIf(ID(), obstacle, thenCase, elseCase);
        };
        pushFor(op);
        auto body = dfs(op->body_);
        popFor(op);
        return makeFor(op->id(), op->iter_, op->begin_, op->end_, op->step_,
                       op->len_, op->property_, body);
    }
};
} // namespace

Stmt stateMachineSimplify(const Stmt &op) { return Simplify()(op); }

} // namespace ir
