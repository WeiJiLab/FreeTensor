#include <plugin/state_machine_simplify.h>
#include <pass/scalar_prop_const.h>
#include <pass/z3_simplify.h>

#include <analyze/all_uses.h>

namespace ir {

namespace {
class Z3Condition : public Z3Simplify {
    const ASTHashMap<Expr, bool> &assumptions_;

  public:
    Z3Condition(const SymbolTableInterface &symbolTable,
                const ASTHashMap<Expr, bool> &assumptions)
        : Z3Simplify(symbolTable), assumptions_(assumptions) {}
    Expr solve(const Expr &expr) {
        auto all_iters = allIters(expr);
        int cond_cnt = 0;
        for (const auto &iter : all_iters) {
            auto loop = symbolTable_.loop(iter);
            if (loop->step_->nodeType() == ASTNodeType::IntConst) {
                auto iter_var = makeVar(iter);
                if (loop->step_.as<IntConstNode>()->val_ > 0) {
                    push((*this)(makeGE(iter_var, loop->begin_)));
                    push((*this)(makeLT(iter_var, loop->end_)));
                } else {
                    push((*this)(makeLE(iter_var, loop->begin_)));
                    push((*this)(makeGT(iter_var, loop->end_)));
                }
                cond_cnt += 2;
            }
        }
        for (const auto &[expr, val] : assumptions_) {
            if (val)
                push((*this)(expr));
            else
                push((*this)(makeLNot(expr)));
            cond_cnt++;
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

class AddIdPostFix : public Mutator {
    const std::string &postfix_;

  public:
    AddIdPostFix(const std::string &postfix) : postfix_(postfix) {}

  protected:
    Stmt visitStmt(const Stmt &stmt) override {
        auto ret = Mutator::visitStmt(stmt);
        ret->setId(ret->id().strId() + postfix_);
        return ret;
    }
};

class Simplify : public ScalarPropConst {
    std::optional<Expr> obstacle_;
    ASTHashMap<Expr, bool> assumptions_;

    Expr simplifyPredicate(const Expr &pred) {
        auto cond = visitExpr(pred);
        if (cond->isConst())
            return cond;

        auto z3_cond =
            Z3Condition(this->symbolTableSnapshot(), assumptions_).solve(cond);
        return z3_cond;
    }

  protected:
    Stmt visitStmt(const Stmt &stmt) override {
        auto ret =
            !obstacle_ ? ScalarPropConst::visitStmt(stmt) : deepCopy(stmt);
        return ret;
    }

    Stmt visit(const If &op) override {
        auto cond = simplifyPredicate(op->cond_);
        if (cond->isConst()) {
            // constant branch, eliminate one
            if (cond.as<BoolConstNode>()->val_)
                return visitStmt(op->thenCase_);
            else
                return op->elseCase_.isValid() ? visitStmt(op->elseCase_)
                                               : makeStmtSeq("", {});
        }

        if (!allReads(cond).empty())
            std::cerr << op << std::endl;

        ASSERT(allReads(cond).empty());
        obstacle_ = cond;
        return visitStmt(op);
    }

    Stmt visit(const For &op) override {
        // special case: unrolled while loop
        if (op->id().strId().find("unrolled-while") != std::string::npos) {
            ASSERT(op->len_->isConst() &&
                   op->len_.as<IntConstNode>()->val_ ==
                       std::numeric_limits<int32_t>::max());
            ASSERT(op->body_->nodeType() == ASTNodeType::If &&
                   !op->body_.as<IfNode>()->elseCase_.isValid());

            auto output = makeStmtSeq(op->id(), {}).as<StmtSeqNode>();
            const Expr &cond = op->body_.as<IfNode>()->cond_;
            const Stmt &body = op->body_.as<IfNode>()->thenCase_;
            auto step_cnt = 1;
            while (true) {
                auto _stepping = simplifyPredicate(cond);
                ASSERT(_stepping->isConst());
                auto stepping = _stepping.as<BoolConstNode>()->val_;

                if (!stepping)
                    break;

                if (obstacle_) {
                    output->stmts_.push_back(AddIdPostFix(
                    ".while-" + std::to_string(step_cnt++))(deepCopy(op)));
                    break;
                }

                output->stmts_.push_back(AddIdPostFix(
                    ".while-" + std::to_string(step_cnt++))((*this)(body)));
            }

            return output;
        }

        std::function<Stmt(Stmt)> dfs = [&](Stmt body) {
            auto backup_state_begin = backup_state();
            body = (*this)(body);

            if (!obstacle_)
                return body;

            auto obstacle = *obstacle_;

            // auto itersInObstacle = allIters(obstacle);
            // if (!itersInObstacle.count(op->iter_))
            //     return body;

            obstacle_ = std::nullopt;
            auto backup_state_end = backup_state();

            restore_state(backup_state_begin);
            assumptions_[obstacle] = true;
            auto thenCase = AddIdPostFix(".true")(dfs(body));

            restore_state(backup_state_begin);
            assumptions_[obstacle] = false;
            auto elseCase = AddIdPostFix(".false")(dfs(body));

            assumptions_.erase(obstacle);
            restore_state(backup_state_end);

            return makeIf(ID(), obstacle, thenCase, elseCase);
        };

        for (auto &wr : allWrites(op->body_))
            if (has_constant(wr))
                kill_constant(wr, std::nullopt);

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
