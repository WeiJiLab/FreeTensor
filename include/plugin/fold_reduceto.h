#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <stmt.h>

namespace ir {

class FoldReduceTo : public Mutator {
    std::optional<Stmt> checkAndCombine(const Stmt &op1, const Stmt &op2) {
        // not same type node, return nullopt
        if (!((op1->nodeType() == ASTNodeType::ReduceTo ||
               op1->nodeType() == ASTNodeType::Store) &&
              op2->nodeType() == ASTNodeType::ReduceTo))
            return std::nullopt;

        if (op1->nodeType() == ASTNodeType::ReduceTo) {
            auto red1 = op1.as<ReduceToNode>(), red2 = op2.as<ReduceToNode>();

            auto red1_lhs = deepCopy(red1).as<ReduceToNode>(),
                 red2_lhs = deepCopy(red2).as<ReduceToNode>();
            red1_lhs->expr_ = red2_lhs->expr_ = makeIntConst(0);
            if (!HashComparator()(red1_lhs, red2_lhs))
                return std::nullopt;

            if (allReads(red2).count(red1->var_))
                return std::nullopt;

            Expr rhs;
            switch (red1->op_) {
            case ReduceOp::Add:
                rhs = makeAdd(red1->expr_, red2->expr_);
                break;
            case ReduceOp::Min:
                rhs = makeMin(red1->expr_, red2->expr_);
                break;
            case ReduceOp::Max:
                rhs = makeMax(red1->expr_, red2->expr_);
                break;
            case ReduceOp::Mul:
                rhs = makeMul(red1->expr_, red2->expr_);
                break;
            default:
                ASSERT(false);
                break;
            }

            return makeReduceTo(red1->id().strId(), red1->var_, red1->indices_,
                                red1->op_, rhs, red1->atomic_);
        } else {
            auto st1 = op1.as<StoreNode>();
            auto red2 = op2.as<ReduceToNode>();
            auto st1_lhs = deepCopy(st1).as<StoreNode>();
            st1_lhs->expr_ = makeIntConst(0);
            auto st2_lhs = makeStore(red2->id(), red2->var_, red2->indices_,
                                     makeIntConst(0));
            if (!HashComparator()(st1_lhs, st2_lhs))
                return std::nullopt;

            if (allReads(red2).count(st1->var_))
                return std::nullopt;

            Expr rhs;
            switch (red2->op_) {
            case ReduceOp::Add:
                rhs = makeAdd(st1->expr_, red2->expr_);
                break;
            case ReduceOp::Min:
                rhs = makeMin(st1->expr_, red2->expr_);
                break;
            case ReduceOp::Max:
                rhs = makeMax(st1->expr_, red2->expr_);
                break;
            case ReduceOp::Mul:
                rhs = makeMul(st1->expr_, red2->expr_);
                break;
            default:
                ASSERT(false);
                break;
            }

            return makeStore(st1->id(), st1->var_, st1->indices_, rhs);
        }
    }

  protected:
    Stmt visit(const StmtSeq &op_) override {
        auto unchecked_op = Mutator::visit(op_);
        ASSERT(unchecked_op->nodeType() == ASTNodeType::StmtSeq);
        auto op = unchecked_op.as<StmtSeqNode>();
        if (op->stmts_.empty())
            return op;
        Stmt last_stmt = op->stmts_[0];
        std::vector<Stmt> new_stmts;
        for (size_t i = 1; i < op->stmts_.size(); ++i)
            if (auto new_stmt = checkAndCombine(last_stmt, op->stmts_[i]))
                last_stmt = *new_stmt;
            else {
                new_stmts.push_back(last_stmt);
                last_stmt = op->stmts_[i];
            }
        new_stmts.push_back(last_stmt);
        return makeStmtSeq(op->id(), new_stmts);
    }
};

inline Stmt foldReduceTo(const Stmt &op) { return FoldReduceTo()(op); }

DEFINE_PASS_FOR_FUNC(foldReduceTo)

} // namespace ir
