#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <stmt.h>

namespace ir {

class FoldIfStmtToExpr : public Mutator {
    std::optional<Stmt> checkAndCombine(const Expr &cond, const Stmt &op1,
                                        const Stmt &op2) {
        // not same type node, return nullopt
        if (op1->nodeType() != op2->nodeType())
            return std::nullopt;

        auto comparator = HashComparator();

        // check For stmt
        if (op1->nodeType() == ASTNodeType::For) {
            // deepCopy for later modify
            auto op1_for = deepCopy(op1).as<ForNode>();
            auto op2_for = deepCopy(op2).as<ForNode>();
            // save previous body
            Stmt op1_body = op1_for->body_, op2_body = op2_for->body_;
            // set body to empty and compare the loop
            op1_for->body_ = op2_for->body_ = makeStmtSeq("", {});
            // if the loops are same, recursively check their body
            if (comparator(op1_for, op2_for))
                if (auto inner = checkAndCombine(cond, op1_body, op2_body)) {
                    // inner combined, wrap it with this level for loop
                    op1_for->body_ = *inner;
                    return op1_for;
                }
        }

        // check Store inner body
        if (op1->nodeType() == ASTNodeType::Store) {
            // deepCopy for later modify
            auto op1_store = deepCopy(op1).as<StoreNode>();
            auto op2_store = deepCopy(op2).as<StoreNode>();
            // save previous rhs
            Expr op1_rhs = op1_store->expr_, op2_rhs = op2_store->expr_;
            // set rhs to 0 and compare the statement
            op1_store->expr_ = op2_store->expr_ = makeIntConst(0);
            // if the stores are same, combine them and return
            if (comparator(op1_store, op2_store)) {
                op1_store->expr_ = makeIfExpr(cond, op1_rhs, op2_rhs);
                return op1_store;
            }
        }

        // not match, return nullopt
        return std::nullopt;
    }

  protected:
    Stmt visit(const If &op_) override {
        auto unchecked_op = Mutator::visit(op_);
        ASSERT(unchecked_op->nodeType() == ASTNodeType::If);
        auto op = unchecked_op.as<IfNode>();
        return checkAndCombine(op->cond_, op->thenCase_, op->elseCase_)
            .value_or(op);
    }
};

Stmt foldIfStmtToExpr(const Stmt &op) { return FoldIfStmtToExpr()(op); }

DEFINE_PASS_FOR_FUNC(foldIfStmtToExpr)

} // namespace ir
