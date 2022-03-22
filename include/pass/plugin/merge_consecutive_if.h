#include <hash.h>
#include <mutator.h>

namespace ir {

class MergeConsecutiveIf : public Mutator {
    Stmt visit(const If &orig_op) {
        auto unchecked_op = Mutator::visit(orig_op);
        ASSERT(unchecked_op->nodeType() == ASTNodeType::If);
        auto op = unchecked_op.as<IfNode>();

        if (op->thenCase_->nodeType() == ASTNodeType::If) {
            auto inner = op->thenCase_.as<IfNode>();
            if (HashComparator()(inner->elseCase_, op->elseCase_))
                return makeIf("mergeif." + op->id().strId() + "." +
                                  op->thenCase_->id().strId(),
                              makeLAnd(op->cond_, inner->cond_),
                              inner->thenCase_, op->elseCase_);
        }

        return op;
    }
};

Stmt mergeConsecutiveIf(const Stmt &op) { return MergeConsecutiveIf()(op); }

DEFINE_PASS_FOR_FUNC(mergeConsecutiveIf)

} // namespace ir