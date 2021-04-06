#include <algorithm>
#include <climits>

#include <pass/make_const_shape.h>

namespace ir {

Stmt MakeConstShape::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (std::find(mtypes_.begin(), mtypes_.end(), op->buffer_->mtype()) ==
        mtypes_.end()) {
        return op;
    }

    size_t ndim = op->buffer_->tensor().shape().size();
    for (size_t i = 0; i < ndim; i++) {
        Expr &dim = op->buffer_->tensor().shape()[i];
        const Expr &oldDim = _op->buffer_->tensor().shape()[i];
        if (dim->nodeType() == ASTNodeType::IntConst) {
            continue;
        }
        int result = INT_MAX;
        if (upper_.count(oldDim)) {
            for (auto &&b : upper_.at(oldDim)) {
                if (b.expr_->nodeType() == ASTNodeType::IntConst) {
                    result = std::min(result, b.expr_.as<IntConstNode>()->val_);
                }
            }
        }
        if (result == INT_MAX) {
            throw InvalidProgram("Unable to relax dimension " +
                                 std::to_string(i) + ": " + toString(dim) +
                                 " of " + op->id() + ": " + op->name_ +
                                 " to a constant");
        }
        dim = makeIntConst(result);
        op->pinned_ = true;
    }
    return op;
}

Stmt makeConstShape(const Stmt &_op, const std::vector<MemType> &mtypes) {
    Stmt op;
    SimplifyPass::LowerBoundsMap lower;
    SimplifyPass::UpperBoundsMap upper;
    std::tie(op, lower, upper) = simplifyAndGetBounds(_op);
    op = MakeConstShape(mtypes, upper)(op);
    return op;
}

} // namespace ir
