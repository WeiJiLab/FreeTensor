#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <stmt.h>

namespace ir {

class ReverseForLoop : public Mutator {
    const ID &id_;

  public:
    ReverseForLoop(const ID &id) : id_(id) {}

    using Mutator::visit;
    Stmt visit(const For &op) override {
        if (op->id() != id_)
            return Mutator::visit(op);

        ASSERT(op->step_->nodeType() == ASTNodeType::IntConst &&
               op->step_.as<IntConstNode>()->val_ == -1);

        return makeFor(op->id(), op->iter_, makeAdd(op->end_, makeIntConst(1)),
                       makeAdd(op->begin_, makeIntConst(1)), makeIntConst(1),
                       op->len_, op->property_, (*this)(op->body_));
    }
};

inline Stmt reverseForLoop(const Stmt &op, const ID &id) {
    return ReverseForLoop(id)(op);
}

} // namespace ir
