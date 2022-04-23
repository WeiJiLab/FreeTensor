#include <hash.h>
#include <mutator.h>

#include <pass/shrink_for.h>

namespace ir {

class LinearTransformLoops : public Mutator {
    std::vector<ID> targets_;
    std::optional<std::unordered_map<std::string, Expr>> indices_;
    std::function<std::vector<Expr>(std::vector<Var>)> transform_;

  public:
    LinearTransformLoops(
        const std::vector<ID> &targets,
        const std::function<std::vector<Expr>(std::vector<Var>)> &transform)
        : targets_(targets), indices_(), transform_(transform) {
        if (targets.size() == 0)
            throw InvalidSchedule("Targets loop must be non-empty.");
    }

  protected:
    Expr visit(const Var &op) {
        if (indices_ && indices_->count(op->name_))
            return (*indices_)[op->name_];
        return op;
    }

    Stmt visit(const For &op) {
        if (op->id() == targets_[0]) {
            if (indices_)
                throw InvalidProgram("Non-unique ID provided.");

            std::vector<Var> names;
            std::vector<For> old_loops;
            std::vector<For> new_loops;

            auto do_loop = [&](For walk) {
                old_loops.push_back(walk);
                names.push_back(makeVar(walk->iter_).as<VarNode>());
                if (walk->step_->nodeType() != ASTNodeType::IntConst ||
                    walk->step_.as<IntConstNode>()->val_ != 1)
                    throw InvalidSchedule(
                        "Unsupported step: " + toString(walk->step_) + " in " +
                        toString(walk));
                new_loops.push_back(
                    makeFor(walk->id(), walk->iter_, makeIntConst(-1000000),
                            makeIntConst(1000000), walk->step_,
                            makeIntConst(2000000),
                            walk->property_, makeStmtSeq(ID(), {}))
                        .as<ForNode>());
            };

            auto walk = op;
            do_loop(walk);
            for (size_t i = 1; i < targets_.size(); ++i) {
                if (walk->body_->nodeType() != ASTNodeType::For ||
                    walk->body_->id() != targets_[i])
                    throw InvalidSchedule(
                        "Directed nested loops required, but found " +
                        toString(op));
                walk = walk->body_.as<ForNode>();
                do_loop(walk);
            }
            Stmt inner_body = walk->body_;

            auto new_iter = transform_(names);
            auto inner_condition = makeBoolConst(true);
            for (size_t i = 0; i < targets_.size(); ++i)
                inner_condition =
                    makeLAnd(inner_condition,
                             makeLAnd(makeLE(old_loops[i]->begin_, new_iter[i]),
                                      makeLT(new_iter[i], old_loops[i]->end_)));

            indices_.emplace();
            for (size_t i = 0; i < targets_.size(); ++i)
                (*indices_)[names[i]->name_] = new_iter[i];

            inner_body = makeIf(ID(), inner_condition, (*this)(inner_body));

            for (size_t i = 0; i < targets_.size() - 1; ++i)
                new_loops[i]->body_ = new_loops[i + 1];
            new_loops[targets_.size() - 1]->body_ = inner_body;

            return new_loops[0];
        }
        return Mutator::visit(op);
    }
};

inline Stmt linearTransformLoops(
    const Stmt &op, const std::vector<ID> &targets,
    const std::function<std::vector<Expr>(std::vector<Var>)> &transform) {
    return shrinkFor(LinearTransformLoops(targets, transform)(op));
}

} // namespace ir