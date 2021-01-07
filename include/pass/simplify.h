#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <unordered_map>

#include <mutator.h>
#include <visitor.h>

namespace ir {

/**
 * Find all the variables in an expression, and determine the inner most scope
 * where these variables are defined
 */
class FindInnerMostScope : public Visitor {
    const std::unordered_map<std::string, int> &varScope_;
    int innerMost_ = 0;

  public:
    FindInnerMostScope(const std::unordered_map<std::string, int> &varScope)
        : varScope_(varScope) {}
    int innnerMost() const { return innerMost_; }

  protected:
    virtual void visit(const Var &op) override;
    virtual void visit(const Load &op) override;
};

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op);

class SimplifyPass : public Mutator {
  public:
    typedef std::unordered_map<const ExprNode *, std::vector<Expr>> BoundsMap;

  private:
    const std::unordered_map<const ExprNode *, uint64_t> &hash_;
    const BoundsMap &lower_, &upper_;
    bool isFixPoint_ = true;

    // defining scope table
    std::unordered_map<std::string, int> varScope_;
    int curScope_ = 0;

  public:
    SimplifyPass(const std::unordered_map<const ExprNode *, uint64_t> &hash,
                 const BoundsMap &lower, const BoundsMap &upper)
        : hash_(hash), lower_(lower), upper_(upper) {}

    bool isFixPoint() const { return isFixPoint_; }

  private:
    uint64_t getHash(const Expr &op);

    template <class T> Expr doSimplify(const T &_op) {
        auto op = Mutator::visit(_op);

        // To avoid divergence
        if (getHash(op) != getHash(_op)) {
            // E.g.
            // (1) a[0 - 0] -> a[0]
            // (2) (1 + 1) * a[0] -> 2 * a[0 - 0], because of the old bound
            return op;
        }

        Expr best = nullptr;
        auto bestScope = -1;
        // lower_ / upper_ for _op and op shall be the same, but those for op
        // are not updated, so using _op
        if (lower_.count(_op.get()) && upper_.count(_op.get())) {
            for (auto &&lower : lower_.at(_op.get())) {
                auto hl = getHash(lower);
                for (auto &&upper : upper_.at(_op.get())) {
                    auto hr = getHash(upper);
                    if (hl == hr) {
                        // We need to choose the simplest one. Other wise
                        // we are always picking the original expression
                        auto scope = findInnerMostScope(varScope_, lower);
                        if (!best.isValid() || scope < bestScope) {
                            best = lower, bestScope = scope;
                        }
                        break;
                    }
                }
            }
        }
        if (best.isValid() && getHash(best) != getHash(op)) {
            isFixPoint_ = false;
            return best;
        }
        return op;
    }

    template <class T, class Cmp> bool checkUpperCmp0(const T &op, Cmp &&cmp) {
        if (upper_.count(op->info_norm_form_.get())) {
            for (auto &&upper : upper_.at(op->info_norm_form_.get())) {
                if (upper->nodeType() == ASTNodeType::IntConst &&
                    cmp(upper.template as<IntConstNode>()->val_, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    template <class T, class Cmp> bool checkLowerCmp0(const T &op, Cmp &&cmp) {
        if (lower_.count(op->info_norm_form_.get())) {
            for (auto &&lower : lower_.at(op->info_norm_form_.get())) {
                if (lower->nodeType() == ASTNodeType::IntConst &&
                    cmp(lower.template as<IntConstNode>()->val_, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

  protected:
    Expr visit(const Var &op) override { return doSimplify(op); }
    Expr visit(const Add &op) override { return doSimplify(op); }
    Expr visit(const Sub &op) override { return doSimplify(op); }
    Expr visit(const Mul &op) override { return doSimplify(op); }
    Expr visit(const Div &op) override;
    Expr visit(const Mod &op) override;
    Expr visit(const Min &op) override;
    Expr visit(const Max &op) override;
    Expr visit(const LT &op) override;
    Expr visit(const LE &op) override;
    Expr visit(const GT &op) override;
    Expr visit(const GE &op) override;
    Expr visit(const EQ &op) override;
    Expr visit(const NE &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

Stmt simplifyPass(const Stmt &op);

// return {simplified, lower, upper}
std::tuple<Stmt, SimplifyPass::BoundsMap, SimplifyPass::BoundsMap>
simplifyAndGetBounds(const Stmt &op);

} // namespace ir

#endif // SIMPLIFY_H