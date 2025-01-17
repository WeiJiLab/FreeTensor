#ifndef FREE_TENSOR_GPU_LOWER_VECTOR_H
#define FREE_TENSOR_GPU_LOWER_VECTOR_H

#include <unordered_map>

#include <analyze/analyze_linear.h>
#include <func.h>
#include <pass/z3_simplify.h>

namespace freetensor {

namespace gpu {

class LowerVector : public Z3SimplifyWithSymbolTable {
    typedef Z3Simplify BaseClass;

    static constexpr int VEC_LEN[] = {4, 2};

    Var var_;
    Expr begin_;
    int vecLen_, isIndex_ = 0;
    bool simplifyOnly_ = false;

    AnalyzeLinear analyzeLinear_;

  private:
    std::string vecType(DataType dtype) const;
    bool hasVectorIndex(const Expr &index);
    Expr getIndex(const Expr &index);

  protected:
    using BaseClass::visit;

    Stmt visit(const For &op) override;
    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

Stmt lowerVector(const Stmt &op);

DEFINE_PASS_FOR_FUNC(lowerVector)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_LOWER_VECTOR_H
