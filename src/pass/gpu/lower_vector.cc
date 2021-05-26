#include <algorithm>

#include <analyze/hash.h>
#include <pass/gpu/lower_vector.h>
#include <pass/simplify.h>

namespace ir {

namespace gpu {

std::string LowerVector::vecType(DataType dtype) const {
    std::string ret;
    switch (dtype) {
    case DataType::Float32:
        ret = "float";
        break;
    case DataType::Int32:
        ret = "int";
        break;
    default:
        ERROR("Unsupported data type " + toString(dtype));
    }
    return ret + std::to_string(vecLen_);
}

bool LowerVector::hasVectorIndex(const Expr &index) {
    // Any expression is analyzed as a linear expression
    analyzeLinear_(index);
    auto &&lin = analyzeLinear_.result().at(index);

    auto it = std::find_if(lin.coeff_.begin(), lin.coeff_.end(),
                           [this](const std::pair<uint64_t, Scale<int>> &item) {
                               return item.first == varHash_;
                           });
    if (it != lin.coeff_.end()) {
        // TODO: k_ can be -1
        if (it->second.k_ != 1) {
            throw InvalidSchedule("Vectorized non-contiguous memory "
                                  "accessis not supported");
        }

        auto toProve = makeEQ(
            makeMod(makeSub(index, makeVar(var_)), makeIntConst(vecLen_)),
            makeIntConst(0));
        simplifyOnly_ = true;
        toProve = (*this)(toProve);
        simplifyOnly_ = false;
        if (!prove(toProve)) {
            throw InvalidSchedule("Vectorized memory access should be"
                                  "aligned to the vector length: " +
                                  toString(toProve) + " does not hold");
        }
        return true;
    } else {
        return false;
    }
}

Expr LowerVector::getIndex(const Expr &index) {
    Expr ret;
    isIndex_++;
    try {
        ret = (*this)(index);
    } catch (const InvalidSchedule &e) {
        isIndex_--;
        throw;
    }
    isIndex_--;
    return ret;
}

Stmt LowerVector::visit(const For &op) {
    if (op->vectorize_) {
        if (!var_.empty()) {
            throw InvalidSchedule("Nested vectorized loops is not supported");
        }

        varHash_ = getHash(makeVar(op->iter_));
        for (int vecLen : VEC_LEN) {
            var_ = op->iter_;
            vecLen_ = vecLen;
            For ret;
            try {
                auto toProve = makeEQ(makeMod(op->len_, makeIntConst(vecLen)),
                                      makeIntConst(0));
                simplifyOnly_ = true;
                toProve = (*this)(toProve);
                simplifyOnly_ = false;
                if (!prove(toProve)) {
                    throw InvalidSchedule("The loop length should be divisible "
                                          "by the vector length");
                }
                ret = deepCopy(op).as<ForNode>();
                ret->len_ = makeFloorDiv(ret->len_, makeIntConst(vecLen));
                ret->end_ = makeAdd(ret->begin_, ret->len_);
                ret->body_ = (*this)(ret->body_);
            } catch (const InvalidSchedule &e) {
                WARNING("Vectorizing loop " + op->id() + " to length " +
                        std::to_string(vecLen) +
                        " failed because: " + e.what());
                var_.clear();
                continue;
            }
            var_.clear();
            ret->vectorize_ = false; // done
            return ret;
        }
    }
    return Z3Simplify::visit(op);
}

Expr LowerVector::visit(const Var &op) {
    if (!simplifyOnly_ && var_ == op->name_) {
        auto ret = makeMul(op, makeIntConst(vecLen_));
        if (!isIndex_) {
            switch (vecLen_) {
            case 4:
                ret = makeIntrinsic("int4{%, %, %, %}",
                                    {ret, makeAdd(ret, makeIntConst(1)),
                                     makeAdd(ret, makeIntConst(2)),
                                     makeAdd(ret, makeIntConst(3))},
                                    DataType::Custom);
                break;
            case 2:
                ret = makeIntrinsic("int2{%, %}",
                                    {ret, makeAdd(ret, makeIntConst(1))},
                                    DataType::Custom);
                break;
            default:
                ASSERT(false);
            }
        }
        return ret;
    }
    return Z3Simplify::visit(op);
}

Expr LowerVector::visit(const Load &op) {
    if (!simplifyOnly_ && !var_.empty() && !op->indices_.empty()) {
        ASSERT(op->indices_.size() == 1); // Please do make_1d_var first
        if (hasVectorIndex(op->indices_[0])) {
            Expr index = getIndex(op->indices_[0]);
            auto dtype = buffers_.at(op->var_)->tensor().dtype();
            auto vtype = vecType(dtype);
            return makeIntrinsic("*((" + vtype + "*)&(%))",
                                 {makeLoad(op->var_, {index})},
                                 DataType::Custom);
        }
    }
    return Z3Simplify::visit(op);
}

Stmt LowerVector::visit(const Store &op) {
    if (!var_.empty() && !op->indices_.empty()) {
        ASSERT(op->indices_.size() == 1); // Please do make_1d_var first
        if (hasVectorIndex(op->indices_[0])) {
            Expr index = getIndex(op->indices_[0]);
            auto dtype = buffers_.at(op->var_)->tensor().dtype();
            auto vtype = vecType(dtype);
            return makeEval(
                "",
                makeIntrinsic("*((" + vtype + "*)&(%)) = make_" + vtype + "(%)",
                              {makeLoad(op->var_, {index}), (*this)(op->expr_)},
                              DataType::Void));
        }
    }
    return Z3Simplify::visit(op);
}

Stmt LowerVector::visit(const ReduceTo &op) {
    if (!var_.empty() && !op->indices_.empty()) {
        ASSERT(op->indices_.size() == 1); // Please do make_1d_var first
        if (hasVectorIndex(op->indices_[0])) {
            Expr index = getIndex(op->indices_[0]);
            auto dtype = buffers_.at(op->var_)->tensor().dtype();
            auto vtype = vecType(dtype);
            auto newLoad = makeLoad(op->var_, {index});
            switch (op->op_) {
            case ReduceOp::Add:
                return makeEval(
                    "", makeIntrinsic(
                            "*((" + vtype + "*)&(%)) += make_" + vtype + "(%)",
                            {newLoad, (*this)(op->expr_)}, DataType::Void));
            case ReduceOp::Max:
                return makeEval(
                    "",
                    makeIntrinsic("*((" + vtype + "*)&(%)) = max(*((*" + vtype +
                                      ")&(%)), make_" + vtype + "(%))",
                                  {newLoad, newLoad, (*this)(op->expr_)},
                                  DataType::Void));
            case ReduceOp::Min:
                return makeEval(
                    "",
                    makeIntrinsic("*((" + vtype + "*)&(%)) = min(*((*" + vtype +
                                      ")&(%)), make_" + vtype + "(%))",
                                  {newLoad, newLoad, (*this)(op->expr_)},
                                  DataType::Void));
            default:
                ASSERT(false);
            }
        }
    }
    return Z3Simplify::visit(op);
}

Stmt LowerVector::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    auto ret = Z3Simplify::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

Stmt lowerVector(const Stmt &_op) {
    auto op = LowerVector()(_op);
    return simplifyPass(op);
}

} // namespace gpu

} // namespace ir