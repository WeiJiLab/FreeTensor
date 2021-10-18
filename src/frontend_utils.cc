#include <except.h>
#include <frontend_utils.h>

namespace ir {

int FrontendVar::ndim() const {
    int ndim = fullShape_.size();
    for (auto &&idx : indices_) {
        if (idx.type() == FrontendVarIdxType::Single) {
            ndim--;
        }
    }
    return ndim;
}

Expr FrontendVar::shape(const Expr &idx) const {
    Expr ret;
    size_t j = 0, k = 0;
    for (size_t i = 0, n = fullShape_.size(); i < n; i++) {
        if (j < indices_.size() &&
            indices_[j].type() == FrontendVarIdxType::Single) {
            j++;
            continue;
        }
        Expr dimLen = fullShape_.at(i);
        while (j < indices_.size() &&
               indices_[j].type() == FrontendVarIdxType::Slice) {
            dimLen = makeSub(indices_[j].stop(), indices_[j].start());
            j++;
        }
        if (idx->nodeType() == ASTNodeType::IntConst &&
            (size_t)idx.as<IntConstNode>()->val_ == k) {
            return dimLen;
        } else {
            ret = ret.isValid()
                      ? makeIfExpr(makeEQ(idx, makeIntConst(k)), dimLen, ret)
                      : dimLen;
        }
        k++;
    }
    ASSERT(ret.isValid());
    return ret;
}

Expr FrontendVar::asLoad() const {
    if (ndim() != 0) {
        throw InvalidProgram(
            name_ + " is of a " + std::to_string(fullShape_.size()) +
            "-D shape, but " + std::to_string((int)fullShape_.size() - ndim()) +
            "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeLoad(name_, std::move(indices));
}

Stmt FrontendVar::asStore(const std::string &id, const Expr &value) const {
    if (ndim() != 0) {
        throw InvalidProgram(
            name_ + " is of a " + std::to_string(fullShape_.size()) +
            "-D shape, but " + std::to_string((int)fullShape_.size() - ndim()) +
            "-D indices are given");
    }
    std::vector<Expr> indices;
    indices.reserve(indices_.size());
    for (auto &&idx : indices_) {
        ASSERT(idx.type() == FrontendVarIdxType::Single);
        indices.emplace_back(idx.single());
    }
    return makeStore(id, name_, std::move(indices), value);
}

std::vector<FrontendVarIdx>
FrontendVar::chainIndices(const std::vector<FrontendVarIdx> &next) const {
    std::vector<FrontendVarIdx> indices;
    auto it = next.begin();
    for (auto &&idx : indices_) {
        if (idx.type() == FrontendVarIdxType::Single) {
            indices.emplace_back(idx);
        } else {
            if (it != next.end()) {
                if (it->type() == FrontendVarIdxType::Single) {
                    indices.emplace_back(FrontendVarIdx::fromSingle(
                        makeAdd(idx.start(), it->single())));
                } else {
                    indices.emplace_back(FrontendVarIdx::fromSlice(
                        makeAdd(idx.start(), it->start()),
                        makeAdd(idx.stop(), it->start())));
                }
                it++;
            } else {
                indices.emplace_back(idx); // still a slice
            }
        }
    }
    for (; it != next.end(); it++) {
        indices.emplace_back(*it);
    }
    return indices;
}

} // namespace ir
