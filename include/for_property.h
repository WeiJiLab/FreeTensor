#ifndef FOR_PROPERTY_H
#define FOR_PROPERTY_H

#include <expr.h>
#include <parallel_scope.h>
#include <reduce_op.h>
#include <sub_tree.h>

namespace ir {

struct ReductionItem : public ASTPart {
    ReduceOp op_;
    std::string var_;
    SubTreeList<ExprNode> begins_, ends_;

    template <class Tbegins, class Tends>
    ReductionItem(ReduceOp op, const std::string &var, Tbegins &&begins,
                  Tends &&ends)
        : op_(op), var_(var), begins_(std::forward<Tbegins>(begins)),
          ends_(std::forward<Tends>(ends)) {}
};

struct ForProperty : public ASTPart {
    ParallelScope parallel_;
    bool unroll_, vectorize_;
    SubTreeList<ReductionItem> reductions_;
    std::vector<std::string> noDeps_; // vars that are explicitly marked to have
                                      // no dependencies over this loop
    bool preferLibs_; // Aggresively transform to external library calls in
                      // auto-schedule

    ForProperty()
        : parallel_(), unroll_(false), vectorize_(false), preferLibs_(false) {}

    Ref<ForProperty> withParallel(const ParallelScope &parallel) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->parallel_ = parallel;
        return ret;
    }
    Ref<ForProperty> withUnroll(bool unroll = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->unroll_ = unroll;
        return ret;
    }
    Ref<ForProperty> withVectorize(bool vectorize = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->vectorize_ = vectorize;
        return ret;
    }
    Ref<ForProperty> withNoDeps(const std::vector<std::string> &noDeps) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->noDeps_ = noDeps;
        return ret;
    }
    Ref<ForProperty> withPreferLibs(bool preferLibs = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->preferLibs_ = preferLibs;
        return ret;
    }
};

inline Ref<ReductionItem> deepCopy(const Ref<ReductionItem> &r) {
    return Ref<ReductionItem>::make(*r);
}

inline Ref<ForProperty> deepCopy(const Ref<ForProperty> &p) {
    return Ref<ForProperty>::make(*p);
}

} // namespace ir

#endif // FOR_PROPERTY_H