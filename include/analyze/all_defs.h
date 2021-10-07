#ifndef ALL_DEFS_H
#define ALL_DEFS_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

class AllDefs : public Visitor {
    const std::unordered_set<AccessType> &atypes_;
    std::vector<std::string> results_;

  public:
    AllDefs(const std::unordered_set<AccessType> &atypes) : atypes_(atypes) {}

    const std::vector<std::string> &results() const { return results_; }

  protected:
    void visit(const VarDef &op) override;
};

/**
 * Collect IDs of all `VarDef` nodes of specific `AccessType`s
 */
std::vector<std::string> allDefs(const Stmt &op,
                                 const std::unordered_set<AccessType> &atypes);

} // namespace ir

#endif // ALL_DEFS_H
