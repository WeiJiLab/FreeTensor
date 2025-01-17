#include <analyze/all_defs.h>

namespace freetensor {

void AllDefs::visit(const VarDef &op) {
    Visitor::visit(op);
    if (atypes_.count(op->buffer_->atype())) {
        results_.emplace_back(op->id(), op->name_);
    }
}

std::vector<std::pair<ID, std::string>>
allDefs(const Stmt &op, const std::unordered_set<AccessType> &atypes) {
    AllDefs visitor(atypes);
    visitor(op);
    return visitor.results();
}

} // namespace freetensor
