#ifndef STATE_MACHINE_SIMPLIFY_H
#define STATE_MACHINE_SIMPLIFY_H

#include <func.h>
#include <stmt.h>

namespace ir {

Stmt stateMachineSimplify(const Stmt &op);

DEFINE_PASS_FOR_FUNC(stateMachineSimplify)

} // namespace ir

#endif
