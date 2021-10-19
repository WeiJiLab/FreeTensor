#ifndef GPU_MULTIPLEX_BUFFERS_H
#define GPU_MULTIPLEX_BUFFERS_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

namespace gpu {

class FindParallelLoops : public Visitor {
    std::vector<For> loops_, stack_;
    std::unordered_map<std::string, std::unordered_set<std::string>> affecting_;

  public:
    const std::vector<For> &loops() const { return loops_; }
    const std::unordered_map<std::string, std::unordered_set<std::string>> &
    affecting() const {
        return affecting_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
};

class MultiplexMutator : public Mutator {
    std::vector<For> stack_;
    std::unordered_map<std::string, int> defPos_;
    std::unordered_map<std::string, std::string> defs_; // name -> ID
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        &affecting_; // VarDef ID -> For ID

  public:
    MultiplexMutator(
        const std::unordered_map<std::string, std::unordered_set<std::string>>
            &affecting)
        : affecting_(affecting) {}

  private:
    template <class T> T alterAccess(const T &op) {
        if (!defPos_.count(op->var_)) {
            return op;
        }
        if (affecting_.count(defs_.at(op->var_))) {
            auto &&aff = affecting_.at(defs_.at(op->var_));
            int pos = defPos_.at(op->var_);
            for (int i = pos - 1; i >= 0; i--) {
                if (aff.count(stack_[i]->id())) {
                    auto &indices = op->indices_;
                    indices.insert(indices.begin(), makeVar(stack_[i]->iter_));
                }
            }
        }
        return op;
    }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

/**
 * If a shared or global VarDef is inside a parallel For region, it should be
 * enlarged so that each thread or block will access different parts of it
 *
 * E.g. Alter from `shmem[i]` to `shmem[threadIdx.x, i]`
 */
Stmt multiplexBuffers(const Stmt &op);

DEFINE_PASS_FOR_FUNC(multiplexBuffers)

} // namespace gpu

} // namespace ir

#endif // GPU_MULTIPLEX_BUFFERS_H