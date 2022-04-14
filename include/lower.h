#ifndef IR_LOWER_H
#define IR_LOWER_H

#include <driver/target.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/float_simplify.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/gpu/lower_vector.h>
#include <pass/gpu/make_sync.h>
#include <pass/gpu/multiplex_buffers.h>
#include <pass/gpu/normalize_threads.h>
#include <pass/gpu/simplex_buffers.h>
#include <pass/make_1d_var.h>
#include <pass/make_const_shape.h>
#include <pass/make_parallel_reduction.h>
#include <pass/merge_and_hoist_if.h>
#include <pass/move_out_first_or_last_iter.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/scalar_prop_const.h>
#include <pass/shrink_for.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <pass/tensor_prop_const.h>
#include <pass/use_builtin_div.h>

#include <iomanip>

namespace ir {

template <class T> T lower(const T &t, const Ref<Target> &target) {
    T func = t;
    auto log = [](int line) {
        // auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        // std::cerr << "lower: " << line << std::put_time(std::localtime(&t), " - %F %T") << std::endl;
    };
    #define LOG log(__LINE__);
    LOG
    func = scalarPropConst(func);
    LOG
    func = removeDeadVar(func);
    LOG
    func = propOneTimeUse(func);
    LOG
    func = floatSimplify(func); // After propOneTimeUse
    LOG
    func = simplifyPass(func);
    LOG
    func = moveOutFirstOrLastIter(func);
    LOG
    func = sinkVar(func);
    LOG
    func = shrinkVar(func);
    LOG
    func = mergeAndHoistIf(func);
    LOG
    func = tensorPropConst(func);
    LOG
    func = removeWrites(func);
    LOG
    func = removeCyclicAssign(func); // After remove_writes
    LOG
    func = removeDeadVar(func);      // After remove_writes and prop_const
    LOG
    func = makeParallelReduction(func);
    LOG
    func = shrinkFor(func); // After remove_writes and make_parallel_reduction
    LOG

    if (target.isValid()) {
        switch (target->type()) {
        case TargetType::GPU:
            // Before gpu_nromalize_threads
            func = gpu::lowerParallelReduction(func);

            // TODO: Support dynamic shared memory size, but the size should be
            // determined outside of kernels
            func = gpu::multiplexBuffers(func);
            func = gpu::simplexBuffers(func);
            // FIXME: MemType::GPUGlobal should also be make const, but only
            // inside a kernel
            func =
                makeConstShape(func, std::vector<MemType>{MemType::GPUShared,
                                                          MemType::GPULocal});
            func = gpu::normalizeThreads(func); // After gpu_multiplex_buffers
            func = gpu::makeSync(func);         // After gpu_normalize_threads
            func = make1dVar(func); // FIXME: make1dVar will break the shape of
                                    // returned tensors
            func = gpu::lowerVector(func); // After make_1d_var
            break;

        case TargetType::CPU:
            func = cpu::lowerParallelReduction(func);
            break;

        default:
            ASSERT(false);
        }
    }
    LOG

    // After passes including architecture-specific ones
    func = useBuiltinDiv(func);
    LOG

    return func;
}

} // namespace ir

#endif // IR_LOWER_H
