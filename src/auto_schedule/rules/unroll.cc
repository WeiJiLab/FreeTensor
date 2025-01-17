#include <analyze/get_loop_nest_tree.h>
#include <auto_schedule/rules/parallelize.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/rules/unroll.h>
#include <auto_schedule/utils.h>
#include <schedule/unroll.h>

namespace freetensor {

static std::vector<int> unrollConfigsCpu = {0, 16, 64, 512};
static std::vector<int> unrollConfigsGpu = {0, 16, 64, 512, 1024};

void UnrollPart::apply(Schedule &schedule, SketchTarget &target) {
    Stmt root;
    int vthreadSize = 1;
    if (targetType_ == TargetType::GPU) {
        SketchPart part = target.getPart(SketchPartType::ThreadBind);
        ID lastParallelizedID = part.as<ThreadBindPart>()->lastParallelizedID_;
        if (!lastParallelizedID.isValid()) {
            return;
        }
        root = schedule.find(lastParallelizedID).as<ForNode>()->body_;
        vthreadSize = part.as<ThreadBindPart>()->vthreadSize_;
    } else {
        SketchPart part = target.getPart(SketchPartType::Parallelize);
        ID lastParallelizedID = part.as<ParallelizePart>()->lastParallelizedID_;
        if (!lastParallelizedID.isValid()) {
            return;
        }
        root = schedule.find(lastParallelizedID).as<ForNode>()->body_;
    }
    std::function<int(const Ref<LoopNest> &nest)> visitNest =
        [&](const Ref<LoopNest> &nest) {
            int sz = 0;
            for (auto &&subNest : nest->subLoops_) {
                sz += visitNest(subNest);
            }
            if (sz == 0) {
                sz = vthreadSize;
            }
            auto &&loop = nest->loop_;
            if (loop.isValid()) { // not root
                if (loop->property_->parallel_ == serialScope &&
                    !loop->property_->vectorize_ && !loop->property_->unroll_ &&
                    loop->len_->nodeType() == ASTNodeType::IntConst &&
                    sz * loop->len_.as<IntConstNode>()->val_ <= maxSize_) {
                    sz *= loop->len_.as<IntConstNode>()->val_;
                    schedule.unroll(loop->id());
                }
            }
            return sz;
        };
    visitNest(getLoopNestTree(root));
}

void UnrollPart::genRandAnnotation(std::default_random_engine &gen) {
    std::vector<int> &unrollConfigs =
        targetType_ == TargetType::GPU ? unrollConfigsGpu : unrollConfigsCpu;
    maxSize_ = unrollConfigs[randomInt(unrollConfigs.size() - 1, gen)];
}

bool UnrollPart::mutate(std::default_random_engine &gen) {
    std::vector<int> &unrollConfigs =
        targetType_ == TargetType::GPU ? unrollConfigsGpu : unrollConfigsCpu;
    maxSize_ = unrollConfigs[randomInt(unrollConfigs.size() - 1, gen)];
    return true;
}
bool UnrollPart::crossover(const SketchPart &part,
                           std::default_random_engine &gen) {
    if (auto p = part.as<UnrollPart>(); p.isValid()) {
        maxSize_ = p->maxSize_;
        return true;
    }
    return false;
}

std::vector<Sketch> UnrollRule::genPart(const Sketch &sketch) {
    Sketch newSketch = sketch.clone();
    newSketch.addPart(Ref<UnrollPart>::make(targetType_));
    newSketch.addLog("unroll");
    return {newSketch};
}

RuleStatus UnrollRule::analyze(const Sketch &sketch) {
    if (sketch.nowTarget().hasPart(SketchPartType::Unroll))
        return RuleStatus::Skip;
    if (sketch.nowTarget().hasPart(
            SketchPartType::MultiLevelTilingWithFusion) ||
        sketch.nowTarget().hasPart(SketchPartType::MultiLevelTilingWithFusion))
        return RuleStatus::ApplyAndSkipRest;
    return RuleStatus::Skip;
}

} // namespace freetensor
