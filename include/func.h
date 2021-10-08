#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <ast.h>
#include <buffer.h>
#include <frontend_utils.h>
#include <stmt.h>
#include <tensor.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    SubTree<StmtNode> body_;
    Ref<pybind11::object> src_;

    DEFINE_NODE_TRAIT(Func);

    ~FuncNode() {
#pragma omp critical
        { src_ = nullptr; }
    }
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               Tbody &&body, const pybind11::object &src) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->body_ = std::forward<Tbody>(body);
#pragma omp critical
    { f->src_ = Ref<pybind11::object>::make(src); }
    return f;
}
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               Tbody &&body, const Ref<pybind11::object> &src) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->body_ = std::forward<Tbody>(body);
    f->src_ = src;
    return f;
}

Func deepCopy(const Func &func);

#define DEFINE_PASS_FOR_FUNC(pass)                                             \
    template <typename... T> Func pass(const Func &func, T &&...args) {        \
        return makeFunc(func->name_, func->params_,                            \
                        pass(func->body_, std::forward<T>(args)...),           \
                        func->src_);                                           \
    }

} // namespace ir

#endif // FUNC_H
