#include <debug.h>
#include <expr.h>
#include <ffi.h>
#include <stmt.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_ast(py::module_ &m) {
    py::class_<AST> pyAST(m, "AST");
    pyAST.def(py::init<>())
        .def("match",
             [](const AST &op, const AST &other) { return match(op, other); })
        .def("__str__",
             [](const AST &op) { return op.isValid() ? toString(op) : ""; })
        .def("__repr__", [](const AST &op) {
            return op.isValid() ? "<AST: " + toString(op) + ">" : "None";
        });

    py::class_<Stmt> pyStmt(m, "Stmt", pyAST);
    pyStmt.def(py::init<>());

    py::class_<Expr> pyExpr(m, "Expr", pyAST);
    pyExpr.def(py::init<>())
        .def(py::init([](int val) { return makeIntConst(val); }))
        .def(py::init([](float val) { return makeFloatConst(val); }))
        .def(
            "__add__",
            [](const Expr &lhs, const Expr &rhs) { return makeAdd(lhs, rhs); },
            py::is_operator())
        .def(
            "__radd__",
            [](const Expr &rhs, const Expr &lhs) { return makeAdd(lhs, rhs); },
            py::is_operator())
        .def(
            "__sub__",
            [](const Expr &lhs, const Expr &rhs) { return makeSub(lhs, rhs); },
            py::is_operator())
        .def(
            "__rsub__",
            [](const Expr &rhs, const Expr &lhs) { return makeSub(lhs, rhs); },
            py::is_operator())
        .def(
            "__mul__",
            [](const Expr &lhs, const Expr &rhs) { return makeMul(lhs, rhs); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const Expr &rhs, const Expr &lhs) { return makeMul(lhs, rhs); },
            py::is_operator())
        .def(
            "__truediv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const Expr &rhs, const Expr &lhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__floordiv__",
            [](const Expr &lhs, const Expr &rhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__rfloordiv__",
            [](const Expr &rhs, const Expr &lhs) { return makeDiv(lhs, rhs); },
            py::is_operator())
        .def(
            "__mod__",
            [](const Expr &lhs, const Expr &rhs) { return makeMod(lhs, rhs); },
            py::is_operator())
        .def(
            "__rmod__",
            [](const Expr &rhs, const Expr &lhs) { return makeMod(lhs, rhs); },
            py::is_operator())
        .def(
            "__lt__",
            [](const Expr &lhs, const Expr &rhs) { return makeLT(lhs, rhs); },
            py::is_operator())
        .def(
            "__le__",
            [](const Expr &lhs, const Expr &rhs) { return makeLE(lhs, rhs); },
            py::is_operator())
        .def(
            "__gt__",
            [](const Expr &lhs, const Expr &rhs) { return makeGT(lhs, rhs); },
            py::is_operator())
        .def(
            "__ge__",
            [](const Expr &lhs, const Expr &rhs) { return makeGE(lhs, rhs); },
            py::is_operator())
        .def(
            "__eq__",
            [](const Expr &lhs, const Expr &rhs) { return makeEQ(lhs, rhs); },
            py::is_operator())
        .def(
            "__ne__",
            [](const Expr &lhs, const Expr &rhs) { return makeNE(lhs, rhs); },
            py::is_operator());
    py::implicitly_convertible<int, Expr>();
    py::implicitly_convertible<float, Expr>();

    // Statements
    m.def("makeAny", &makeAny);
    m.def("makeStmtSeq",
          static_cast<Stmt (*)(const std::string &, const std::vector<Stmt> &)>(
              &makeStmtSeq),
          "id"_a, "stmts"_a);
    m.def("makeVarDef",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const Buffer &, const Stmt &)>(&makeVarDef),
          "nid"_a, "name"_a, "buffer"_a, "body"_a);
    m.def("makeVar", &makeVar, "name"_a);
    m.def("makeStore",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const std::vector<Expr> &, const Expr &)>(
              &makeStore),
          "nid"_a, "var"_a, "indices"_a, "expr"_a);
    m.def("makeLoad", &makeLoad, "var"_a, "indices"_a);
    m.def("makeIntConst", &makeIntConst, "val"_a);
    m.def("makeFloatConst", &makeFloatConst, "val"_a);
    m.def("makeFor",
          static_cast<Stmt (*)(const std::string &, const std::string &,
                               const Expr &, const Expr &, const std::string &,
                               const Stmt &)>(&makeFor),
          "nid"_a, "iter"_a, "begin"_a, "end"_a, "parallel"_a, "body"_a);
    m.def("makeIf",
          static_cast<Stmt (*)(const std::string &, const Expr &, const Stmt &,
                               const Stmt &)>(&makeIf),
          "nid"_a, "cond"_a, "thenCase"_a, "elseCase"_a = Stmt());
    m.def("makeEval",
          static_cast<Stmt (*)(const std::string &, const Expr &)>(&makeEval),
          "nid"_a, "expr"_a);

    // Expressions
    m.def("makeMin",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeMin), "lhs"_a,
          "rhs"_a);
    m.def("makeMax",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeMax), "lhs"_a,
          "rhs"_a);
    m.def("makeLAnd",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeLAnd), "lhs"_a,
          "rhs"_a);
    m.def("makeLOr",
          static_cast<Expr (*)(const Expr &, const Expr &)>(&makeLOr), "lhs"_a,
          "rhs"_a);
    m.def("makeLNot", static_cast<Expr (*)(const Expr &)>(&makeLNot), "expr"_a);
    m.def("makeIntrinsic",
          static_cast<Expr (*)(const std::string &, const std::vector<Expr> &)>(
              &makeIntrinsic),
          "fmt"_a, "params"_a);
}

} // namespace ir

