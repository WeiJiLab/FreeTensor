#include <ffi.h>
#include <frontend/frontend_var.h>
#include <frontend/inlined_invoke.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_frontend(py::module_ &m) {
    py::class_<FrontendVarIdx>(m, "FrontendVarIdx")
        .def(py::init(&FrontendVarIdx::fromSingle))
        .def(py::init(&FrontendVarIdx::fromSlice))
        .def("__repr__",
             [](const FrontendVarIdx &idx) { return toString(idx); });

    m.def("all_reads", static_cast<std::unordered_set<std::string> (*)(
                           const FrontendVarIdx &)>(&allReads));

    py::class_<FrontendVar, Ref<FrontendVar>>(m, "FrontendVar")
        .def(py::init<const std::string &, const std::vector<Expr> &, DataType,
                      MemType, const std::vector<FrontendVarIdx> &>())
        .def_property_readonly("name", &FrontendVar::name)
        .def_property_readonly("full_shape", &FrontendVar::fullShape)
        .def_property_readonly("indices", &FrontendVar::indices)
        .def_property_readonly("dtype", &FrontendVar::dtype)
        .def_property_readonly("mtype", &FrontendVar::mtype)
        .def_property_readonly("ndim", &FrontendVar::ndim)
        .def("shape", &FrontendVar::shape)
        .def("as_load", &FrontendVar::asLoad)
        .def("as_store", &FrontendVar::asStore)
        .def("chain_indices", &FrontendVar::chainIndices)
        .def("__repr__", [](const FrontendVar &var) { return toString(var); });

    m.def("inlined_invoke", &inlinedInvoke);
}

} // namespace freetensor
