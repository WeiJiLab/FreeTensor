#include <ffi.h>
#include <schedule.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_schedule(py::module_ &m) {
    py::class_<Schedule>(m, "Schedule")
        .def(py::init<const Stmt &>())
        .def("ast", &Schedule::ast)
        .def("split", &Schedule::split, "id"_a, "factor"_a = -1,
             "nparts"_a = -1)
        .def("reorder", &Schedule::reorder, "order"_a)
        .def("merge", &Schedule::merge, "loop1"_a, "loop2"_a)
        .def("fission", &Schedule::fission, "loop"_a, "after"_a)
        .def("fuse", &Schedule::fuse, "loop0"_a, "loop1"_a)
        .def("swap", &Schedule::swap, "order"_a)
        .def("cache_read", &Schedule::cacheRead, "stmt"_a, "var"_a)
        .def("cache_write", &Schedule::cacheWrite, "stmt"_a, "var"_a);
}

} // namespace ir
