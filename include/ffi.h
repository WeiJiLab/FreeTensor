#ifndef FFI_H
#define FFI_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace ir {

namespace py = pybind11;

void init_ffi_tensor(py::module_ &m);
void init_ffi_buffer(py::module_ &m);
void init_ffi_ast(py::module_ &m);

} // namespace ir

#endif // FFI_H
