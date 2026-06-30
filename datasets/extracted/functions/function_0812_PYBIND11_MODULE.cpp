#include "tensorflow/core/util/determinism.h"
#include "pybind11/pybind11.h"  
PYBIND11_MODULE(_pywrap_determinism, m) {
  m.def("enable", &tensorflow::EnableOpDeterminism);
  m.def("is_enabled", &tensorflow::OpDeterminismRequired);
}