#include <pybind11/pybind11.h>
#include "python/tensorstore/index.h"
#include <cstddef>
#include <string>
#include <variant>
#include <vector>
#include "python/tensorstore/sequence_parameter.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/index_space/internal/numpy_indexing_spec.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_python {
IndexVectorOrScalarContainer ToIndexVectorOrScalarContainer(
    const OptionallyImplicitIndexVectorOrScalarContainer& x,
    Index implicit_value) {
  if (auto* index = std::get_if<OptionallyImplicitIndex>(&x)) {
    return index->value_or(implicit_value);
  }
  const auto& v = std::get<SequenceParameter<OptionallyImplicitIndex>>(x);
  std::vector<Index> out_v;
  out_v.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out_v.push_back(v[i].value_or(implicit_value));
  }
  return out_v;
}
internal_index_space::IndexVectorOrScalarView ToIndexVectorOrScalar(
    const IndexVectorOrScalarContainer& x) {
  constexpr static Index temp = 0;
  if (auto* index = std::get_if<Index>(&x)) {
    return *index;
  } else {
    const auto& v = std::get<std::vector<Index>>(x);
    if (v.empty()) {
      return span(&temp, 0);
    }
    return span(v);
  }
}
std::string IndexVectorRepr(const IndexVectorOrScalarContainer& x,
                            bool implicit, bool subscript) {
  return internal::IndexVectorRepr(ToIndexVectorOrScalar(x), implicit,
                                   subscript);
}
}  
}  
namespace pybind11 {
namespace detail {
handle type_caster<tensorstore::internal_python::PythonDimensionIndex>::cast(
    tensorstore::internal_python::PythonDimensionIndex x,
    return_value_policy , handle ) {
  return int_(x.value).release();
}
bool type_caster<tensorstore::internal_python::PythonDimensionIndex>::load(
    handle src, bool convert) {
  value.value = PyNumber_AsSsize_t(src.ptr(), PyExc_IndexError);
  if (value.value == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return false;
  }
  return true;
}
handle type_caster<tensorstore::internal_python::OptionallyImplicitIndex>::cast(
    tensorstore::internal_python::OptionallyImplicitIndex x,
    return_value_policy , handle ) {
  if (x.value == tensorstore::kImplicit) return none().release();
  return int_(x.value).release();
}
bool type_caster<tensorstore::internal_python::OptionallyImplicitIndex>::load(
    handle src, bool convert) {
  if (src.is_none()) {
    value.value = tensorstore::kImplicit;
    return true;
  }
  value.value = PyNumber_AsSsize_t(src.ptr(), PyExc_IndexError);
  if (value.value == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return false;
  }
  return true;
}
}  
}  