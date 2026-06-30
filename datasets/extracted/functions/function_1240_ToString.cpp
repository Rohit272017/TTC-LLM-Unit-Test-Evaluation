#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include "tensorflow/lite/experimental/shlo/data_type.h"
namespace shlo_ref {
std::string ToString(const QuantizedElementTypePerTensor& t) {
  std::stringstream sstr;
  sstr << "QuantizedPerTensor[" << ToString(t.StorageType()) << ", "
       << ToString(t.ExpressedType()) << "]";
  return sstr.str();
}
std::string ToString(const QuantizedElementTypePerAxis& t) {
  std::stringstream sstr;
  sstr << "QuantizedPerAxis[" << ToString(t.StorageType()) << ", "
       << ToString(t.ExpressedType()) << ", " << t.QuantizedDimension() << "]";
  return sstr.str();
}
QuantizedElementTypePerTensor BaselineType(
    const QuantizedElementTypePerTensor& type) {
  QuantizedElementTypePerTensor baseline = type;
  std::visit(
      [](auto& scale) -> void {
        scale = std::remove_reference_t<decltype(scale)>(1);
      },
      baseline.Scale());
  std::visit(
      [](auto& zero_point) -> void {
        zero_point = std::remove_reference_t<decltype(zero_point)>(0);
      },
      baseline.ZeroPoint());
  return baseline;
}
QuantizedElementTypePerAxis BaselineType(
    const QuantizedElementTypePerAxis& type) {
  QuantizedElementTypePerAxis baseline = type;
  std::visit(
      [](auto& scales) -> void {
        using T = std::remove_reference_t<decltype(scales[0])>;
        absl::c_fill(scales, static_cast<T>(1));
      },
      baseline.Scales());
  std::visit(
      [](auto& zero_points) -> void {
        using T = std::remove_reference_t<decltype(zero_points[0])>;
        absl::c_fill(zero_points, static_cast<T>(0));
      },
      baseline.ZeroPoints());
  return baseline;
}
}  