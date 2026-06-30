#include "tensorflow/lite/tools/optimize/modify_model_interface.h"
#include <string>
#include "pybind11/pybind11.h"  
#include "tensorflow/lite/schema/schema_generated.h"
namespace pybind11 {
PYBIND11_MODULE(_pywrap_modify_model_interface, m) {
  m.def("modify_model_interface",
        [](const std::string& input_file, const std::string& output_file,
           const int input_type, const int output_type) -> int {
          return tflite::optimize::ModifyModelInterface(
              input_file, output_file,
              static_cast<tflite::TensorType>(input_type),
              static_cast<tflite::TensorType>(output_type));
        });
}
}  