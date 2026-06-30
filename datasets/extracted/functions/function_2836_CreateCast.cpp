#include "tensorflow/lite/delegates/gpu/common/tasks/cast.h"
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
namespace tflite {
namespace gpu {
GPUOperation CreateCast(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
  ElementwiseDescriptor op_desc;
  const std::string conversion =
      GetTypeConversion(gpu_info, definition.src_tensors[0].GetDataType(),
                        definition.dst_tensors[0].GetDataType(), 4);
  op_desc.code =
      "out_value = " + absl::Substitute(conversion, "in_value") + ";\n";
  return CreateGpuOperation(definition, std::move(op_desc));
}
}  
}  