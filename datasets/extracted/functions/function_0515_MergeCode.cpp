#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include <algorithm>
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/rename.h"
namespace tflite {
namespace gpu {
namespace gl {
absl::Status MergeCode(CompiledNodeAttributes* attr,
                       CompiledNodeAttributes* merged_attr) {
  absl::flat_hash_set<std::string> known_names;
  for (const auto& parameter : merged_attr->code.parameters) {
    known_names.insert(parameter.name);
  }
  for (const auto& object : merged_attr->code.objects) {
    known_names.insert(object.first);
  }
  int index =
      merged_attr->code.parameters.size() + merged_attr->code.objects.size();
  RETURN_IF_ERROR(Rename(
      [&](absl::string_view name) -> std::string {
        std::string n(name.begin(), name.end());
        std::string ret = n;
        while (known_names.find(ret) != known_names.end()) {
          ret = absl::StrCat(n, index++);
        }
        known_names.insert(ret);
        return ret;
      },
      &attr->code));
  std::move(attr->code.objects.begin(), attr->code.objects.end(),
            std::back_inserter(merged_attr->code.objects));
  std::move(attr->code.parameters.begin(), attr->code.parameters.end(),
            std::back_inserter(merged_attr->code.parameters));
  std::move(attr->node_indices.begin(), attr->node_indices.end(),
            std::back_inserter(merged_attr->node_indices));
  return absl::OkStatus();
}
}  
}  
}  