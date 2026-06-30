#include "tensorflow/core/graph/tensor_id.h"
#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
namespace tensorflow {
TensorId::TensorId(const SafeTensorId& id) : TensorId(id.first, id.second) {}
SafeTensorId::SafeTensorId(const TensorId& id)
    : SafeTensorId(string(id.first), id.second) {}
TensorId ParseTensorName(const string& name) {
  return ParseTensorName(StringPiece(name.data(), name.size()));
}
TensorId ParseTensorName(StringPiece name) {
  const char* base = name.data();
  const char* p = base + name.size() - 1;
  unsigned int index = 0;
  unsigned int mul = 1;
  while (p > base && (*p >= '0' && *p <= '9')) {
    index += ((*p - '0') * mul);
    mul *= 10;
    p--;
  }
  TensorId id;
  if (p > base && *p == ':' && mul > 1) {
    id.first = StringPiece(base, p - base);
    id.second = index;
  } else if (absl::StartsWith(name, "^")) {
    id.first = StringPiece(base + 1);
    id.second = Graph::kControlSlot;
  } else {
    id.first = name;
    id.second = 0;
  }
  return id;
}
bool IsTensorIdControl(const TensorId& tensor_id) {
  return tensor_id.index() == Graph::kControlSlot;
}
}  