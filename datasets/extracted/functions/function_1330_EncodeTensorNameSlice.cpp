#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/ordered_code.h"
#include "tensorflow/core/lib/strings/str_util.h"
namespace tensorflow {
namespace checkpoint {
const char kSavedTensorSlicesKey[] = "";
string EncodeTensorNameSlice(const string& name, const TensorSlice& slice) {
  string buffer;
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, 0);
  tensorflow::strings::OrderedCode::WriteString(&buffer, name);
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, slice.dims());
  for (int d = 0; d < slice.dims(); ++d) {
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.start(d));
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.length(d));
  }
  return buffer;
}
Status DecodeTensorNameSlice(const string& code, string* name,
                             tensorflow::TensorSlice* slice) {
  StringPiece src(code);
  uint64 x;
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the leading number: src = ", src);
  }
  if (x != 0) {
    return errors::Internal(
        "The leading number should always be 0 for any valid key: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadString(&src, name)) {
    return errors::Internal("Failed to parse the tensor name: src = ", src);
  }
  if (!tensorflow::strings::OrderedCode::ReadNumIncreasing(&src, &x)) {
    return errors::Internal("Failed to parse the tensor rank: src = ", src);
  }
  if (x == 0) {
    return errors::Internal("Expecting positive rank of the tensor, got ", x,
                            ", src = ", src);
  }
  if (x >= kint32max) {
    return errors::Internal("Too many elements ", x);
  }
  slice->SetFullSlice(x);
  for (int d = 0; d < static_cast<int32>(x); ++d) {
    int64_t start, length;
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &start)) {
      return errors::Internal("Failed to parse start: src = ", src);
    }
    if (!tensorflow::strings::OrderedCode::ReadSignedNumIncreasing(&src,
                                                                   &length)) {
      return errors::Internal("Failed to parse length: src = ", src);
    }
    if (length >= 0) {
      slice->set_start(d, start);
      slice->set_length(d, length);
    }
  }
  return absl::OkStatus();
}
Status ParseShapeAndSlice(const string& shape_and_slice, TensorShape* shape,
                          TensorSlice* slice, TensorShape* shape_slice) {
  CHECK(!shape_and_slice.empty());
  std::vector<string> splits = str_util::Split(shape_and_slice, ' ');
  if (splits.size() < 2) {
    return errors::InvalidArgument(
        "Need least two elements in shape_and_slice specification: ",
        shape_and_slice);
  }
  slice->Clear();
  auto status = slice->Parse(splits.back(), slice);
  if (!status.ok()) return status;
  splits.pop_back();
  shape->Clear();
  for (const auto& s : splits) {
    int64_t dim;
    if (!strings::safe_strto64(s, &dim)) {
      return errors::InvalidArgument(
          "Non numerical dimension in shape_and_slice: ", shape_and_slice);
    }
    shape->AddDim(dim);
  }
  return slice->SliceTensorShape(*shape, shape_slice);
}
}  
}  