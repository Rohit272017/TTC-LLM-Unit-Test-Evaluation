#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include <string>
#include <utility>
#include <variant>
namespace tensorflow {
namespace {
using Signature = DeviceCompilationClusterSignature;
using TensorTypeAndShape = Signature::TensorTypeAndShape;
struct SignatureHumanStringAppender {
  explicit SignatureHumanStringAppender(std::string* dest) : dest(dest) {}
  std::string* dest;
  void operator()(const Tensor& arg) {
    absl::StrAppend(dest, "; ", arg.DebugString());
  }
  void operator()(const TensorTypeAndShape& arg) {
    absl::StrAppend(dest, ",", DataTypeString(arg.first));
    absl::StrAppend(dest, " [", absl::StrJoin(arg.second, ","), "]");
  }
};
struct SignatureNotEqual {
  bool operator()(const Tensor& arg, const Tensor& other) {
    return arg.dtype() != other.dtype() || arg.shape() != other.shape() ||
           arg.tensor_data() != other.tensor_data();
  }
  bool operator()(const TensorTypeAndShape& arg,
                  const TensorTypeAndShape& other) {
    return arg.first != other.first || arg.second != other.second;
  }
  bool operator()(const Tensor& arg, const TensorTypeAndShape& other) {
    return true;
  }
  bool operator()(const TensorTypeAndShape& arg, const Tensor& other) {
    return true;
  }
};
struct SignatureHashCombiner {
  explicit SignatureHashCombiner(const uint64 h) : h(h) {}
  uint64 h;
  uint64 operator()(const Tensor& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.dtype())));
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
    for (int dim = 0; dim < arg.dims(); ++dim) {
      h = Hash64Combine(h, std::hash<int>()(arg.dim_size(dim)));
    }
    return h;
  }
  uint64 operator()(const TensorTypeAndShape& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
    return h;
  }
};
}  
std::string Signature::HumanString() const {
  std::string result = name;
  for (const auto& arg : args) {
    std::visit(SignatureHumanStringAppender(&result), arg);
  }
  return result;
}
bool Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (args.size() != other.args.size()) return false;
  for (int i = 0, end = args.size(); i < end; ++i) {
    if (std::visit(SignatureNotEqual(), args[i], other.args[i])) {
      return false;
    }
  }
  return true;
}
uint64 Signature::Hash::operator()(const Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.args) {
    h = std::visit(SignatureHashCombiner(h), arg);
  }
  return h;
}
absl::StatusOr<Signature> Signature::Build(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
      case XlaCompiler::Argument::kConstantResource:
        signature.args.push_back(arg.constant_value);
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.args.push_back(
            TensorTypeAndShape(arg.type, arg.DimensionSizesAsInlinedVector()));
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }
  return std::move(signature);
}
}  