#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Types.h"  
#include "mlir/Support/DebugStringHelper.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/ml_dtypes.h"
namespace tensorflow {
using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Builder;
using mlir::DenseStringElementsAttr;
using mlir::ElementsAttr;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::Type;
using tensorflow::errors::InvalidArgument;
static TensorProto ConvertToProto(const Tensor& input_tensor,
                                  bool use_tensor_content = true) {
  TensorProto tensor_proto;
  if (use_tensor_content)
    input_tensor.AsProtoTensorContent(&tensor_proto);
  else
    input_tensor.AsProtoField(&tensor_proto);
  return tensor_proto;
}
static std::string MangleTensor(const Tensor& tensor) {
  return mangling_util::MangleTensor(ConvertToProto(tensor));
}
template <typename T>
absl::StatusOr<ElementsAttr> ConvertFlatTensor(const Tensor& input_tensor,
                                               ShapedType type) {
  auto arr = input_tensor.flat<T>();
  return ElementsAttr(mlir::DenseElementsAttr::get(
      type, llvm::ArrayRef(arr.data(), arr.size())));
}
ElementsAttr ConvertTensorOfCustomFloatType(const Tensor& tensor,
                                            RankedTensorType type) {
  auto buffer =
      llvm::ArrayRef(static_cast<char*>(tensor.data()), tensor.TotalBytes());
  return mlir::DenseElementsAttr::getFromRawBuffer(type, buffer);
}
absl::StatusOr<ElementsAttr> ConvertStringTensor(const Tensor& input_tensor,
                                                 ShapedType type) {
  auto arr = input_tensor.flat<tstring>();
  std::vector<mlir::StringRef> string_refs;
  string_refs.reserve(arr.size());
  for (int i = 0; i < arr.size(); i++) {
    const auto& val = arr(i);
    string_refs.push_back({val.data(), val.size()});
  }
  return ElementsAttr(DenseStringElementsAttr::get(type, string_refs));
}
absl::StatusOr<ElementsAttr> ConvertTensor(const Tensor& input_tensor,
                                           Builder* builder) {
  const auto& input_dtype = input_tensor.dtype();
  const auto& input_shape = input_tensor.shape();
  Type elt_type;
  TF_RETURN_IF_ERROR(ConvertDataType(input_dtype, *builder, &elt_type));
  SmallVector<int64_t, 4> shape;
  ConvertToMlirShape(input_shape, &shape);
  auto type = RankedTensorType::get(shape, elt_type);
#define CONVERT_FLAT(DTYPE, CTYPE) \
  case DTYPE:                      \
    return ConvertFlatTensor<CTYPE>(input_tensor, type);
  switch (input_dtype) {
    CONVERT_FLAT(DT_BOOL, bool)
    CONVERT_FLAT(DT_FLOAT, float)
    CONVERT_FLAT(DT_DOUBLE, double)
    CONVERT_FLAT(DT_INT8, int8)
    CONVERT_FLAT(DT_INT16, int16)
    CONVERT_FLAT(DT_INT32, int32)
    CONVERT_FLAT(DT_INT64, int64_t)
    CONVERT_FLAT(DT_UINT8, uint8)
    CONVERT_FLAT(DT_UINT16, uint16)
    CONVERT_FLAT(DT_UINT32, uint32)
    CONVERT_FLAT(DT_UINT64, uint64)
    CONVERT_FLAT(DT_COMPLEX64, std::complex<float>)
    CONVERT_FLAT(DT_COMPLEX128, std::complex<double>)
    case DT_BFLOAT16:
    case DT_HALF:
    case DT_FLOAT8_E5M2:
    case DT_FLOAT8_E4M3FN:
      return ConvertTensorOfCustomFloatType(input_tensor, type);
    case DT_STRING:
      return ConvertStringTensor(input_tensor, type);
    default:
      return ElementsAttr(
          mlir::TF::TensorProtoAttr::get(type, MangleTensor(input_tensor)));
  }
#undef CONVERT_FLAT
}
int NumberOfMaterializedElements(const TensorProto& tensor) {
  if (!tensor.tensor_content().empty()) return -1;
#define MATCH(DTYPE, FIELD) \
  case DTYPE:               \
    return tensor.FIELD##_val().size()
  switch (tensor.dtype()) {
    MATCH(DT_FLOAT, float);
    MATCH(DT_DOUBLE, double);
    MATCH(DT_INT8, int);
    MATCH(DT_UINT8, int);
    MATCH(DT_INT16, int);
    MATCH(DT_UINT16, int);
    MATCH(DT_INT32, int);
    MATCH(DT_UINT32, uint32);
    MATCH(DT_INT64, int64);
    MATCH(DT_UINT64, uint64);
    MATCH(DT_BOOL, bool);
    MATCH(DT_HALF, half);
    MATCH(DT_BFLOAT16, half);
    MATCH(DT_STRING, string);
    case DT_COMPLEX64:
    case DT_COMPLEX128:
    default:
      return -1;
  }
}
absl::StatusOr<ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
                                                Builder* builder) {
  TensorShape input_tensor_shape(input_tensor.tensor_shape());
  if (NumberOfMaterializedElements(input_tensor) == 1 &&
      input_tensor_shape.num_elements() > 1) {
    TensorProto tensor_copy = input_tensor;
    auto* shape = tensor_copy.mutable_tensor_shape();
    shape->clear_dim();
    shape->add_dim()->set_size(1);
    TF_ASSIGN_OR_RETURN(ElementsAttr single_attr,
                        ConvertTensorProto(tensor_copy, builder));
    llvm::SmallVector<int64_t> original_dimensions;
    for (auto dim : input_tensor_shape) original_dimensions.push_back(dim.size);
    return ElementsAttr(mlir::SplatElementsAttr::get(
        single_attr.getShapedType().clone(original_dimensions),
        single_attr.getValues<mlir::Attribute>()[0]));
  }
  Tensor t;
  if (!t.FromProto(input_tensor))
    return InvalidArgument("Failed to parse input_tensor.");
  return ConvertTensor(t, builder);
}
void ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                               TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(ShapedType::isDynamic(d) ? kTFDynamicSize
                                                               : d);
  }
}
PartialTensorShape ConvertTypeToTensorShape(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    return PartialTensorShape();
  }
  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    TensorShapeProto tensor_shape_proto;
    ConvertToTensorShapeProto(tensor_type.getShape(), &tensor_shape_proto);
    return PartialTensorShape(tensor_shape_proto);
  }
  return TensorShape();
}
mlir::TF::ShapeAttr ConvertTypeToTensorShapeAttr(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    return mlir::TF::ShapeAttr::get(type.getContext(), std::nullopt);
  }
  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return mlir::TF::ShapeAttr::get(type.getContext(), tensor_type.getShape());
  }
  return mlir::TF::ShapeAttr::get(type.getContext(), ArrayRef<int64_t>());
}
absl::StatusOr<TensorSpecProto> ConvertTypeToTensorSpecProto(
    const mlir::Type& type) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  TensorSpecProto tensor_spec;
  tensor_spec.set_dtype(dtype);
  *tensor_spec.mutable_shape() = ConvertTypeToTensorShape(type).AsProto();
  return tensor_spec;
}
absl::StatusOr<mlir::Attribute> ConvertTensorShapeProto(
    const TensorShapeProto& shape, mlir::MLIRContext* context) {
  if (shape.unknown_rank())
    return mlir::TF::ShapeAttr::get(context, std::nullopt);
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(shape.dim().size());
  for (const auto& dim : shape.dim()) {
    dims.push_back(dim.size() == kTFDynamicSize ? ShapedType::kDynamic
                                                : dim.size());
  }
  return mlir::TF::ShapeAttr::get(context, llvm::ArrayRef(dims));
}
void ConvertStringElementsAttr(
    const DenseStringElementsAttr attr,
    protobuf::RepeatedPtrField<std::string>* output) {
  for (const auto& val : attr.getRawStringData())
    output->Add({val.data(), val.size()});
}
template <typename T>
void ConvertComplexElementsAttr(const mlir::DenseElementsAttr attr,
                                protobuf::RepeatedField<T>* output) {
  for (const auto& val : attr.getValues<std::complex<T>>()) {
    output->Add(val.real());
    output->Add(val.imag());
  }
}
Status ConvertTensorProtoAttr(const mlir::TF::TensorProtoAttr attr,
                              TensorProto* output_tensor) {
  auto mangled_tensor = attr.getValue();
  absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
  return mangling_util::DemangleTensor(tensor_view, output_tensor);
}
template <typename T>
void ConvertElementsAttr(const mlir::DenseElementsAttr attr,
                         protobuf::RepeatedField<T>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
  } else {
    output->Reserve(attr.getNumElements());
    for (auto value : attr.getValues<T>()) output->AddAlreadyReserved(value);
  }
}
template <typename T, typename Cord>
void ConvertFloatElementsAttr(const mlir::DenseElementsAttr attr,
                              protobuf::RepeatedField<T>* output,
                              Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}
void ConvertHalfElementsAttr(const mlir::DenseElementsAttr attr,
                             protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<Eigen::half>() != Eigen::half(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<Eigen::half>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const Eigen::half value : attr.getValues<Eigen::half>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
  }
}
template <typename T, typename U = T, typename Cord>
void ConvertIntElementsAttr(const mlir::DenseElementsAttr attr,
                            protobuf::RepeatedField<T>* output,
                            Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<U>() != U(0))
      output->Add(static_cast<T>(attr.getSplatValue<U>()));
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}
template <typename T, typename U = T, typename Cord>
void ConvertUIntElementsAttr(const mlir::DenseElementsAttr attr,
                             protobuf::RepeatedField<T>* output,
                             Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<U>() != U(0))
      output->Add(static_cast<T>(attr.getSplatValue<U>()));
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}
void ConvertBfloat16ElementsAttr(const mlir::DenseElementsAttr attr,
                                 protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<bfloat16>() != bfloat16(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<bfloat16>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const bfloat16 value : attr.getValues<bfloat16>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
  }
}
template <typename T>
void ConvertFloat8ElementsAttr(const mlir::DenseElementsAttr attr,
                               std::string* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0))
      output->push_back(
          Eigen::numext::bit_cast<uint8_t>(attr.getSplatValue<T>()));
  } else {
    output->reserve(attr.getNumElements());
    for (const T value : attr.getValues<T>())
      output->push_back(Eigen::numext::bit_cast<uint8_t>(value));
  }
}
Status ConvertToTensorProto(const ElementsAttr attr, TensorProto* output) {
  auto type = attr.getShapedType();
  auto shape = type.getShape();
  DataType output_dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &output_dtype));
  output->set_dtype(output_dtype);
  ConvertToTensorShapeProto(shape, output->mutable_tensor_shape());
  if (auto tensor_attr = mlir::dyn_cast<mlir::TF::TensorProtoAttr>(attr))
    return ConvertTensorProtoAttr(tensor_attr, output);
  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr) return errors::InvalidArgument("Unsupported elements attr");
  switch (output_dtype) {
    case DT_BOOL:
      ConvertElementsAttr(dense_attr, output->mutable_bool_val());
      break;
    case DT_BFLOAT16:
      ConvertBfloat16ElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_COMPLEX64:
      ConvertComplexElementsAttr(dense_attr, output->mutable_scomplex_val());
      break;
    case DT_COMPLEX128:
      ConvertComplexElementsAttr(dense_attr, output->mutable_dcomplex_val());
      break;
    case DT_DOUBLE:
      ConvertFloatElementsAttr(dense_attr, output->mutable_double_val(),
                               output->mutable_tensor_content());
      break;
    case DT_HALF:
      ConvertHalfElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_FLOAT:
      ConvertFloatElementsAttr(dense_attr, output->mutable_float_val(),
                               output->mutable_tensor_content());
      break;
    case DT_FLOAT8_E5M2:
      ConvertFloat8ElementsAttr<tsl::float8_e5m2>(dense_attr,
                                                  output->mutable_float8_val());
      break;
    case DT_FLOAT8_E4M3FN:
      ConvertFloat8ElementsAttr<tsl::float8_e4m3fn>(
          dense_attr, output->mutable_float8_val());
      break;
    case tensorflow::DT_INT4:
      ConvertIntElementsAttr<int, tsl::int4>(dense_attr,
                                             output->mutable_int_val(),
                                             output->mutable_tensor_content());
      break;
    case tensorflow::DT_UINT4:
      ConvertUIntElementsAttr<int, tsl::uint4>(
          dense_attr, output->mutable_int_val(),
          output->mutable_tensor_content());
      break;
    case DT_QUINT8:
    case DT_INT8:
      ConvertUIntElementsAttr<int, int8_t>(dense_attr,
                                           output->mutable_int_val(),
                                           output->mutable_tensor_content());
      break;
    case DT_QUINT16:
    case DT_INT16:
      ConvertIntElementsAttr<int, int16_t>(dense_attr,
                                           output->mutable_int_val(),
                                           output->mutable_tensor_content());
      break;
    case DT_INT32:
      ConvertIntElementsAttr(dense_attr, output->mutable_int_val(),
                             output->mutable_tensor_content());
      break;
    case DT_INT64:
      ConvertIntElementsAttr(dense_attr, output->mutable_int64_val(),
                             output->mutable_tensor_content());
      break;
    case DT_STRING:
      ConvertStringElementsAttr(mlir::cast<DenseStringElementsAttr>(dense_attr),
                                output->mutable_string_val());
      break;
    case DT_UINT8:
      ConvertUIntElementsAttr<int, uint8_t>(dense_attr,
                                            output->mutable_int_val(),
                                            output->mutable_tensor_content());
      break;
    case DT_UINT16:
      ConvertUIntElementsAttr<int, uint16_t>(dense_attr,
                                             output->mutable_int_val(),
                                             output->mutable_tensor_content());
      break;
    case DT_UINT32:
      ConvertUIntElementsAttr(dense_attr, output->mutable_uint32_val(),
                              output->mutable_tensor_content());
      break;
    case DT_UINT64:
      ConvertUIntElementsAttr(dense_attr, output->mutable_uint64_val(),
                              output->mutable_tensor_content());
      break;
    default:
      return errors::Unimplemented(absl::StrCat("Unimplemented data type ",
                                                DataTypeString(output_dtype)));
  }
  return absl::OkStatus();
}
Status ConvertToTensor(const mlir::ElementsAttr attr, Tensor* output_tensor) {
  TensorProto tensor_proto;
  TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor_proto));
  if (!output_tensor->FromProto(tensor_proto)) {
    return InvalidArgument("Couldn't convert tensor proto to tensor.");
  }
  return absl::OkStatus();
}
}  