#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "stablehlo/dialect/Base.h"
#include "xla/array.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/lib/approx_topk.h"
#include "xla/hlo/builder/lib/approx_topk_shape.h"
#include "xla/hlo/builder/lib/matrix.h"  
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/hlo/translate/mhlo_to_hlo/layout_util.h"
#include "xla/hlo/translate/mhlo_to_hlo/location_exporter.h"
#include "xla/hlo/translate/mhlo_to_hlo/module_attributes_exporter.h"
#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/types.h"
using ::int64_t;
using ::tsl::int16;
using ::tsl::int32;
using ::tsl::int8;
using ::tsl::StatusOr;  
using ::tsl::uint16;
using ::tsl::uint32;
using ::tsl::uint64;
using ::tsl::uint8;
constexpr char kJaxBufferDonor[] = "jax.buffer_donor";
constexpr char kResultLayout[] = "result_layout";
constexpr char kSourceLayout[] = "source_layout";
constexpr char kAggregateToTopk[] = "aggregate_to_topk";
constexpr char kApiVersion[] = "api_version";
constexpr char kApproxTopK[] = "ApproxTopK";
constexpr char kBackendConfig[] = "backend_config";
constexpr char kCallTargetName[] = "call_target_name";
constexpr char kCalledComputations[] = "called_computations";
constexpr char kHasSideEffect[] = "has_side_effect";
constexpr char kIsFallback[] = "is_fallback";
constexpr char kRecallTarget[] = "recall_target";
constexpr char kReductionDim[] = "reduction_dim";
constexpr char kReductionInputSizeOverride[] = "reduction_input_size_override";
constexpr char kTopK[] = "top_k";
constexpr char kMhloCrossProgramPrefetches[] = "mhlo.cross_program_prefetches";
constexpr char kMhloFrontendAttributes[] = "mhlo.frontend_attributes";
constexpr char kMhloInputOutputAlias[] = "mhlo.input_output_alias";
constexpr char kMhloIsDynamic[] = "mhlo.is_dynamic";
constexpr char kMhloLiteral[] = "mhlo.literal";
constexpr char kMhloParameterReplication[] = "mhlo.parameter_replication";
constexpr char kMhloReplication[] = "mhlo.is_same_data_across_replicas";
constexpr char kMhloSharding[] = "mhlo.sharding";
constexpr char kMhloSpmdOutputSharding[] = "mhlo.spmd_output_sharding";
constexpr char kMhloSpmdParametersShardings[] =
    "mhlo.spmd_parameters_shardings";
constexpr char kMhloUseAutoSpmdPartitioning[] =
    "mhlo.use_auto_spmd_partitioning";
constexpr char kMhloXlaEntryComputationParameterLayouts[] =
    "mhlo.xla_entry_computation_parameter_layouts";
constexpr char kMhloXlaEntryComputationParameterTiles[] =
    "mhlo.xla_entry_computation_parameter_tiles";
constexpr char kMhloXlaEntryComputationResultLayout[] =
    "mhlo.xla_entry_computation_result_layout";
constexpr char kMhloXlaEntryComputationResultTiles[] =
    "mhlo.xla_entry_computation_result_tiles";
constexpr char kArgEmptyTuple[] = "arg_empty_tuple";
constexpr char kArgPrefix[] = "Arg_";
constexpr char kArgTuple[] = "arg_tuple";
constexpr char kDefaultLayoutAttrName[] = "xla_shape";
constexpr char kExecutionThread[] = "execution_thread";
constexpr char kLayout[] = "layout";
constexpr char kMain[] = "main";
constexpr char kRegionPrefix[] = "region_";
constexpr char kTfAliasingOutput[] = "tf.aliasing_output";
template <typename T>
T Unwrap(T t) {
  return t;
}
template <typename T>
T* Unwrap(const std::unique_ptr<T>& t) {
  return t.get();
}
static mlir::LogicalResult GetXlaOp(
    mlir::Value val, const llvm::DenseMap<mlir::Value, xla::XlaOp>& val_map,
    xla::XlaOp* result, mlir::Operation* op) {
  auto iter = val_map.find(val);
  if (iter == val_map.end()) {
    return op->emitOpError(
        "requires all operands to be defined in the parent region for export");
  }
  *result = iter->second;
  return mlir::success();
}
bool IsBoundedOrStatic(mlir::Type ty) {
  auto ranked_ty = mlir::dyn_cast_or_null<mlir::RankedTensorType>(ty);
  if (!ranked_ty) return false;
  if (ranked_ty.hasStaticShape()) return true;
  auto encoding = mlir::dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>(
      ranked_ty.getEncoding());
  if (!encoding || encoding.getBounds().empty()) return false;
  int64_t rank = ranked_ty.getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (ranked_ty.isDynamicDim(dim) &&
        encoding.getBounds()[dim] == mlir::ShapedType::kDynamic)
      return false;
  }
  return true;
}
template <typename T>
xla::Array<T> ArrayFromDenseElementsAttr(mlir::DenseElementsAttr dense_attr) {
  constexpr xla::PrimitiveType type =
      xla::primitive_util::NativeToPrimitiveType<T>();
  xla::Shape shape = xla::TypeToShape(dense_attr.getType());
  xla::Array<T> array(shape.dimensions());
  if constexpr (!xla::primitive_util::IsSubByteNonPredType(type)) {
    array.SetValues(dense_attr.getValues<T>());
  } else {
    auto values = dense_attr.getValues<llvm::APInt>();
    for (int i = 0; i < values.size(); i++) {
      if constexpr (xla::primitive_util::IsUnsignedIntegralType(type)) {
        array.data()[i] = T{values[i].getZExtValue()};
      } else {
        static_assert(xla::primitive_util::IsSignedIntegralType(type));
        array.data()[i] = T{values[i].getSExtValue()};
      }
    }
  }
  return array;
}
absl::StatusOr<xla::Literal> CreateArrayLiteralFromAttr(mlir::ElementsAttr attr,
                                                        xla::Layout layout) {
  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr)
    return tsl::errors::Unimplemented("Only dense elements attr are supported");
  xla::Shape shape = xla::TypeToShape(dense_attr.getType());
  return xla::primitive_util::PrimitiveTypeSwitch<absl::StatusOr<xla::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<xla::Literal> {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using cpp_type =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          xla::Array<cpp_type> source_data =
              ArrayFromDenseElementsAttr<cpp_type>(dense_attr);
          return xla::LiteralUtil::CreateFromArrayWithLayout(source_data,
                                                             layout);
        }
        return tsl::errors::Internal(absl::StrCat(  
            "Unsupported type: ",
            xla::PrimitiveType_Name(shape.element_type())));
      },
      shape.element_type());
}
static int ConvertAPInt(llvm::APInt i) { return i.getSExtValue(); }
static uint32_t Convertuint32_t(uint32_t i) { return i; }
static uint64_t Convertuint64_t(uint64_t i) { return i; }
static double ConvertAPFloat(llvm::APFloat value) {
  const auto& semantics = value.getSemantics();
  bool losesInfo = false;
  if (&semantics != &llvm::APFloat::IEEEdouble())
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value.convertToDouble();
}
static inline bool Convertbool(bool value) { return value; }
static absl::string_view ConvertStringRef(mlir::StringRef value) {
  return {value.data(), value.size()};
}
static std::vector<int64_t> ConvertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}
static std::vector<int64_t> ConvertDenseIntAttr(
    std::optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}
static std::vector<int64_t> Convert_broadcast_dimensions(
    std::optional<mlir::DenseIntElementsAttr> broadcast_dimensions) {
  if (!broadcast_dimensions.has_value()) return {};
  return ConvertDenseIntAttr(*broadcast_dimensions);
}
static std::vector<xla::CrossProgramPrefetch> Convert_cross_program_prefetches(
    mlir::ArrayAttr prefetches) {
  std::vector<xla::CrossProgramPrefetch> cross_program_prefetches;
  for (auto prefetch : prefetches) {
    auto cpp = mlir::cast<mlir::mhlo::CrossProgramPrefetchAttr>(prefetch);
    xla::CrossProgramPrefetch xla_cpp;
    xla_cpp.set_parameter(cpp.getParameter());
    for (auto index : cpp.getIndices()) xla_cpp.add_index(index);
    cross_program_prefetches.push_back(xla_cpp);
  }
  return cross_program_prefetches;
}
static xla::FftType Convert_fft_type(mlir::mhlo::FftType fft_type) {
  xla::FftType fft_type_enum;
  if (!FftType_Parse(std::string(mlir::mhlo::stringifyFftType(fft_type)),
                     &fft_type_enum))
    return xla::FftType::FFT;
  return fft_type_enum;
}
static std::vector<std::pair<int64_t, int64_t>> Convert_padding(
    std::optional<mlir::DenseIntElementsAttr> padding) {
  return xla::ConvertNx2Attribute(padding).value();
}
static std::optional<bool> Convert_use_global_device_ids(
    std::optional<bool> use_global_device_ids) {
  if (!use_global_device_ids) return {};
  return *use_global_device_ids;
}
static std::vector<std::pair<int64_t, int64_t>> Convert_source_target_pairs(
    std::optional<mlir::DenseIntElementsAttr> source_target_pairs) {
  return xla::ConvertNx2Attribute(source_target_pairs).value();
}
static std::vector<xla::ReplicaGroup> Convert_replica_groups(
    mlir::DenseIntElementsAttr groups) {
  return xla::ConvertReplicaGroups(groups).value();
}
static void SetLayout(xla::Shape& shape, mlir::DenseIntElementsAttr layout) {
  if (shape.IsArray()) {
    shape.mutable_layout()->clear_minor_to_major();
    for (auto l : layout) {
      shape.mutable_layout()->mutable_minor_to_major()->push_back(
          l.getSExtValue());
    }
  } else if (shape.IsToken()) {
    assert(layout.empty() && "Invalid layout for token type");
  } else {
    assert(!shape.IsTuple() &&
           "Exporting layout for tuples is not implemented yet");
    assert(false && "Exporting unknown type with layout");
  }
}
static void SetLayout(xla::Shape& shape, mlir::ArrayAttr layouts) {
  if (shape.IsTuple()) {
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      SetLayout(*shape.mutable_tuple_shapes(i),
                mlir::cast<mlir::DenseIntElementsAttr>(layouts[i]));
    }
  } else {
    assert(layouts.size() == 1);
    SetLayout(shape, mlir::cast<mlir::DenseIntElementsAttr>(layouts[0]));
  }
}
static std::vector<xla::Shape> ConvertTypesToShapesWithLayout(
    mlir::TypeRange value_types, mlir::ArrayAttr layouts) {
  std::vector<xla::Shape> shapes_with_layout;
  for (auto [type, layout] : llvm::zip(value_types, layouts)) {
    xla::Shape shape = xla::TypeToShape(type);
    SetLayout(shape, mlir::cast<mlir::DenseIntElementsAttr>(layout));
    shapes_with_layout.push_back(std::move(shape));
  }
  return shapes_with_layout;
}
static xla::TriangularSolveOptions::Transpose Convert_transpose_a(
    mlir::mhlo::Transpose transpose) {
  return xla::ConvertTranspose(mlir::mhlo::stringifyTranspose(transpose))
      .value();
}
static xla::Layout ExtractLayout(
    mlir::Operation* op, int rank,
    llvm::StringRef attr_name = kDefaultLayoutAttrName) {
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    llvm::SmallVector<int64_t, 4> minor_to_major;
    DCHECK_EQ(rank, attr.size());
    minor_to_major.reserve(attr.size());
    for (const llvm::APInt& i : attr) {
      minor_to_major.push_back(i.getZExtValue());
    }
    return xla::LayoutUtil::MakeLayout(minor_to_major);
  }
  return xla::LayoutUtil::MakeDescendingLayout(rank);
}
static mlir::FailureOr<xla::Shape> ExtractXlaShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    return *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    std::vector<xla::Shape> subshapes;
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      subshapes.push_back(xla::TypeToShape(result.getType()));
      if (subshapes.back().element_type() == xla::PRIMITIVE_TYPE_INVALID) {
        return op->emitError()
               << "result #" << index << " type is not supported";
      }
    }
    if (subshapes.size() > 1) {
      return xla::ShapeUtil::MakeTupleShape(subshapes);
    }
    return subshapes[0];
  }
}
#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)               \
  static std::vector<int64_t> Convert_##attribute(           \
      std::optional<mlir::DenseIntElementsAttr> attribute) { \
    return ConvertDenseIntAttr(attribute);                   \
  }
I64_ELEMENTS_ATTR_TO_VECTOR(broadcast_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(permutation);
I64_ELEMENTS_ATTR_TO_VECTOR(start_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(limit_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(strides);
I64_ELEMENTS_ATTR_TO_VECTOR(slice_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(fft_length);
I64_ELEMENTS_ATTR_TO_VECTOR(dimensions);
I64_ELEMENTS_ATTR_TO_VECTOR(window_strides);
I64_ELEMENTS_ATTR_TO_VECTOR(lhs_dilation);
I64_ELEMENTS_ATTR_TO_VECTOR(rhs_dilation);
#undef I64_ELEMENTS_ATTR_TO_VECTOR
#define BOOL_ELEMENTS_ATTR_TO_VECTOR(attribute)           \
  static std::vector<bool> Convert_##attribute(           \
      std::optional<mlir::DenseElementsAttr> attribute) { \
    if (!attribute) return {};                            \
    auto values = attribute->getValues<bool>();           \
    return {values.begin(), values.end()};                \
  }
BOOL_ELEMENTS_ATTR_TO_VECTOR(window_reversal);
#undef BOOL_ELEMENTS_ATTR_TO_VECTOR
static std::vector<int64_t> Convert_ArrayRef(llvm::ArrayRef<int64_t> values) {
  return {values.begin(), values.end()};
}
static std::unique_ptr<xla::PrecisionConfig> Convert_precision_config(
    std::optional<mlir::ArrayAttr> optional_precision_config_attr) {
  if (!optional_precision_config_attr.has_value()) return nullptr;
  auto precision_config = std::make_unique<xla::PrecisionConfig>();
  for (auto attr : optional_precision_config_attr.value()) {
    xla::PrecisionConfig::Precision p;
    auto operand_precision =
        mlir::mhlo::stringifyPrecision(
            mlir::cast<mlir::mhlo::PrecisionAttr>(attr).getValue())
            .str();
    if (xla::PrecisionConfig::Precision_Parse(operand_precision, &p)) {
      precision_config->add_operand_precision(p);
    } else {
      auto* context = attr.getContext();
      mlir::emitError(mlir::UnknownLoc::get(context))
          << "unexpected operand precision " << operand_precision;
      return nullptr;
    }
  }
  return precision_config;
}
static xla::DotDimensionNumbers Convert_dot_dimension_numbers(
    mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers_attr) {
  xla::DotDimensionNumbers dot_dimension_numbers;
  auto rhs_contracting_dimensions =
      dot_dimension_numbers_attr.getRhsContractingDimensions();
  auto lhs_contracting_dimensions =
      dot_dimension_numbers_attr.getLhsContractingDimensions();
  auto rhs_batch_dimensions =
      dot_dimension_numbers_attr.getRhsBatchingDimensions();
  auto lhs_batch_dimensions =
      dot_dimension_numbers_attr.getLhsBatchingDimensions();
  for (const auto& val : rhs_contracting_dimensions) {
    dot_dimension_numbers.add_rhs_contracting_dimensions(val);
  }
  for (const auto& val : lhs_contracting_dimensions) {
    dot_dimension_numbers.add_lhs_contracting_dimensions(val);
  }
  for (const auto& val : rhs_batch_dimensions) {
    dot_dimension_numbers.add_rhs_batch_dimensions(val);
  }
  for (const auto& val : lhs_batch_dimensions) {
    dot_dimension_numbers.add_lhs_batch_dimensions(val);
  }
  return dot_dimension_numbers;
}
static xla::SparsityDescriptor Convert_sparsity_descriptor(
    mlir::mhlo::SparsityDescriptorAttr sparsity_attr, bool is_lhs) {
  xla::SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(xla::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_index(is_lhs ? 0 : 1);
  sparsity_descriptor.set_dimension(sparsity_attr.getDimension());
  sparsity_descriptor.set_n(sparsity_attr.getN());
  sparsity_descriptor.set_m(sparsity_attr.getM());
  return sparsity_descriptor;
}
xla::ChannelHandle Convert_channel_handle(mlir::mhlo::ChannelHandleAttr attr) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(attr.getHandle());
  channel_handle.set_type(
      static_cast<xla::ChannelHandle::ChannelType>(attr.getType()));
  return channel_handle;
}
std::optional<xla::ChannelHandle> Convert_channel_handle(
    std::optional<mlir::mhlo::ChannelHandleAttr> attr) {
  if (!attr.has_value()) return std::nullopt;
  return Convert_channel_handle(attr.value());
}
static xla::ComparisonDirection Convert_comparison_direction(
    llvm::StringRef comparison_direction_string) {
  return xla::StringToComparisonDirection(comparison_direction_string.str())
      .value();
}
static xla::GatherDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::GatherDimensionNumbersAttr input) {
  xla::GatherDimensionNumbers output;
  auto offset_dims = input.getOffsetDims();
  std::copy(
      offset_dims.begin(), offset_dims.end(),
      tsl::protobuf::RepeatedFieldBackInserter(output.mutable_offset_dims()));
  auto collapsed_slice_dims = input.getCollapsedSliceDims();
  std::copy(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_collapsed_slice_dims()));
  auto operand_batching_dims = input.getOperandBatchingDims();
  std::copy(operand_batching_dims.begin(), operand_batching_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_operand_batching_dims()));
  auto start_indices_batching_dims = input.getStartIndicesBatchingDims();
  std::copy(start_indices_batching_dims.begin(),
            start_indices_batching_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_start_indices_batching_dims()));
  auto start_index_map = input.getStartIndexMap();
  std::copy(start_index_map.begin(), start_index_map.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_start_index_map()));
  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}
static xla::ScatterDimensionNumbers Convert_scatter_dimension_numbers(
    mlir::mhlo::ScatterDimensionNumbersAttr input) {
  xla::ScatterDimensionNumbers output;
  auto update_window_dims = input.getUpdateWindowDims();
  std::copy(update_window_dims.begin(), update_window_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_update_window_dims()));
  auto inserted_window_dims = input.getInsertedWindowDims();
  std::copy(inserted_window_dims.begin(), inserted_window_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_inserted_window_dims()));
  auto input_batching_dims = input.getInputBatchingDims();
  std::copy(input_batching_dims.begin(), input_batching_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_input_batching_dims()));
  auto scatter_indices_batching_dims = input.getScatterIndicesBatchingDims();
  std::copy(scatter_indices_batching_dims.begin(),
            scatter_indices_batching_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_indices_batching_dims()));
  auto scatter_dims_to_operand_dims = input.getScatterDimsToOperandDims();
  std::copy(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_dims_to_operand_dims()));
  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}
static std::optional<xla::OpSharding> CreateOpShardingFromAttribute(
    mlir::Operation* op) {
  auto shardingAttr = op->getAttrOfType<mlir::StringAttr>(kMhloSharding);
  if (!shardingAttr) return std::nullopt;
  return xla::ConvertSharding(shardingAttr.getValue());
}
void ConstructFrontendAttributesFromAttribute(
    const mlir::DictionaryAttr& frontend_attributes_dict,
    xla::FrontendAttributes& frontend_attributes) {
  for (const auto& attr : frontend_attributes_dict)
    if (auto value_str_attr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
      frontend_attributes.mutable_map()->insert(
          {attr.getName().str(), value_str_attr.getValue().str()});
}
static xla::FrontendAttributes CreateXlaFrontendAttributesFromOp(
    mlir::Operation* op) {
  xla::FrontendAttributes frontend_attributes;
  auto frontend_attributes_dict =
      op->getAttrOfType<mlir::DictionaryAttr>(kMhloFrontendAttributes);
  if (!frontend_attributes_dict) return frontend_attributes;
  ConstructFrontendAttributesFromAttribute(frontend_attributes_dict,
                                           frontend_attributes);
  return frontend_attributes;
}
static void ExtractFrontendAttributesFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<xla::FrontendAttributes>>* fe_attrs) {
  fe_attrs->resize(function.getNumArguments(), std::nullopt);
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto fe_attr = function.getArgAttrOfType<mlir::DictionaryAttr>(
            i, kMhloFrontendAttributes)) {
      xla::FrontendAttributes frontend_attributes;
      ConstructFrontendAttributesFromAttribute(fe_attr, frontend_attributes);
      (*fe_attrs)[i] = frontend_attributes;
    }
}
static bool SomeOptionalShardingsAreSet(
    llvm::ArrayRef<std::optional<xla::OpSharding>> shardings) {
  return llvm::any_of(shardings,
                      [](const std::optional<xla::OpSharding>& sharding) {
                        return sharding.has_value();
                      });
}
static void ExtractShardingsFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<xla::OpSharding>>* arg_shardings,
    llvm::SmallVectorImpl<std::optional<xla::OpSharding>>* ret_shardings) {
  arg_shardings->resize(function.getNumArguments(),
                        std::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto sharding =
            function.getArgAttrOfType<mlir::StringAttr>(i, kMhloSharding))
      (*arg_shardings)[i] = xla::ConvertSharding(sharding.getValue());
  ret_shardings->resize(function.getNumResults(),
                        std::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumResults(); i < end; ++i)
    if (auto sharding =
            function.getResultAttrOfType<mlir::StringAttr>(i, kMhloSharding))
      (*ret_shardings)[i] = xla::ConvertSharding(sharding.getValue());
}
std::optional<xla::OpSharding> CreateTupleSharding(
    llvm::ArrayRef<std::optional<xla::OpSharding>> tuple_shardings) {
  if (tuple_shardings.empty() ||
      !SomeOptionalShardingsAreSet(tuple_shardings)) {
    return std::nullopt;
  }
  xla::OpSharding sharding;
  sharding.set_type(xla::OpSharding::TUPLE);
  for (const std::optional<xla::OpSharding>& tuple_sharding : tuple_shardings) {
    if (tuple_sharding) {
      *sharding.add_tuple_shardings() = *tuple_sharding;
    } else {
      xla::OpSharding fallback_sharding;
      fallback_sharding.set_type(xla::OpSharding::REPLICATED);
      *sharding.add_tuple_shardings() = fallback_sharding;
    }
  }
  return sharding;
}
xla::XlaOp CreateTupleIfMultipleOps(
    xla::XlaBuilder* builder, llvm::ArrayRef<xla::XlaOp> ops,
    llvm::ArrayRef<std::optional<xla::OpSharding>> shardings) {
  if (ops.size() == 1) {
    return ops[0];
  }
  xla::XlaScopedShardingAssignment scoped_sharding(
      builder, CreateTupleSharding(shardings));
  return Tuple(builder, ops);
}
llvm::SmallVector<std::optional<xla::OpSharding>> GetResultShardings(
    std::optional<xla::OpSharding> op_sharding, int64_t num_results) {
  if (!op_sharding) {
    return {};
  }
  llvm::SmallVector<std::optional<xla::OpSharding>> res_shardings;
  res_shardings.reserve(num_results);
  if (op_sharding->type() == xla::OpSharding::TUPLE) {
    assert(op_sharding->tuple_shardings_size() == num_results);
    res_shardings.assign(op_sharding->tuple_shardings().begin(),
                         op_sharding->tuple_shardings().end());
  } else {
    res_shardings.append(num_results, op_sharding);
  }
  return res_shardings;
}
llvm::SmallVector<std::optional<xla::OpSharding>> GetXlaOpShardings(
    llvm::ArrayRef<xla::XlaOp> xla_ops) {
  llvm::SmallVector<std::optional<xla::OpSharding>> shardings;
  shardings.reserve(xla_ops.size());
  for (const xla::XlaOp& xla_op : xla_ops) {
    auto sharding = xla_op.builder()->GetOpSharding(xla_op);
    assert(sharding.ok() && "can't find XlaOp for argument");
    shardings.push_back(*sharding);
  }
  return shardings;
}
namespace mlir {
namespace {
class ConvertToHloModule {
 public:
  using ValueLoweringMap = llvm::DenseMap<Value, xla::XlaOp>;
  using FunctionLoweringMap =
      llvm::DenseMap<mlir::func::FuncOp, xla::XlaComputation>;
  explicit ConvertToHloModule(mlir::ModuleOp module,
                              xla::XlaBuilder& module_builder,
                              MlirToHloConversionOptions options)
      : module_(module), module_builder_(module_builder), options_(options) {}
  LogicalResult Run() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>(kMain);
    if (!main)
      return module_.emitError(
          "conversion requires module with `main` function");
    for (auto func : module_.getOps<func::FuncOp>()) {
      if (func.empty()) continue;
      if (failed(RunOnFunction(func))) return failure();
    }
    return success();
  }
  LogicalResult RunOnFunction(mlir::func::FuncOp f);
  ::xla::HloModuleProto ConsumeMainProto() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>(kMain);
    CHECK(main) << "requires module to have main function";  
    return lowered_computation_[main].proto();
  }
  LogicalResult LowerRegionAsComputation(
      mlir::Region* region, xla::XlaComputation* func,
      llvm::ArrayRef<mlir::Value> implicit_operands = {},
      llvm::ArrayRef<mlir::Value> implicit_results = {},
      bool ensure_single_arg = false,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings = {},
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings = {});
  LogicalResult LowerBasicBlockAsFunction(
      Block* block, xla::XlaBuilder* builder, bool is_entry_function,
      bool ensure_single_arg,
      const std::vector<bool>& entry_args_same_across_replicas,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
      llvm::ArrayRef<std::optional<xla::FrontendAttributes>> fe_attrs,
      xla::XlaComputation* result,
      llvm::ArrayRef<mlir::Value> implicit_operands = {},
      llvm::ArrayRef<mlir::Value> implicit_results = {});
  LogicalResult LowerCast(mlir::Operation* inst,
                          const MlirToHloConversionOptions& options,
                          ConvertToHloModule::ValueLoweringMap* value_lowering);
  LogicalResult LowerCompositeCall(
      mlir::Operation* inst, xla::XlaBuilder* module_builder,
      xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      xla::XlaOp* return_value);
  LogicalResult LowerConstant(
      mlir::Operation* inst, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      ElementsAttr const_attr);
  LogicalResult LowerFunctionCall(
      mlir::func::CallOp call_op, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);
  LogicalResult LowerInfeed(
      mlir::Operation* inst, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);
  LogicalResult LowerReturn(
      Operation* inst, bool is_entry_function,
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
      llvm::ArrayRef<mlir::Value> implicit_results, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      xla::XlaOp* return_value, const MlirToHloConversionOptions& options);
  LogicalResult PropagateLayouts(const MlirToHloConversionOptions& options,
                                 mlir::Operation* inst, xla::XlaOp xla_op);
  func::FuncOp LookUpSymbol(FlatSymbolRefAttr symbol) {
    return module_.lookupSymbol<mlir::func::FuncOp>(symbol);
  }
  xla::XlaComputation& GetLoweredComputation(func::FuncOp func) {
    return lowered_computation_[func];
  }
  LogicalResult Lower(
      mlir::Operation* inst, bool is_entry_function,
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
      llvm::ArrayRef<mlir::Value> implicit_results, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      xla::XlaOp* return_value);
  const MlirToHloConversionOptions& GetOptions() const { return options_; }
  xla::StackFrameIndexProto BuildStackFramesIndexProto() {
    return stack_frame_indexes_builder_.Build();
  }
 private:
  LogicalResult SetEntryTupleShapesAndLeafReplication(
      Block* block, const std::vector<bool>& entry_args_same_across_replicas,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
      std::vector<bool>* leaf_replication);
  LogicalResult SetEntryTupleShardings(
      Block* block, xla::XlaBuilder* builder,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes);
  mlir::ModuleOp module_;
  xla::XlaBuilder& module_builder_;
  mlir::StackFrameIndexBuilder stack_frame_indexes_builder_;
  FunctionLoweringMap lowered_computation_;
  size_t region_id_ = 0;
  MlirToHloConversionOptions options_;
};
}  
}  
namespace {
struct OpLoweringContext {
  llvm::DenseMap<mlir::Value, xla::XlaOp>* values;
  mlir::ConvertToHloModule* converter;
  xla::XlaBuilder* builder;
  mlir::StackFrameIndexBuilder* frame_index_builder;
};
mlir::LogicalResult GetTuple(mlir::Operation* op,
                             mlir::Operation::operand_range values,
                             OpLoweringContext ctx,
                             llvm::SmallVectorImpl<xla::XlaOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetXlaOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}
mlir::LogicalResult GetXlaOps(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> values,
                              OpLoweringContext ctx,
                              llvm::SmallVectorImpl<xla::XlaOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetXlaOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}
bool SimplyReturnedOp(mlir::Operation* op) {
  for (auto operand : op->getOperands()) {
    if (!llvm::isa<mlir::BlockArgument>(operand)) return false;
  }
  auto users = op->getUsers();
  if (users.empty()) return false;
  auto first_user = *users.begin();
  for (auto user : users) {
    if (first_user != user) return false;
  }
  if (llvm::isa<mlir::func::ReturnOp>(first_user)) return true;
  return false;
}
void BuildGetTupleElementsForTupleResults(mlir::Operation* op, xla::XlaOp tuple,
                                          OpLoweringContext ctx,
                                          unsigned num_implicit_results = 0) {
  const std::optional<xla::OpSharding>& sharding = ctx.builder->sharding();
  if (sharding.has_value()) {
    bool is_tuple_sharding = sharding->type() == xla::OpSharding::TUPLE;
    assert(!is_tuple_sharding || (op->getNumResults() + num_implicit_results ==
                                  sharding->tuple_shardings_size()));
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      xla::XlaScopedShardingAssignment scoped_sharding(
          ctx.builder,
          is_tuple_sharding ? sharding->tuple_shardings(index) : sharding);
      (*ctx.values)[result] = xla::GetTupleElement(tuple, index);
    }
  } else {
    xla::XlaScopedShardingAssignment scoped_sharding(ctx.builder, std::nullopt);
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      (*ctx.values)[result] = xla::GetTupleElement(tuple, index);
    }
  }
}
}  
namespace mlir {
namespace mhlo {
namespace {
LogicalResult ExportXlaOp(CollectiveBroadcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op->getResult(0)] = xla::CollectiveBroadcast(
      operand, Convert_replica_groups(op.getReplicaGroups()),
      Convert_channel_handle(op.getChannelHandle()));
  return success();
}
LogicalResult ExportXlaOp(CompositeOp, OpLoweringContext) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicBroadcastInDimOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicConvOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicGatherOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicIotaOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicPadOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(DynamicReshapeOp op, OpLoweringContext ctx) {
  auto resultType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType) return op->emitOpError() << "expected ranked result";
  auto resultBounds = hlo::encodingToBounds(resultType.getEncoding());
  if (resultBounds.empty())
    return op->emitOpError() << "expected bounded result";
  auto shapeType =
      mlir::dyn_cast<RankedTensorType>(op.getOutputShape().getType());
  if (!shapeType || !shapeType.getElementType().isInteger(32))
    return op->emitOpError() << "expected output shape to be tensor<Nxi32>";
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  xla::XlaOp outputShape;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getOutputShape(), value_map, &outputShape, op)))
    return failure();
  SmallVector<xla::XlaOp> dimSizes;
  SmallVector<int64_t> newSizeBounds;
  std::vector<bool> dimsAreDynamic;
  for (auto i = 0; i < resultType.getRank(); ++i) {
    auto runtimeSizeX1 = xla::Slice(outputShape, {i}, {i + 1}, {1});
    dimSizes.push_back(xla::Reshape(runtimeSizeX1, {}));
    auto dimSize = resultType.getDimSize(i);
    auto dimBound = resultBounds[i];
    if (!hlo::isStaticDimSize(dimSize) && !hlo::isStaticDimSize(dimBound))
      return op->emitOpError() << "unbounded dynamism is not supported";
    newSizeBounds.push_back(hlo::isStaticDimSize(dimSize) ? dimSize : dimBound);
    dimsAreDynamic.push_back(!hlo::isStaticDimSize(dimSize));
  }
  value_map[op] =
      xla::DynamicReshape(operand, dimSizes, newSizeBounds, dimsAreDynamic);
  return success();
}
LogicalResult ExportXlaOp(RealDynamicSliceOp op, OpLoweringContext ctx) {
  return failure();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CopyOp op, OpLoweringContext ctx) {
  if (op.getCrossProgramPrefetchIndex() && !SimplyReturnedOp(op))
    return op->emitOpError() << "synchronous CopyOp should not include "
                                "cross_program_prefetch_index attribute.";
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(
          GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Copy(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(AddDependencyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  auto operand_shape = ctx.builder->GetShape(operand).value();
  value_map[op] = xla::internal::XlaBuilderFriend::BuildAddDependency(
      ctx.builder, operand, token, operand_shape);
  return success();
}
LogicalResult ExportXlaOp(AllGatherOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands))) {
    return failure();
  }
  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();
  auto all_gather_dim = op.getAllGatherDim();
  int64_t shard_count = 0;
  for (size_t i = 0; i < operands.size(); ++i) {
    TensorType operand_type =
        mlir::cast<TensorType>(op.getOperand(i).getType());
    TensorType result_type = mlir::cast<TensorType>(op.getType(i));
    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();
    if (i == 0) {
      shard_count = result_type.getDimSize(all_gather_dim) /
                    operand_type.getDimSize(all_gather_dim);
    }
  }
  if (shape_or->IsTuple()) {
    std::optional<xla::Layout> layout = std::nullopt;
    if (shape_or->has_layout()) {
      layout = shape_or->layout();
    }
    auto tuple = xla::AllGatherTuple(
        operands, all_gather_dim, shard_count,
        Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), layout,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    value_map[op->getResults()[0]] = xla::AllGather(
        operands[0], all_gather_dim, shard_count,
        Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  }
  return success();
}
LogicalResult ExportXlaOp(AllReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }
  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands)))
    return failure();
  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();
  if (shape_or->IsTuple()) {
    std::optional<xla::Shape> shape_with_layout = std::nullopt;
    if (shape_or->has_layout()) shape_with_layout = shape_or.value();
    auto tuple = xla::AllReduceTuple(
        operands, computation, Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), shape_with_layout,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    value_map[op->getResults()[0]] = xla::AllReduce(
        operands[0], computation, Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  }
  return success();
}
LogicalResult ExportXlaOp(AllToAllOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands))) {
    return failure();
  }
  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();
  if (shape_or->IsTuple()) {
    std::optional<xla::Layout> layout = std::nullopt;
    if (shape_or->has_layout()) {
      layout = shape_or->layout();
    }
    auto tuple = xla::AllToAllTuple(
        operands, Convert_replica_groups(op.getReplicaGroups()), layout,
        Convert_channel_handle(op.getChannelHandle()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    std::optional<uint64_t> splitDimension = op.getSplitDimension();
    std::optional<uint64_t> concatDimension = op.getConcatDimension();
    std::optional<uint64_t> splitCount = op.getSplitCount();
    value_map[op->getResults()[0]] = xla::AllToAll(
        operands[0], *splitDimension, *concatDimension, *splitCount,
        Convert_replica_groups(op.getReplicaGroups()),
        std::nullopt, Convert_channel_handle(op.getChannelHandle()));
  }
  return success();
}
LogicalResult ExportXlaOp(ReduceScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  TensorType operand_type = mlir::cast<TensorType>(op.getOperand().getType());
  TensorType result_type = op.getType();
  if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
    return failure();
  auto scatter_dim = op.getScatterDimension();
  int64_t shard_count = operand_type.getDimSize(scatter_dim) /
                        result_type.getDimSize(scatter_dim);
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }
  value_map[op] = xla::ReduceScatter(
      operand, computation, scatter_dim, shard_count,
      Convert_replica_groups(op.getReplicaGroups()),
      Convert_channel_handle(op.getChannelHandle()), std::nullopt,
      Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  return success();
}
LogicalResult ExportXlaOp(AsyncStartOp op, OpLoweringContext ctx) {
  for (auto* user : op.getResult().getUsers()) {
    if (!isa<AsyncUpdateOp, AsyncDoneOp>(user)) {
      return op.emitOpError() << "Users of AsyncStart's return value must be "
                              << "async_update or async_done";
    }
  }
  auto& value_map = *ctx.values;
  Value result = op.getResult();
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
      FlatSymbolRefAttr::get(op->getContext(), op.getCalledComputation()));
  auto all_gather_op =
      dyn_cast_or_null<AllGatherOp>(callee.getBody().front().front());
  if (all_gather_op && SimplyReturnedOp(all_gather_op)) {
    TensorType operand_type =
        mlir::cast<TensorType>(all_gather_op.getOperand(0).getType());
    TensorType result_type = mlir::cast<TensorType>(all_gather_op.getType(0));
    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();
    if (operands.size() != 1) return failure();
    auto all_gather_dim = all_gather_op.getAllGatherDim();
    int64_t shard_count = result_type.getDimSize(all_gather_dim) /
                          operand_type.getDimSize(all_gather_dim);
    value_map[result] = xla::internal::XlaBuilderFriend::BuildAllGatherStart(
        ctx.builder, operands[0], all_gather_dim, shard_count,
        Convert_replica_groups(all_gather_op.getReplicaGroups()),
        Convert_channel_handle(all_gather_op.getChannelHandle()),
        ExtractLayout(all_gather_op,
                      mlir::cast<RankedTensorType>(result_type).getRank()),
        Convert_use_global_device_ids(all_gather_op.getUseGlobalDeviceIds()));
    return success();
  }
  auto all_reduce_op =
      dyn_cast_or_null<AllReduceOp>(callee.getBody().front().front());
  if (all_reduce_op && SimplyReturnedOp(all_reduce_op)) {
    xla::XlaComputation computation;
    if (failed(ctx.converter->LowerRegionAsComputation(
            &all_reduce_op.getComputation(), &computation))) {
      return failure();
    }
    if (operands.size() != 1) return failure();
    value_map[result] = xla::internal::XlaBuilderFriend::BuildAllReduceStart(
        ctx.builder, operands[0], computation,
        Convert_replica_groups(all_reduce_op.getReplicaGroups()),
        Convert_channel_handle(all_reduce_op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(all_reduce_op.getUseGlobalDeviceIds()));
    return success();
  }
  auto collective_permute_op =
      dyn_cast_or_null<CollectivePermuteOp>(callee.getBody().front().front());
  if (collective_permute_op && SimplyReturnedOp(collective_permute_op)) {
    value_map[result] =
        xla::internal::XlaBuilderFriend::BuildCollectivePermuteStart(
            ctx.builder, operands[0],
            Convert_source_target_pairs(
                collective_permute_op.getSourceTargetPairs()),
            Convert_channel_handle(collective_permute_op.getChannelHandle()));
    return mlir::success();
  }
  auto copy_op = dyn_cast_or_null<CopyOp>(callee.getBody().front().front());
  if (copy_op && SimplyReturnedOp(copy_op)) {
    std::optional<int> cross_program_prefetch_index =
        copy_op.getCrossProgramPrefetchIndex()
            ? std::make_optional(*copy_op.getCrossProgramPrefetchIndex())
            : std::nullopt;
    value_map[result] = xla::internal::XlaBuilderFriend::BuildCopyStart(
        ctx.builder, operands[0], cross_program_prefetch_index);
    return mlir::success();
  }
  auto send_op = dyn_cast_or_null<SendOp>(callee.getBody().front().front());
  if (send_op && SimplyReturnedOp(send_op)) {
    xla::XlaOp operand;
    if (operands.size() == 2)
      operand = operands[0];
    else
      operand =
          Tuple(ctx.builder, absl::Span<const xla::XlaOp>(operands).subspan(
                                 0, operands.size() - 1));
    xla::XlaOp token = operands[operands.size() - 1];
    value_map[result] = xla::internal::XlaBuilderFriend::BuildSend(
        ctx.builder, operand, token,
        Convert_channel_handle(send_op.getChannelHandle()),
        send_op.getIsHostTransfer());
    return mlir::success();
  }
  auto recv_op = dyn_cast_or_null<RecvOp>(callee.getBody().front().front());
  if (recv_op && SimplyReturnedOp(recv_op)) {
    auto result_types =
        mlir::cast<AsyncBundleType>(result.getType()).getTypes()[1];
    mlir::Type received_type = mlir::TupleType::get(op->getContext(), {});
    if (isa<TupleType>(result_types)) {
      received_type = mlir::cast<TupleType>(result_types).getType(0);
    }
    value_map[result] = xla::internal::XlaBuilderFriend::BuildRecv(
        ctx.builder, operands[0], xla::TypeToShape(received_type),
        Convert_channel_handle(recv_op.getChannelHandle()),
        recv_op.getIsHostTransfer());
    return mlir::success();
  }
  if (failed(ctx.converter->RunOnFunction(callee))) return failure();
  xla::XlaComputation& computation =
      ctx.converter->GetLoweredComputation(callee);
  computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
      op.getExecutionThread().str());
  auto [xla_op, computation_id] =
      xla::internal::XlaBuilderFriend::BuildAsyncStart(
          ctx.builder, operands, op.getExecutionThread().str(), computation,
          xla::TypeToShape(result.getType()));
  value_map[result] = xla_op;
  computation.mutable_proto()->mutable_computations(0)->set_id(computation_id);
  return success();
}
LogicalResult ExportXlaOp(AsyncUpdateOp op, OpLoweringContext ctx) {
  if (!isa<AsyncStartOp, AsyncUpdateOp>(op.getBundle().getDefiningOp())) {
    auto theerror = op.emitError()
                    << "Defining op of AsyncUpdate's operand must be "
                    << "async_start or async_update";
    if (op.getBundle().getDefiningOp()) {
      return theerror << ", but got "
                      << op.getBundle().getDefiningOp()->getName();
    } else {
      return theerror << ".";
    }
  }
  for (auto* user : op.getResult().getUsers()) {
    if (!isa<AsyncUpdateOp, AsyncDoneOp>(user)) {
      return op.emitOpError() << "Users of AsyncUpdate's return value must be "
                              << "async_update or async_done";
    }
  }
  auto& value_map = *ctx.values;
  Value result = op.getResult();
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getBundle(), value_map, &operand, op)))
    return failure();
  value_map[result] = xla::internal::XlaBuilderFriend::BuildAsyncUpdate(
      ctx.builder, operand, xla::TypeToShape(result.getType()));
  return success();
}
LogicalResult ExportXlaOp(AsyncDoneOp op, OpLoweringContext ctx) {
  if (!isa<AsyncStartOp, AsyncUpdateOp>(op.getBundle().getDefiningOp())) {
    auto theerror = op.emitError()
                    << "Defining op of AsyncDone's operand must be "
                    << "async_start or async_update";
    if (op.getBundle().getDefiningOp())
      return theerror << ", but got "
                      << op.getBundle().getDefiningOp()->getName();
    return theerror << ".";
  }
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getBundle(), value_map, &operand, op)))
    return failure();
  Operation* start = op;
  while (start != nullptr && !isa<AsyncStartOp>(start)) {
    start = start->getOperand(0).getDefiningOp();
    if (start == nullptr || !isa<AsyncStartOp, AsyncUpdateOp>(start)) {
      return op.emitError() << "Defining op of AsyncDone's operand must be "
                            << "async_start or async_update";
    }
  }
  if (!isa<AsyncStartOp>(start)) {
    return op.emitError() << "Could not find async chain start";
  }
  mlir::func::FuncOp callee =
      ctx.converter->LookUpSymbol(FlatSymbolRefAttr::get(
          op->getContext(), cast<AsyncStartOp>(start).getCalledComputation()));
  auto all_gather_op =
      dyn_cast_or_null<AllGatherOp>(callee.getBody().front().front());
  if (all_gather_op && SimplyReturnedOp(all_gather_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildAllGatherDone(
            ctx.builder, operand, xla::TypeToShape(all_gather_op.getType(0)));
    return success();
  }
  auto all_reduce_op =
      dyn_cast_or_null<AllReduceOp>(callee.getBody().front().front());
  if (all_reduce_op && SimplyReturnedOp(all_reduce_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildAllReduceDone(
            ctx.builder, operand, xla::TypeToShape(all_reduce_op.getType(0)));
    return success();
  }
  auto collective_permute_op =
      dyn_cast_or_null<CollectivePermuteOp>(callee.getBody().front().front());
  if (collective_permute_op && SimplyReturnedOp(collective_permute_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildCollectivePermuteDone(
            ctx.builder, operand,
            xla::TypeToShape(collective_permute_op.getType()));
    return success();
  }
  auto copy_op = dyn_cast_or_null<CopyOp>(callee.getBody().front().front());
  if (copy_op && SimplyReturnedOp(copy_op)) {
    value_map[op.getResult(0)] = xla::internal::XlaBuilderFriend::BuildCopyDone(
        ctx.builder, operand, xla::TypeToShape(copy_op.getType()));
    return success();
  }
  auto send_op = dyn_cast_or_null<SendOp>(callee.getBody().front().front());
  if (send_op && SimplyReturnedOp(send_op)) {
    value_map[op.getResult(0)] = xla::internal::XlaBuilderFriend::BuildSendDone(
        ctx.builder, operand,
        Convert_channel_handle(send_op.getChannelHandle()),
        send_op.getIsHostTransfer());
    return success();
  }
  auto recv_op = dyn_cast_or_null<RecvOp>(callee.getBody().front().front());
  if (recv_op && SimplyReturnedOp(recv_op)) {
    auto result_types =
        mlir::cast<AsyncBundleType>(op.getBundle().getType()).getTypes()[1];
    mlir::Type received_type = mlir::TupleType::get(op->getContext(), {});
    if (isa<TupleType>(result_types)) {
      received_type = mlir::cast<TupleType>(result_types).getType(0);
    }
    xla::XlaOp xla_recv = xla::internal::XlaBuilderFriend::BuildRecvDone(
        ctx.builder, operand, xla::TypeToShape(received_type),
        Convert_channel_handle(recv_op.getChannelHandle()),
        recv_op.getIsHostTransfer());
    if (op.getNumResults() == 1) {
      value_map[op.getResult(0)] = xla_recv;
    } else {
      BuildGetTupleElementsForTupleResults(op, xla_recv, ctx);
    }
    return success();
  }
  std::vector<xla::Shape> subshapes;
  for (const auto& item : op.getResults().getType()) {
    subshapes.push_back(xla::TypeToShape(item));
  }
  xla::Shape data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  xla::XlaOp exportedOp = xla::internal::XlaBuilderFriend::BuildAsyncDone(
      ctx.builder, operand, data_shape);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = exportedOp;
  } else {
    BuildGetTupleElementsForTupleResults(op, exportedOp, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(BitcastConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op] = xla::BitcastConvertType(
      operand,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}
LogicalResult ExportXlaOp(BroadcastInDimOp op, OpLoweringContext ctx) {
  auto type = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!type) return failure();
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op] =
      BroadcastInDim(operand, Convert_ArrayRef(type.getShape()),
                     Convert_broadcast_dimensions(op.getBroadcastDimensions()));
  return success();
}
LogicalResult ExportXlaOp(StochasticConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, random;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getRandom(), value_map, &random, op)))
    return failure();
  value_map[op] = xla::StochasticConvertType(
      operand, random,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}
LogicalResult ExportXlaOp(CosineOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Cos(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(TanOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Tan(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(DotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  value_map[op] = xla::Dot(
      lhs, rhs, Unwrap(Convert_precision_config(op.getPrecisionConfig())),
      preferred_element_type);
  return mlir::success();
}
LogicalResult ExportXlaOp(DotGeneralOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  auto precision_config = Convert_precision_config(op.getPrecisionConfig());
  if (op.getAlgorithmAttr()) {
    absl::StatusOr<xla::PrecisionConfig::Algorithm> algorithm =
        xla::ConvertDotAlgorithm(op.getAlgorithmAttr());
    if (!algorithm.ok()) {
      return op.emitError(algorithm.status().ToString());
    }
    precision_config->set_algorithm(algorithm.value());
  }
  auto xlaOp = xla::DotGeneral(
      lhs, rhs, Convert_dot_dimension_numbers(op.getDotDimensionNumbers()),
      Unwrap(precision_config), preferred_element_type);
  value_map[op] = xlaOp;
  return mlir::success();
}
LogicalResult ExportXlaOp(SparseDotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  llvm::SmallVector<xla::XlaOp> sparse_meta;
  if (failed(GetTuple(op, op.getMeta(), ctx, sparse_meta))) return failure();
  std::vector<xla::SparsityDescriptor> sparsity;
  if (op.getLhsSparsity().has_value()) {
    sparsity.push_back(
        Convert_sparsity_descriptor(*op.getLhsSparsity(), true));
  }
  if (op.getRhsSparsity().has_value()) {
    sparsity.push_back(
        Convert_sparsity_descriptor(*op.getRhsSparsity(), false));
  }
  value_map[op] =
      xla::SparseDot(lhs, rhs, absl::MakeSpan(sparse_meta), sparsity,
                     Convert_dot_dimension_numbers(op.getDotDimensionNumbers()),
                     Unwrap(Convert_precision_config(op.getPrecisionConfig())),
                     preferred_element_type);
  return mlir::success();
}
LogicalResult ExportXlaOp(DomainOp op, OpLoweringContext ctx) {
  auto& valueMap = *ctx.values;
  xla::Shape shape = xla::TypeToShape(op.getResult().getType());
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), valueMap, &operand, op)))
    return failure();
  auto entry = xla::ConvertSharding(op.getEntryMetadata());
  if (!entry) return failure();
  auto exit = xla::ConvertSharding(op.getExitMetadata());
  if (!exit) return failure();
  valueMap[op] = xla::internal::XlaBuilderFriend::BuildDomain(
      ctx.builder, operand, *exit, *entry, shape);
  return success();
}
LogicalResult ExportXlaOp(IfOp op, OpLoweringContext ctx) {
  xla::XlaComputation true_branch;
  xla::XlaComputation false_branch;
  auto& value_map = *ctx.values;
  llvm::SetVector<mlir::Value> implicit_true_operand_set,
      implicit_false_operand_set;
  getUsedValuesDefinedAbove(op.getTrueBranch(), op.getTrueBranch(),
                            implicit_true_operand_set);
  getUsedValuesDefinedAbove(op.getFalseBranch(), op.getFalseBranch(),
                            implicit_false_operand_set);
  llvm::SmallVector<mlir::Value> implicit_true_operands =
      implicit_true_operand_set.takeVector();
  llvm::SmallVector<mlir::Value> implicit_false_operands =
      implicit_false_operand_set.takeVector();
  llvm::SmallVector<std::optional<xla::OpSharding>> ret_shardings =
      GetResultShardings(ctx.builder->sharding(), op->getNumResults());
  llvm::SmallVector<xla::XlaOp> true_args;
  if (failed(GetXlaOps(op, implicit_true_operands, ctx, true_args)))
    return failure();
  llvm::SmallVector<xla::XlaOp> false_args;
  if (failed(GetXlaOps(op, implicit_false_operands, ctx, false_args)))
    return failure();
  llvm::SmallVector<std::optional<xla::OpSharding>> true_arg_shardings,
      false_arg_shardings;
  if (!ret_shardings.empty()) {
    true_arg_shardings = GetXlaOpShardings(true_args);
    false_arg_shardings = GetXlaOpShardings(false_args);
  }
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getTrueBranch(), &true_branch, implicit_true_operands,
          {}, true,
          true_arg_shardings, ret_shardings)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getFalseBranch(), &false_branch, implicit_false_operands,
          {}, true,
          false_arg_shardings, ret_shardings))) {
    return failure();
  }
  xla::XlaOp pred;
  if (failed(GetXlaOp(op.getPred(), value_map, &pred, op))) return failure();
  xla::XlaOp true_arg =
      CreateTupleIfMultipleOps(ctx.builder, true_args, true_arg_shardings);
  xla::XlaOp false_arg =
      CreateTupleIfMultipleOps(ctx.builder, false_args, false_arg_shardings);
  auto ifop =
      xla::Conditional(pred, true_arg, true_branch, false_arg, false_branch);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = ifop;
  } else {
    BuildGetTupleElementsForTupleResults(op, ifop, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(CaseOp op, OpLoweringContext ctx) {
  llvm::DenseMap<mlir::Value, xla::XlaOp>& value_map = *ctx.values;
  MutableArrayRef<Region> branches = op.getBranches();
  llvm::SmallVector<xla::XlaOp, 4> branch_operands(branches.size());
  std::vector<xla::XlaComputation> computations(branches.size());
  std::vector<xla::XlaComputation*> computations_p(branches.size());
  for (unsigned i = 0; i < branches.size(); ++i) {
    llvm::SetVector<mlir::Value> implicit_operand_set;
    getUsedValuesDefinedAbove(branches[i], branches[i], implicit_operand_set);
    llvm::SmallVector<mlir::Value> implicit_operands =
        implicit_operand_set.takeVector();
    llvm::SmallVector<std::optional<xla::OpSharding>> ret_shardings =
        GetResultShardings(ctx.builder->sharding(), op->getNumResults());
    llvm::SmallVector<xla::XlaOp> args;
    if (failed(GetXlaOps(op, implicit_operands, ctx, args))) return failure();
    llvm::SmallVector<std::optional<xla::OpSharding>> arg_shardings;
    if (!ret_shardings.empty()) {
      arg_shardings = GetXlaOpShardings(args);
    }
    branch_operands[i] =
        CreateTupleIfMultipleOps(ctx.builder, args, arg_shardings);
    computations_p[i] = &computations[i];
    if (failed(ctx.converter->LowerRegionAsComputation(
            &branches[i], computations_p[i], implicit_operands,
            {}, true, arg_shardings,
            ret_shardings)))
      return failure();
  }
  xla::XlaOp index;
  if (failed(GetXlaOp(op.getIndex(), value_map, &index, op))) return failure();
  xla::XlaOp caseop = xla::Conditional(index, computations_p, branch_operands);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = caseop;
  } else {
    BuildGetTupleElementsForTupleResults(op, caseop, ctx);
  }
  return success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CompareOp op,
                                OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  auto dir = Convert_comparison_direction(
      mlir::mhlo::stringifyComparisonDirection(op.getComparisonDirection()));
  auto type_attr = op.getCompareTypeAttr();
  xla::XlaOp xla_result;
  if (type_attr && type_attr.getValue() != mlir::mhlo::ComparisonType::NOTYPE) {
    auto type = xla::StringToComparisonType(
                    stringifyComparisonType(type_attr.getValue()).str())
                    .value();
    xla_result = xla::Compare(lhs, rhs, {}, dir, type);
  } else {
    xla_result = xla::Compare(lhs, rhs, dir);
  }
  value_map[op] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(ConstantOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(mlir::mhlo::ConvolutionOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  xla::XlaOp xla_result = xla::ConvGeneralDilated(
      lhs, rhs, Convert_window_strides(op.getWindowStrides()),
      Convert_padding(op.getPadding()),
      Convert_lhs_dilation(op.getLhsDilation()),
      Convert_rhs_dilation(op.getRhsDilation()),
      xla::ConvertConvDimensionNumbers(op.getDimensionNumbers()),
      Convertuint64_t(op.getFeatureGroupCount()),
      Convertuint64_t(op.getBatchGroupCount()),
      Unwrap(Convert_precision_config(op.getPrecisionConfig())),
      preferred_element_type, Convert_window_reversal(op.getWindowReversal()));
  value_map[op] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(ConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op] = xla::ConvertElementType(
      operand,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}
LogicalResult ExportXlaOp(CustomCallOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  llvm::SmallVector<xla::XlaOp> args;
  if (failed(GetTuple(op, op.getInputs(), ctx, args))) return failure();
  if (op.getCallTargetName() == kApproxTopK) {
    auto isSupportedAttrName = [](NamedAttribute attr) {
      auto name = attr.getName();
      return name == kCallTargetName || name == kBackendConfig ||
             name == kApiVersion || name == kCalledComputations ||
             name == kHasSideEffect;
    };
    for (const auto& attr : op->getAttrs()) {
      if (!isSupportedAttrName(attr))
        return op.emitOpError()
               << attr.getName().getValue()
               << " is not a supported attribute for ApproxTopK";
    }
    auto backend_config =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(op.getBackendConfigAttr());
    if (!backend_config)
      return op.emitOpError() << "Missing backend_config attribute";
    for (auto attr : backend_config) {
      auto name = attr.getName();
      if (!(name == kTopK || name == kReductionDim || name == kRecallTarget ||
            name == kAggregateToTopk || name == kReductionInputSizeOverride ||
            name == kIsFallback))
        return op.emitOpError()
               << name.getValue() << " is not a supported backend_config"
               << " attribute for ApproxTopK";
    }
    auto checkI64Attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<IntegerAttr>(attr_name);
      if (!attr || !attr.getType().isInteger(64))
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of i64 type";
      return success();
    };
    auto checkF32Attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<FloatAttr>(attr_name);
      if (!attr || !attr.getType().isF32())
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of f32 type";
      return success();
    };
    auto checkBoolAttr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<BoolAttr>(attr_name);
      if (!attr)
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of bool type";
      return success();
    };
    if (failed(checkI64Attr(kTopK))) return failure();
    if (failed(checkI64Attr(kReductionDim))) return failure();
    if (failed(checkF32Attr(kRecallTarget))) return failure();
    if (failed(checkBoolAttr(kAggregateToTopk))) return failure();
    if (failed(checkI64Attr(kReductionInputSizeOverride))) return failure();
    bool has_is_fallback = backend_config.contains(kIsFallback);
    if (has_is_fallback && !backend_config.getAs<BoolAttr>(kIsFallback))
      return op.emitOpError()
             << "is_fallback attribute in backend_config must be of bool type";
    int64_t top_k = backend_config.getAs<IntegerAttr>(kTopK).getInt();
    int64_t reduction_dim =
        backend_config.getAs<IntegerAttr>(kReductionDim).getInt();
    float recall_target = backend_config.getAs<FloatAttr>(kRecallTarget)
                              .getValue()
                              .convertToFloat();
    bool aggregate_to_topk =
        backend_config.getAs<BoolAttr>(kAggregateToTopk).getValue();
    int64_t reduction_input_size_override =
        backend_config.getAs<IntegerAttr>(kReductionInputSizeOverride).getInt();
    bool is_fallback = has_is_fallback &&
                       backend_config.getAs<BoolAttr>(kIsFallback).getValue();
    if (args.size() % 2 != 0) {
      return op.emitOpError() << "ApproxTopK takes an even number of operands.";
    }
    auto num_inputs = args.size() / 2;
    absl::Span<const xla::XlaOp> inputs(args.begin(), num_inputs);
    absl::Span<const xla::XlaOp> init_values(args.begin() + num_inputs,
                                             num_inputs);
    if (num_inputs != op.getNumResults()) {
      return op.emitOpError() << "num_results does not match num_inputs";
    }
    SmallVector<RankedTensorType> input_types, init_value_types, result_types;
    for (size_t i = 0; i < num_inputs; ++i) {
      auto input_type =
          mlir::dyn_cast<RankedTensorType>(op.getOperand(i).getType());
      if (!input_type) return failure();
      input_types.push_back(input_type);
      auto init_value_type = mlir::dyn_cast<RankedTensorType>(
          op.getOperand(num_inputs + i).getType());
      if (!init_value_type) return failure();
      init_value_types.push_back(init_value_type);
      auto result_type =
          mlir::dyn_cast<RankedTensorType>(op.getResult(i).getType());
      if (!result_type) return failure();
      result_types.push_back(result_type);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (input_types[0].getShape() != input_types[i].getShape()) {
        return op.emitOpError() << "input shape mismatch at position " << i;
      }
      if (init_value_types[i].getElementType() !=
          input_types[i].getElementType()) {
        return op.emitOpError()
               << "input and init_value element type mismatch at position "
               << i;
      }
      if (input_types[i].getElementType() != result_types[i].getElementType()) {
        return op.emitOpError()
               << "result element type mismatch at position " << i;
      }
      for (size_t j = 0; j < input_types[i].getRank(); ++j) {
        if (j == reduction_dim) {
          auto reduction_output_size = xla::ApproxTopKReductionOutputSize(
              input_types[i].getShape()[j], input_types[i].getRank(), top_k,
              recall_target, aggregate_to_topk, reduction_input_size_override);
          if (!reduction_output_size.ok()) return failure();
          if (result_types[i].getShape()[j] != reduction_output_size->first)
            return op.emitOpError()
                   << "ApproxTopK aggregates to k="
                   << reduction_output_size->first << ", but got "
                   << result_types[i].getShape()[j];
          continue;
        }
        if (input_types[i].getShape()[j] != result_types[i].getShape()[j]) {
          return op.emitOpError() << "result shape mismatch at position " << i
                                  << ", index " << j;
        }
      }
    }
    auto called_computations = op.getCalledComputations();
    if (called_computations.size() != 1) {
      return op.emitOpError()
             << "ApproxTopK takes exactly 1 called_computation.";
    }
    mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
        mlir::cast<FlatSymbolRefAttr>(op.getCalledComputations()[0]));
    mlir::FunctionType callee_type = callee.getFunctionType();
    SmallVector<Type, 4> expected_callee_input_types;
    for (unsigned i = 0; i < num_inputs; ++i) {
      auto scalar = RankedTensorType::get({}, input_types[i].getElementType());
      expected_callee_input_types.push_back(scalar);
      expected_callee_input_types.push_back(scalar);
    }
    FunctionType expected_callee_type = mlir::FunctionType::get(
        op->getContext(), expected_callee_input_types,
        RankedTensorType::get({}, IntegerType::get(op->getContext(), 1)));
    if (callee_type != expected_callee_type) {
      return op.emitOpError()
             << "called_computation type does not match the expected type. Got "
             << callee_type << " expected " << expected_callee_type;
    }
    if (failed(ctx.converter->RunOnFunction(callee))) return failure();
    xla::XlaComputation& comparator =
        ctx.converter->GetLoweredComputation(callee);
    if (reduction_dim < 0 || reduction_dim > input_types[0].getRank())
      return op.emitOpError() << "reduction_dim out of range";
    if (recall_target <= 0 || recall_target > 1.0)
      return op.emitOpError() << "recall_target out of range";
    if (reduction_input_size_override >= 0 &&
        reduction_input_size_override <
            input_types[0].getShape()[reduction_dim])
      return op.emitOpError() << "reduction_input_size_override out of range";
    xla::XlaOp cc_op;
    if (is_fallback) {
      cc_op = xla::ApproxTopKFallback(
          ctx.builder, inputs, init_values, top_k, reduction_dim, comparator,
          recall_target, aggregate_to_topk, reduction_input_size_override);
    } else {
      cc_op = xla::ApproxTopK(ctx.builder, inputs, init_values, top_k,
                              reduction_dim, comparator, recall_target,
                              aggregate_to_topk, reduction_input_size_override);
    }
    BuildGetTupleElementsForTupleResults(op, cc_op, ctx);
    return success();
  }
  if (op.getCalledComputations().size() > 1)
    return op.emitOpError()
           << "cannot export with more than one called computations";
  if (!op.getCalledComputations().empty() && op.getOperandLayouts() &&
      op.getResultLayouts()) {
    return op.emitOpError() << "cannot export if both called computation and "
                               "layouts are specified";
  }
  auto xla_api_version = xla::ConvertCustomCallApiVersion(op.getApiVersion());
  if (!xla_api_version.ok()) return failure();
  std::string backend_config;
  if (*xla_api_version == xla::CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    if (auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            op.getBackendConfig().value_or(mlir::Attribute()))) {
      llvm::raw_string_ostream(backend_config) << dict;
    }
  } else {
    if (auto str = mlir::dyn_cast_or_null<mlir::StringAttr>(
            op.getBackendConfig().value_or(mlir::Attribute()))) {
      llvm::raw_string_ostream(backend_config) << str.strref();
    }
  }
  absl::StatusOr<xla::Literal> literal;
  const xla::Literal* literal_ptr = nullptr;
  auto literal_attr = op->getAttrOfType<DenseElementsAttr>(kMhloLiteral);
  if (literal_attr) {
    literal = CreateArrayLiteralFromAttr(literal_attr, {});
    if (!literal.ok()) return failure();
    literal_ptr = &*literal;
  }
  auto aliasInfo =
      xla::ConvertOutputOperandAliasing(op.getOutputOperandAliases());
  auto output_operand_aliasing = absl::MakeSpan(*aliasInfo);
  auto custom_call_schedule =
      xla::ConvertCustomCallSchedule(op.getCustomCallSchedule());
  if (!custom_call_schedule.ok()) return failure();
  std::string call_target_name(op.getCallTargetName());
  xla::Shape result_shape;
  if (op->getNumResults() == 1) {
    result_shape = xla::TypeToShape(op.getResult(0).getType());
  } else {
    std::vector<xla::Shape> subshapes;
    for (const auto& item : op.getResults().getType()) {
      subshapes.push_back(xla::TypeToShape(item));
    }
    result_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  }
  xla::XlaOp custom_call;
  if (op.getCalledComputations().size() == 1) {
    mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
        mlir::cast<FlatSymbolRefAttr>(op.getCalledComputations()[0]));
    if (failed(ctx.converter->RunOnFunction(callee))) return failure();
    xla::XlaComputation& computation =
        ctx.converter->GetLoweredComputation(callee);
    custom_call = xla::CustomCallWithComputation(
        ctx.builder, call_target_name, args, computation, result_shape,
        backend_config, op.getHasSideEffect(), output_operand_aliasing,
        literal_ptr, *custom_call_schedule, *xla_api_version);
  } else if (op.getOperandLayouts() && op.getResultLayouts()) {
    auto operand_shapes_with_layout = ConvertTypesToShapesWithLayout(
        op.getOperandTypes(), op.getOperandLayouts().value());
    SetLayout(result_shape, op.getResultLayouts().value());
    custom_call = xla::CustomCallWithLayout(
        ctx.builder, call_target_name, args, result_shape,
        operand_shapes_with_layout, backend_config, op.getHasSideEffect(),
        output_operand_aliasing, literal_ptr, *custom_call_schedule,
        *xla_api_version);
  } else {
    custom_call = xla::CustomCall(
        ctx.builder, call_target_name, args, result_shape, backend_config,
        op.getHasSideEffect(), output_operand_aliasing, literal_ptr,
        *custom_call_schedule, *xla_api_version);
  }
  if (op->getNumResults() == 1) {
    value_map[op.getResult(0)] = custom_call;
  } else {
    BuildGetTupleElementsForTupleResults(op, custom_call, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(InfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();
  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }
  xla::Shape data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  auto xla_result = xla::InfeedWithToken(token, data_shape,
                                         std::string(op.getInfeedConfig()));
  ctx.builder->ClearSharding();
  if (!subshapes.empty()) {
    auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }
  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);
  return success();
}
LogicalResult ExportXlaOp(IotaOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Iota(ctx.builder, xla::TypeToShape(op.getType()),
                            op.getIotaDimension());
  return success();
}
LogicalResult ExportXlaOp(MapOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  value_map[op] = xla::Map(ctx.builder, operands, computation,
                           Convert_dimensions(op.getDimensions()));
  return success();
}
LogicalResult ExportXlaOp(OutfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  const auto sharding = ctx.builder->sharding();
  xla::XlaOp operand;
  if (sharding.has_value() &&
      sharding->tuple_shardings_size() != operands.size()) {
    xla::XlaScopedShardingAssignment scoped_sharding(ctx.builder, std::nullopt);
    operand = Tuple(ctx.builder, operands);
  } else {
    operand = Tuple(ctx.builder, operands);
  }
  std::vector<xla::Shape> subshapes;
  for (auto operand : op.getInputs())
    subshapes.push_back(xla::TypeToShape(operand.getType()));
  xla::Shape shape_with_layout = xla::ShapeUtil::MakeTupleShape(subshapes);
  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  value_map[op] = xla::OutfeedWithToken(operand, token, shape_with_layout,
                                        std::string(op.getOutfeedConfig()));
  return success();
}
LogicalResult ExportXlaOp(PartitionIdOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::Shape shape = xla::TypeToShape(op.getResult().getType());
  value_map[op] =
      xla::internal::XlaBuilderFriend::BuildPartitionId(ctx.builder, shape);
  return success();
}
LogicalResult ExportXlaOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.getEdgePaddingLow());
  auto edge_padding_high = ConvertDenseIntAttr(op.getEdgePaddingHigh());
  auto interior_padding = ConvertDenseIntAttr(op.getInteriorPadding());
  for (int64_t i = 0, end = edge_padding_low.size(); i < end; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(edge_padding_low[i]);
    dims->set_edge_padding_high(edge_padding_high[i]);
    dims->set_interior_padding(interior_padding[i]);
  }
  xla::XlaOp operand, padding_value;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getPaddingValue(), value_map, &padding_value, op)))
    return failure();
  value_map[op] = xla::Pad(operand, padding_value, padding_config);
  return success();
}
LogicalResult ExportXlaOp(RecvOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();
  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }
  xla::Shape data_shape;
  if (subshapes.size() == 1)
    data_shape = subshapes[0];
  else
    data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  token = xla::internal::XlaBuilderFriend::BuildRecv(
      ctx.builder, token, data_shape,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());
  xla::XlaOp xla_result = xla::internal::XlaBuilderFriend::BuildRecvDone(
      ctx.builder, token, data_shape,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());
  auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
  if (subshapes.size() == 1) {
    value_map[op.getResult(0)] = data_tuple_element;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }
  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);
  return success();
}
LogicalResult ExportXlaOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }
  xla::XlaOp result =
      xla::Reduce(ctx.builder, operands, init_values, body,
                  Convert_broadcast_dimensions(op.getDimensions()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(ReduceWindowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }
  xla::XlaOp result = xla::ReduceWindowWithGeneralPadding(
      operands, init_values, body,
      ConvertDenseIntAttr(op.getWindowDimensions()),
      ConvertDenseIntAttr(op.getWindowStrides()),
      ConvertDenseIntAttr(op.getBaseDilations()),
      ConvertDenseIntAttr(op.getWindowDilations()),
      Convert_padding(op.getPadding()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(ReshapeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op] =
      xla::Reshape(operand, xla::TypeToShape(op.getType()).dimensions());
  return success();
}
LogicalResult ExportXlaOp(ReturnOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(RngBitGeneratorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto results = op.getResults();
  auto xla_arg_1 = value_map[*op.getODSOperands(0).begin()];
  auto xla_result = xla::RngBitGenerator(
      static_cast<xla::RandomAlgorithm>(op.getRngAlgorithm()),
      Unwrap(xla_arg_1), xla::TypeToShape(results[1].getType()));
  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);
  return mlir::success();
}
LogicalResult ExportXlaOp(XlaRngGetAndUpdateStateOp op, OpLoweringContext ctx) {
  (*ctx.values)[op.getResult()] =
      xla::internal::XlaBuilderFriend::BuildRngGetAndUpdateState(
          ctx.builder, static_cast<int64_t>(op.getDelta()),
          xla::TypeToShape(op.getType()));
  return mlir::success();
}
LogicalResult ExportXlaOp(BatchNormGradOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, scale, mean, variance, grad_output;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getScale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.getMean(), value_map, &mean, op))) return failure();
  if (failed(GetXlaOp(op.getVariance(), value_map, &variance, op)))
    return failure();
  if (failed(GetXlaOp(op.getGradOutput(), value_map, &grad_output, op)))
    return failure();
  auto xla_result =
      xla::BatchNormGrad(operand, scale, mean, variance, grad_output,
                         ConvertAPFloat(op.getEpsilon()), op.getFeatureIndex());
  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);
  return mlir::success();
}
LogicalResult ExportXlaOp(BatchNormTrainingOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, scale, offset;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getScale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.getOffset(), value_map, &offset, op)))
    return failure();
  auto xla_result = xla::BatchNormTraining(operand, scale, offset,
                                           ConvertAPFloat(op.getEpsilon()),
                                           op.getFeatureIndex());
  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);
  return mlir::success();
}
LogicalResult ExportXlaOp(RngOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp a, b;
  if (failed(GetXlaOp(op.getA(), value_map, &a, op))) return failure();
  if (failed(GetXlaOp(op.getB(), value_map, &b, op))) return failure();
  if (op.getRngDistribution() == RngDistribution::UNIFORM) {
    value_map[op] = xla::RngUniform(a, b, xla::TypeToShape(op.getType()));
    return success();
  } else if (op.getRngDistribution() == RngDistribution::NORMAL) {
    value_map[op] = xla::RngNormal(a, b, xla::TypeToShape(op.getType()));
    return success();
  }
  return failure();
}
LogicalResult ExportXlaOp(ScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation update_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getUpdateComputation(),
                                                     &update_computation))) {
    return failure();
  }
  xla::ScatterDimensionNumbers dimension_numbers =
      Convert_scatter_dimension_numbers(op.getScatterDimensionNumbers());
  llvm::SmallVector<xla::XlaOp> operands;
  llvm::SmallVector<xla::XlaOp> updates;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  if (failed(GetTuple(op, op.getUpdates(), ctx, updates))) return failure();
  xla::XlaOp scatter_indices;
  if (failed(GetXlaOp(op.getScatterIndices(), value_map, &scatter_indices, op)))
    return failure();
  auto scatter_op = xla::Scatter(
      operands, scatter_indices, updates, update_computation, dimension_numbers,
      op.getIndicesAreSorted(), op.getUniqueIndices());
  if (op->getNumResults() == 1) {
    value_map[op.getResult(0)] = scatter_op;
    return success();
  }
  BuildGetTupleElementsForTupleResults(op, scatter_op, ctx);
  return success();
}
LogicalResult ExportXlaOp(SelectAndScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation select;
  xla::XlaComputation scatter;
  if (failed(
          ctx.converter->LowerRegionAsComputation(&op.getSelect(), &select)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.getScatter(),
                                                     &scatter))) {
    return failure();
  }
  xla::XlaOp operand, source, init_value;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getSource(), value_map, &source, op)))
    return failure();
  if (failed(GetXlaOp(op.getInitValue(), value_map, &init_value, op)))
    return failure();
  value_map[op] = xla::SelectAndScatterWithGeneralPadding(
      operand, select, ConvertDenseIntAttr(op.getWindowDimensions()),
      ConvertDenseIntAttr(op.getWindowStrides()),
      Convert_padding(op.getPadding()), source, init_value, scatter);
  return success();
}
LogicalResult ExportXlaOp(SendOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  xla::XlaOp operand;
  if (operands.size() == 1)
    operand = operands[0];
  else
    operand = Tuple(ctx.builder, operands);
  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  token = xla::internal::XlaBuilderFriend::BuildSend(
      ctx.builder, operand, token,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());
  value_map[op] = xla::internal::XlaBuilderFriend::BuildSendDone(
      ctx.builder, token, Convert_channel_handle(op.getChannelHandle()),
      op.getIsHostTransfer());
  return success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SetDimensionSizeOp op,
                                OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp array;
  if (failed(GetXlaOp(op.getOperand(), value_map, &array, op)))
    return mlir::failure();
  auto dimension = Convertuint64_t(op.getDimension());
  auto shape_or = ctx.builder->GetShapePtr(array);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }
  xla::XlaOp xla_result;
  if (auto constant = llvm::dyn_cast_or_null<mlir::mhlo::ConstantOp>(
          op.getSize().getDefiningOp());
      constant != nullptr) {
    auto value = constant.getValue();
    auto values = value.getValues<mlir::IntegerAttr>();
    if ((*values.begin()).getValue().getSExtValue() ==
        shape_or.value()->dimensions(dimension)) {
      xla_result = xla::RemoveDynamicDimension(array, dimension);
    }
  }
  if (!xla_result.valid()) {
    xla::XlaOp dynamic_size;
    if (failed(GetXlaOp(op.getSize(), value_map, &dynamic_size, op)))
      return mlir::failure();
    xla_result = xla::SetDimensionSize(array, dynamic_size, dimension);
  }
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SineOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Sin(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(SortOp op, OpLoweringContext ctx) {
  xla::XlaComputation comparator;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComparator(),
                                                     &comparator)))
    return failure();
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  auto sorted =
      xla::Sort(operands, comparator, op.getDimension(), op.getIsStable());
  auto& value_map = *ctx.values;
  auto shape_or = sorted.builder()->GetShape(sorted);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }
  xla::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = sorted;
    return success();
  }
  BuildGetTupleElementsForTupleResults(op, sorted, ctx);
  return success();
}
LogicalResult ExportXlaOp(SubtractOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp lhs;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &lhs, op)))
    return mlir::failure();
  xla::XlaOp rhs;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &rhs, op)))
    return mlir::failure();
  auto xla_result = xla::Sub(Unwrap(lhs), Unwrap(rhs));
  value_map[result] = xla_result;
  return mlir::success();
}
LogicalResult ExportXlaOp(TraceOp op, OpLoweringContext ctx) {
  return success();
}
LogicalResult ExportXlaOp(WhileOp op, OpLoweringContext ctx) {
  xla::XlaComputation condition;
  xla::XlaComputation body;
  llvm::SmallVector<std::optional<xla::OpSharding>> res_shardings =
      GetResultShardings(ctx.builder->sharding(), op->getNumResults());
  llvm::SetVector<mlir::Value> implicit_operand_set;
  getUsedValuesDefinedAbove(op->getRegions(), implicit_operand_set);
  llvm::SmallVector<mlir::Value> implicit_operands =
      implicit_operand_set.takeVector();
  llvm::SmallVector<xla::XlaOp> implicit_args;
  if (failed(GetXlaOps(op, implicit_operands, ctx, implicit_args)))
    return failure();
  llvm::SmallVector<std::optional<xla::OpSharding>> implicit_shardings;
  if (!implicit_args.empty() && !res_shardings.empty()) {
    implicit_shardings = GetXlaOpShardings(implicit_args);
    res_shardings.append(implicit_shardings.begin(), implicit_shardings.end());
    if (std::optional<xla::OpSharding> new_sharding =
            CreateTupleSharding(res_shardings)) {
      ctx.builder->SetSharding(*new_sharding);
    }
  }
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getBody(), &body, implicit_operands,
          implicit_operands,
          true, res_shardings,
          res_shardings)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getCond(), &condition, implicit_operands,
          {},
          true, res_shardings))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();
  operands.append(implicit_args.begin(), implicit_args.end());
  xla::XlaOp operand = operands[0];
  if (operands.size() > 1) operand = Tuple(ctx.builder, operands);
  xla::XlaOp whileop = xla::While(condition, body, operand);
  auto& value_map = *ctx.values;
  auto shape_or = whileop.builder()->GetShape(whileop);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }
  xla::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = whileop;
    return success();
  }
  BuildGetTupleElementsForTupleResults(
      op, whileop, ctx, implicit_args.size());
  return success();
}
LogicalResult ExportXlaOp(OptimizationBarrierOp op, OpLoweringContext ctx) {
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();
  if (operands.empty()) return success();
  auto& value_map = *ctx.values;
  if (operands.size() == 1) {
    value_map[op.getOperation()->getResult(0)] =
        xla::OptimizationBarrier(operands[0]);
  } else {
    auto result = xla::OptimizationBarrier(Tuple(ctx.builder, operands));
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(FusionOp op, OpLoweringContext ctx) {
  if (!op.getFusionKind()) {
    op.emitOpError() << "requires fusion kind for HLO translation";
    return failure();
  }
  xla::XlaComputation fused_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getFusedComputation(),
                                                     &fused_computation)))
    return failure();
  auto& values = *ctx.values;
  auto aliasInfo =
      xla::ConvertOutputOperandAliasing(op.getOutputOperandAliases());
  auto output_operand_aliasing = absl::MakeSpan(*aliasInfo);
  llvm::SmallVector<xla::XlaOp, 4> operands;
  for (auto operand : op.getInputs()) operands.push_back(values[operand]);
  auto fusion_kind_string =
      mlir::mhlo::stringifyFusionKind(op.getFusionKind().value());
  xla::XlaOp fusion = xla::internal::XlaBuilderFriend::BuildFusion(
      ctx.builder, operands,
      absl::string_view(fusion_kind_string.data(), fusion_kind_string.size()),
      fused_computation, output_operand_aliasing);
  if (op.getNumResults() == 1) {
    values[op.getResult(0)] = fusion;
  } else {
    BuildGetTupleElementsForTupleResults(op, fusion, ctx);
  }
  return success();
}
LogicalResult ExportXlaOp(BitcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  xla::XlaOp bitcast = xla::internal::XlaBuilderFriend::BuildBitcast(
      ctx.builder, operand, xla::TypeToShape(op.getType()));
  value_map[op] = bitcast;
  if (ctx.converter->GetOptions().propagate_bitcast_layouts_to_backend_config) {
    xla::HloInstructionProto* bitcast_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(bitcast);
    xla::HloInstructionProto* operand_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(operand);
    xla::LayoutProto result_layout =
        ExtractLayout(op, bitcast_proto->shape().dimensions_size(),
                      kResultLayout)
            .ToProto();
    xla::LayoutProto source_layout =
        ExtractLayout(op, operand_proto->shape().dimensions_size(),
                      kSourceLayout)
            .ToProto();
    xla::gpu::BitcastBackendConfig bitcast_config;
    *bitcast_config.mutable_source_layout() = source_layout;
    *bitcast_config.mutable_result_layout() = result_layout;
    *bitcast_proto->mutable_backend_config() =
        bitcast_config.SerializeAsString();
  }
  return success();
}
LogicalResult ExportXlaOp(UniformQuantizeOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(UniformDequantizeOp op, OpLoweringContext ctx) {
  return failure();
}
LogicalResult ExportXlaOp(TopKOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  auto topk = xla::TopK(operand, op.getK(), op.getLargest());
  BuildGetTupleElementsForTupleResults(op, topk, ctx);
  return success();
}
LogicalResult ExportXlaOp(MinimumBroadcastShapesOp op, OpLoweringContext ctx) {
  return failure();
}
}  
}  
}  
#include "xla/hlo/translate/mhlo_to_hlo/operator_writers.inc"
namespace mlir {
namespace {
LogicalResult ConvertLayout(mlir::Operation* op, const mlir::ArrayAttr& layout,
                            xla::ShapeProto* shape) {
  if (shape->element_type() == xla::TUPLE) {
    auto subshapes = shape->mutable_tuple_shapes();
    size_t subshapes_data_size = subshapes->size();
    if (!subshapes->empty() &&
        subshapes->Mutable(subshapes->size() - 1)->element_type() == xla::TOKEN)
      subshapes_data_size = subshapes->size() - 1;
    if (layout.size() != subshapes_data_size) {
      op->emitOpError() << "Expected layout of size " << layout.size()
                        << ", but found " << subshapes->size();
      return failure();
    }
    for (int i = 0; i < subshapes_data_size; i++) {
      mlir::Attribute child = layout[i];
      if (mlir::isa<mlir::UnitAttr>(child)) {
        continue;
      }
      mlir::ArrayAttr c = mlir::dyn_cast<mlir::ArrayAttr>(child);
      if (!c) {
        op->emitOpError() << "Type Error: Expected layout array attribute";
        return failure();
      }
      if (failed(ConvertLayout(op, c, subshapes->Mutable(i)))) {
        return failure();
      }
    }
  } else {
    int rank = shape->dimensions().size();
    if (rank) {
      if (layout.size() != rank) {
        return failure();  
      }
      std::vector<int64_t> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr = mlir::dyn_cast<mlir::IntegerAttr>(layout[i]);
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }
  }
  return success();
}
LogicalResult ConvertInfeedtLayout(mlir::Operation* op,
                                   const mlir::ArrayAttr& layout,
                                   xla::ShapeProto* shape,
                                   int64_t layout_index = 0) {
  if (shape->element_type() != xla::TUPLE) {
    mlir::ArrayAttr child_layout =
        mlir::dyn_cast<mlir::ArrayAttr>(layout[layout_index]);
    if (!child_layout) {
      op->emitOpError() << "Type Error: Expected layout array attribute";
      return failure();
    }
    int rank = shape->dimensions().size();
    if (rank) {
      if (child_layout.size() != rank) {
        return failure();  
      }
      std::vector<int64_t> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr =
            mlir::dyn_cast<mlir::IntegerAttr>(child_layout[i]);
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }
    return success();
  }
  auto subshapes = shape->mutable_tuple_shapes();
  auto datashape = subshapes->Mutable(0);
  if (datashape->element_type() == xla::TUPLE) {
    auto data_subshapes = datashape->mutable_tuple_shapes();
    if (layout.size() != data_subshapes->size()) {
      op->emitOpError() << "Expected " << data_subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }
    for (int i = 0; i < data_subshapes->size(); i++) {
      if (failed(
              ConvertInfeedtLayout(op, layout, data_subshapes->Mutable(i), i)))
        return failure();
    }
  } else {
    if (layout.size() != subshapes->size()) {
      op->emitOpError() << "Expected " << subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }
    for (int i = 0; i < subshapes->size(); i++) {
      if (failed(ConvertInfeedtLayout(op, layout, subshapes->Mutable(i), i)))
        return failure();
    }
  }
  return success();
}
LogicalResult ExportXlaOperatorWrapped(mlir::Operation* inst,
                                       OpLoweringContext ctx) {
  auto op = dyn_cast<mlir::mhlo::AddOp>(inst);
  if (op && mlir::cast<mlir::TensorType>(op.getResult().getType())
                .getElementType()
                .isSignlessInteger(1)) {
    auto& value_map = *ctx.values;
    auto result = op.getResult();
    xla::XlaOp xla_arg_0;
    if (failed(GetXlaOp(op.getLhs(), value_map, &xla_arg_0, op)))
      return mlir::failure();
    xla::XlaOp xla_arg_1;
    if (failed(GetXlaOp(op.getRhs(), value_map, &xla_arg_1, op)))
      return mlir::failure();
    auto xla_result = xla::Xor(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
    value_map[result] = xla_result;
    return mlir::success();
  }
  return ExportXlaOperator(inst, ctx);
}
LogicalResult ConvertToHloModule::PropagateLayouts(
    const MlirToHloConversionOptions& options, mlir::Operation* inst,
    xla::XlaOp xla_op) {
  if (options.propagate_layouts) {
    auto* shape = xla::internal::XlaBuilderFriend::GetInstruction(xla_op)
                      ->mutable_shape();
    mlir::FailureOr<xla::Shape> mlir_shape_or = ExtractXlaShape(inst);
    if (failed(mlir_shape_or)) return failure();
    *shape = mlir_shape_or->ToProto();
  }
  return success();
}
LogicalResult ConvertToHloModule::LowerCast(
    mlir::Operation* inst, const MlirToHloConversionOptions& options,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto cast_op = cast<mlir::tensor::CastOp>(inst);
  Value operand = cast_op.getOperand();
  auto ty = mlir::dyn_cast<ShapedType>(operand.getType());
  if (!ty || !IsBoundedOrStatic(ty)) {
    inst->emitOpError()
        << "requires static or bounded operand for HLO translation";
    return failure();
  }
  xla::XlaOp xla_operand;
  auto& value_map = *value_lowering;
  if (failed(GetXlaOp(operand, value_map, &xla_operand, cast_op)))
    return failure();
  value_map[cast_op.getResult()] = xla_operand;
  if (failed(PropagateLayouts(options, inst, xla_operand))) {
    return failure();
  }
  return success();
}
LogicalResult ConvertToHloModule::LowerCompositeCall(
    mlir::Operation* inst, xla::XlaBuilder* module_builder,
    xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaOp* return_value) {
  auto& value_map = *value_lowering;
  SmallVector<xla::XlaOp, 1> operands;
  for (const Value& val : inst->getOperands()) {
    xla::XlaOp operand;
    if (failed(GetXlaOp(val, value_map, &operand, inst))) {
      return failure();
    }
    operands.push_back(operand);
  }
  auto composite_op = cast<mhlo::CompositeOp>(inst);
  xla::XlaComputation computation;
  if (failed(LowerBasicBlockAsFunction(
          &module_
              .lookupSymbol<mlir::func::FuncOp>(composite_op.getDecomposition())
              .getBody()
              .front(),
          module_builder_
              .CreateSubBuilder(composite_op.getDecomposition().str())
              .get(),
          false,
          false,
          {},
          {}, {},
          {}, &computation,
          {}))) {
    return failure();
  }
  std::string composite_attributes;
  llvm::raw_string_ostream(composite_attributes)
      << composite_op.getCompositeAttributes();
  xla::XlaOp composite_call = xla::CompositeCall(
      builder, computation, operands, composite_op.getName().str(),
      composite_attributes, composite_op.getVersion());
  unsigned num_results = composite_op.getNumResults();
  if (num_results > 1) {
    for (unsigned i = 0; i != num_results; ++i) {
      value_map[composite_op.getResult(i)] =
          xla::GetTupleElement(composite_call, i);
    }
  } else if (num_results == 1) {
    value_map[composite_op.getResult(0)] = composite_call;
  }
  *return_value = composite_call;
  return success();
}
LogicalResult ConvertToHloModule::LowerConstant(
    mlir::Operation* inst, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    ElementsAttr const_attr) {
  if (!mlir::isa<ShapedType>(inst->getResult(0).getType())) {
    return inst->emitError(
        "expected shaped type during constant mhlo -> hlo translation");
  }
  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(inst);
  if (failed(shape_or)) return failure();
  auto literal_or = CreateArrayLiteralFromAttr(const_attr, shape_or->layout());
  if (!literal_or.ok()) return inst->emitError(literal_or.status().ToString());
  xla::XlaScopedShardingAssignment scoped_sharding(
      builder, CreateOpShardingFromAttribute(inst));
  auto constant = xla::ConstantLiteral(builder, literal_or.value());
  auto& value_map = *value_lowering;
  value_map[inst->getResult(0)] = constant;
  return success();
}
LogicalResult ConvertToHloModule::LowerInfeed(
    mlir::Operation* inst, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  mlir::ArrayAttr layout = inst->getAttrOfType<mlir::ArrayAttr>(kLayout);
  if (!layout) return success();
  auto num_results = inst->getNumResults();
  bool propagate_layout_to_data_tuple = true;
  for (unsigned i = 0; i < num_results; i++) {
    auto iter = value_lowering->find(inst->getResult(i));
    if (iter == value_lowering->end()) {
      inst->emitOpError() << "inst's result value at index " << i
                          << " has no match in value_lowering";
      return failure();
    }
    auto xla_gte_op = iter->second;
    xla::HloInstructionProto* get_tuple_element_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(xla_gte_op);
    assert(xla::StringToHloOpcode(get_tuple_element_proto->opcode()).value() ==
               xla::HloOpcode::kGetTupleElement &&
           "The token-result of mhlo.InfeedOp should be mapped to a "
           "xla::HloOpcode::kGetTupleElement");
    if (i == num_results - 1) {
      xla::HloInstructionProto* xla_infeed_op_proto =
          xla::internal::XlaBuilderFriend::GetInstructionByHandle(
              xla_gte_op.builder(), get_tuple_element_proto->operand_ids(0));
      assert(xla::StringToHloOpcode(xla_infeed_op_proto->opcode()).value() ==
                 xla::HloOpcode::kInfeed &&
             "Expected xla::HloOpcode::kInfeed op");
      auto* shape = xla_infeed_op_proto->mutable_shape();
      if (failed(ConvertInfeedtLayout(inst, layout, shape))) return failure();
      continue;
    }
    auto* shape = get_tuple_element_proto->mutable_shape();
    if (failed(ConvertInfeedtLayout(inst, layout, shape, i))) return failure();
    if (propagate_layout_to_data_tuple) {
      xla::HloInstructionProto* data_tuple_proto =
          xla::internal::XlaBuilderFriend::GetInstructionByHandle(
              xla_gte_op.builder(), get_tuple_element_proto->operand_ids(0));
      auto* data_tuple_shape = data_tuple_proto->mutable_shape();
      assert(xla::StringToHloOpcode(data_tuple_proto->opcode()).value() ==
                 xla::HloOpcode::kGetTupleElement &&
             "Expected a xla:tupleOp for all the data results.");
      if (failed(ConvertInfeedtLayout(inst, layout, data_tuple_shape)))
        return failure();
    }
    propagate_layout_to_data_tuple = false;
  }
  return success();
}
LogicalResult ConvertToHloModule::LowerReturn(
    Operation* inst, bool is_entry_function,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
    llvm::ArrayRef<mlir::Value> implicit_results, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaOp* return_value, const MlirToHloConversionOptions& options) {
  unsigned num_return_values = inst->getNumOperands() + implicit_results.size();
  std::optional<xla::OpSharding> ret_tuple_sharding =
      CreateTupleSharding(ret_shardings);
  auto& value_map = *value_lowering;
  if ((options_.return_tuple && is_entry_function) || num_return_values != 1) {
    std::vector<xla::XlaOp> returns;
    returns.reserve(num_return_values);
    for (Value ret : inst->getOperands()) {
      xla::XlaOp& operand = returns.emplace_back();
      if (failed(GetXlaOp(ret, value_map, &operand, inst))) return failure();
    }
    for (Value ret : implicit_results) {
      xla::XlaOp& operand = returns.emplace_back();
      if (failed(GetXlaOp(ret, value_map, &operand, inst))) return failure();
    }
    if (is_entry_function && ret_tuple_sharding) {
      assert(implicit_results.empty() &&
             "entry functions shouldn't have implicit results");
      for (OpOperand& ret : inst->getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::Shape return_shape = xla::TypeToShape(ret.get().getType());
        absl::StatusOr<xla::XlaOp> reshape =
            ReshapeWithCorrectRepresentationAndSharding(
                builder, returns[index], return_shape,
                options_.layout_preference_fn, options_.shape_representation_fn,
                ret_shardings[index],
                false);
        if (!reshape.ok())
          return inst->emitError() << reshape.status().message();
        returns[index] = reshape.value();
      }
    }
    xla::XlaScopedShardingAssignment scoped_sharding(builder,
                                                     ret_tuple_sharding);
    *return_value = xla::Tuple(builder, returns);
    return success();
  }
  if (num_return_values == 1) {
    Value ret = implicit_results.empty() ? inst->getOperand(0)
                                         : implicit_results.front();
    xla::XlaOp operand;
    if (failed(GetXlaOp(ret, value_map, &operand, inst))) return failure();
    if (ret_tuple_sharding) {
      auto tuple = Tuple(builder, {operand});
      builder->SetSharding(*ret_shardings[0]);
      *return_value = GetTupleElement(tuple, 0);
      builder->ClearSharding();
    } else {
      *return_value = operand;
    }
  }
  return success();
}
LogicalResult ConvertToHloModule::Lower(
    mlir::Operation* inst, bool is_entry_function,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
    llvm::ArrayRef<mlir::Value> implicit_results, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaOp* return_value) {
  if (inst->getDialect() !=
          inst->getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>() &&
      !mlir::isa<mlir::func::ConstantOp, mlir::arith::ConstantOp,
                 mlir::func::CallOp, mlir::tensor::CastOp,
                 mlir::func::ReturnOp>(inst)) {
    inst->emitOpError("unsupported op for export to XLA");
    return failure();
  }
  *return_value = xla::XlaOp();
  if (succeeded(ExportXlaOperatorWrapped(
          inst,
          {value_lowering, this, builder, &stack_frame_indexes_builder_}))) {
    if (inst->getNumResults() == 1) {
      auto iter = value_lowering->find(inst->getResult(0));
      if (iter == value_lowering->end()) {
        inst->emitOpError(
            "inst has a result, but it's not found in value_lowering");
        return failure();
      }
      if (failed(PropagateLayouts(options_, inst, iter->second))) {
        return failure();
      }
    }
    if (isa<mhlo::InfeedOp>(inst)) {
      return LowerInfeed(inst, builder, value_lowering);
    }
    return success();
  }
  if (auto call_op = dyn_cast<mlir::func::CallOp>(inst)) {
    return LowerFunctionCall(call_op, builder, value_lowering);
  }
  if (isa<mlir::tensor::CastOp>(inst)) {
    return LowerCast(inst, options_, value_lowering);
  }
  if (auto composite_op = dyn_cast<mhlo::CompositeOp>(inst)) {
    return LowerCompositeCall(inst, &module_builder_, builder, value_lowering,
                              return_value);
  }
  ElementsAttr const_attr;
  if (matchPattern(inst, m_Constant(&const_attr))) {
    return LowerConstant(inst, builder, value_lowering, const_attr);
  }
  if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
    return LowerReturn(inst, is_entry_function, ret_shardings, implicit_results,
                       builder, value_lowering, return_value, options_);
  }
  inst->emitOpError() << "can't be translated to XLA HLO";
  return failure();
}
LogicalResult ConvertToHloModule::LowerFunctionCall(
    mlir::func::CallOp call_op, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto& value_map = *value_lowering;
  mlir::func::FuncOp callee =
      module_.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
  if (failed(RunOnFunction(callee))) return failure();
  std::vector<xla::XlaOp> operands;
  for (auto operand : call_op.getOperands()) {
    xla::XlaOp xla_operand;
    if (failed(GetXlaOp(operand, value_map, &xla_operand, call_op)))
      return failure();
    operands.push_back(xla_operand);
  }
  xla::FrontendAttributes fe_attrs = CreateXlaFrontendAttributesFromOp(call_op);
  xla::XlaScopedFrontendAttributesAssignment assignment(builder, fe_attrs);
  xla::XlaOp call_result =
      xla::Call(builder, lowered_computation_[callee], operands);
  unsigned num_results = call_op.getNumResults();
  if (num_results > 1) {
    for (unsigned i = 0; i != num_results; ++i) {
      value_map[call_op.getResult(i)] = xla::GetTupleElement(call_result, i);
    }
  } else if (num_results == 1) {
    value_map[call_op.getResult(0)] = call_result;
  }
  return success();
}
LogicalResult ConvertToHloModule::RunOnFunction(mlir::func::FuncOp f) {
  if (lowered_computation_.count(f)) return success();
  if (!llvm::hasSingleElement(f)) {
    return f.emitError("only single block Function supported");
  }
  std::unique_ptr<xla::XlaBuilder> builder_up;
  bool entry_function = f.getName() == kMain;
  if (!entry_function)
    builder_up = module_builder_.CreateSubBuilder(f.getName().str());
  auto& builder = entry_function ? module_builder_ : *builder_up;
  xla::XlaComputation computation;
  std::vector<bool> entry_args_same_across_replicas;
  llvm::SmallVector<std::optional<xla::OpSharding>, 4> arg_shardings;
  llvm::SmallVector<std::optional<xla::OpSharding>, 4> ret_shardings;
  llvm::SmallVector<std::optional<xla::FrontendAttributes>, 4> arg_fe_attrs;
  if (entry_function) {
    bool any_arg_replicated = false;
    entry_args_same_across_replicas.reserve(f.getNumArguments());
    for (int64_t i = 0; i < f.getNumArguments(); ++i) {
      auto attr = f.getArgAttrOfType<mlir::BoolAttr>(i, kMhloReplication);
      entry_args_same_across_replicas.push_back(attr != nullptr &&
                                                attr.getValue());
      any_arg_replicated |= entry_args_same_across_replicas.back();
      auto buffer_donor =
          f.getArgAttrOfType<mlir::BoolAttr>(i, kJaxBufferDonor);
      if (buffer_donor) {
        if (options_.use_tuple_args) {
          builder.AddBufferDonor(0, {i});
        } else {
          builder.AddBufferDonor(i, {});
        }
      }
      auto aliasing_output =
          f.getArgAttrOfType<mlir::IntegerAttr>(i, kTfAliasingOutput);
      if (!aliasing_output) continue;
      xla::ShapeIndex output_index;
      if ((options_.return_tuple && entry_function) || f.getNumResults() != 1) {
        output_index = {aliasing_output.getInt()};
      } else {
        if (aliasing_output.getInt() != 0) {
          return f.emitError(
              "Aliasing output must be 0 if only one output exists");
        }
        output_index = {};
      }
      if (options_.use_tuple_args) {
        builder.SetUpAlias(output_index, 0,
                           {i});
      } else {
        builder.SetUpAlias(output_index, i,
                           {});
      }
    }
    if (!any_arg_replicated) entry_args_same_across_replicas.clear();
    ExtractShardingsFromFunction(f, &arg_shardings, &ret_shardings);
    ExtractFrontendAttributesFromFunction(f, &arg_fe_attrs);
  }
  if (failed(LowerBasicBlockAsFunction(&f.front(), &builder, entry_function,
                                       false, entry_args_same_across_replicas,
                                       arg_shardings, ret_shardings,
                                       arg_fe_attrs, &computation))) {
    return failure();
  }
  if (auto execution_thread =
          f->getAttrOfType<mlir::StringAttr>(kExecutionThread)) {
    computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
        execution_thread.str());
  }
  for (int i = 0; i < f.getNumArguments(); ++i) {
    if (auto pr =
            f.getArgAttrOfType<mlir::ArrayAttr>(i, kMhloParameterReplication)) {
      for (auto b : pr.getValue())
        for (auto& instr : *computation.mutable_proto()
                                ->mutable_computations(0)
                                ->mutable_instructions())
          if (instr.parameter_number() == i)
            instr.mutable_parameter_replication()
                ->add_replicated_at_leaf_buffers(
                    mlir::cast<mlir::BoolAttr>(b).getValue());
    }
  }
  lowered_computation_[f] = std::move(computation);
  return success();
}
LogicalResult ConvertToHloModule::SetEntryTupleShapesAndLeafReplication(
    Block* block, const std::vector<bool>& entry_args_same_across_replicas,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
    std::vector<bool>* leaf_replication) {
  arg_shapes->reserve(block->getNumArguments());
  leaf_replication->reserve(block->getNumArguments());
  for (BlockArgument& arg : block->getArguments()) {
    arg_shapes->push_back(xla::TypeToShape(arg.getType()));
    xla::Shape& arg_shape = arg_shapes->back();
    auto layout_preference_status =
        options_.layout_preference_fn ? options_.layout_preference_fn(arg_shape)
                                      : XlaLayoutPreference::kNoPreference;
    if (!layout_preference_status.ok())
      return block->getParentOp()->emitError()
             << layout_preference_status.status().message();
    auto arg_shape_status = options_.shape_representation_fn
                                ? options_.shape_representation_fn(
                                      arg_shape, false,
                                      layout_preference_status.value())
                                : arg_shape;
    if (!arg_shape_status.ok())
      return block->getParentOp()->emitError()
             << arg_shape_status.status().message();
    arg_shape = std::move(arg_shape_status.value());
    if (entry_args_same_across_replicas.empty()) continue;
    for (int i = 0, e = xla::ShapeUtil::GetLeafCount(arg_shape); i < e; ++i)
      leaf_replication->push_back(
          entry_args_same_across_replicas[arg.getArgNumber()]);
  }
  return success();
}
LogicalResult ConvertToHloModule::SetEntryTupleShardings(
    Block* block, xla::XlaBuilder* builder,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes) {
  if (!arg_shardings.empty() && SomeOptionalShardingsAreSet(arg_shardings)) {
    xla::OpSharding sharding;
    sharding.set_type(xla::OpSharding::TUPLE);
    for (const auto& arg_sharding : llvm::enumerate(arg_shardings)) {
      if (arg_sharding.value().has_value()) {
        auto hlo_sharding = xla::HloSharding::FromProto(*arg_sharding.value());
        if (!hlo_sharding.ok())
          return block->getParentOp()->emitError()
                 << hlo_sharding.status().message();
        auto status = RewriteLayoutWithShardedShape(
            hlo_sharding.value(), false,
            options_.layout_preference_fn, options_.shape_representation_fn,
            &(*arg_shapes)[arg_sharding.index()]);
        if (!status.ok())
          return block->getParentOp()->emitError() << status.message();
        *sharding.add_tuple_shardings() = *arg_sharding.value();
      } else {
        xla::OpSharding fallback_sharding;
        fallback_sharding.set_type(xla::OpSharding::REPLICATED);
        *sharding.add_tuple_shardings() = fallback_sharding;
      }
    }
    builder->SetSharding(sharding);
  }
  return success();
}
namespace {
xla::OpMetadata GetOpNameMetadataFromLocation(Value value) {
  xla::OpMetadata m;
  m.set_op_name(mhlo::GetDebugNameFromLocation(value.getLoc()));
  return m;
}
}  
LogicalResult ConvertToHloModule::LowerBasicBlockAsFunction(
    Block* block, xla::XlaBuilder* builder, bool is_entry_function,
    bool ensure_single_arg,
    const std::vector<bool>& entry_args_same_across_replicas,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
    llvm::ArrayRef<std::optional<xla::FrontendAttributes>> fe_attrs,
    xla::XlaComputation* result, llvm::ArrayRef<mlir::Value> implicit_operands,
    llvm::ArrayRef<mlir::Value> implicit_results) {
  ValueLoweringMap lowering;
  if (is_entry_function && options_.use_tuple_args) {
    llvm::SmallVector<xla::Shape, 4> arg_shapes;
    std::vector<bool> leaf_replication;
    if (failed(SetEntryTupleShapesAndLeafReplication(
            block, entry_args_same_across_replicas, &arg_shapes,
            &leaf_replication)))
      return failure();
    if (failed(
            SetEntryTupleShardings(block, builder, arg_shardings, &arg_shapes)))
      return failure();
    xla::Shape input_shape = xla::ShapeUtil::MakeTupleShape(arg_shapes);
    auto tuple =
        xla::Parameter(builder, 0, input_shape, kArgTuple, leaf_replication);
    builder->ClearSharding();
    for (BlockArgument& arg : block->getArguments()) {
      xla::XlaScopedShardingAssignment scoped_sharding(
          builder, arg_shardings.empty() ? std::nullopt
                                         : arg_shardings[arg.getArgNumber()]);
      lowering[arg] = xla::GetTupleElement(tuple, arg.getArgNumber());
    }
  } else {
    if (ensure_single_arg) {
      llvm::SmallVector<xla::Shape, 4> arg_shapes;
      auto args_size = block->getNumArguments() + implicit_operands.size();
      arg_shapes.reserve(args_size);
      for (BlockArgument& arg : block->getArguments())
        arg_shapes.push_back(xla::TypeToShape(arg.getType()));
      for (Value implicit_operand : implicit_operands)
        arg_shapes.push_back(xla::TypeToShape(implicit_operand.getType()));
      if (args_size > 1) {
        xla::XlaScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty()
                         ? std::nullopt
                         : CreateTupleSharding(arg_shardings));
        auto tuple = xla::Parameter(
            builder, 0, xla::ShapeUtil::MakeTupleShape(arg_shapes), kArgTuple);
        for (BlockArgument& arg : block->getArguments()) {
          auto num = arg.getArgNumber();
          xla::XlaScopedShardingAssignment scoped_sharding(
              builder,
              arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
          lowering[arg] = xla::GetTupleElement(tuple, num);
        }
        for (auto [implicit_index, implicit_operand] :
             llvm::enumerate(implicit_operands)) {
          int64_t arg_index = block->getNumArguments() + implicit_index;
          xla::XlaScopedShardingAssignment scoped_sharding(
              builder,
              arg_shardings.empty() ? std::nullopt : arg_shardings[arg_index]);
          lowering[implicit_operand] = xla::GetTupleElement(tuple, arg_index);
        }
      } else if (args_size == 1) {
        xla::XlaScopedShardingAssignment scoped_sharding(
            builder,
            arg_shardings.empty() ? std::nullopt : arg_shardings.front());
        mlir::Value arg = implicit_operands.empty() ? block->getArgument(0)
                                                    : implicit_operands.front();
        xla::XlaScopedOpMetadataAssignment op_metadata(
            builder, GetOpNameMetadataFromLocation(arg));
        lowering[arg] = xla::Parameter(builder, 0, arg_shapes[0], kArgPrefix);
      } else {
        xla::Parameter(builder, 0, xla::ShapeUtil::MakeTupleShape(arg_shapes),
                       kArgEmptyTuple);
      }
    } else {
      for (BlockArgument& arg : block->getArguments()) {
        auto num = arg.getArgNumber();
        xla::Shape shape = xla::TypeToShape(arg.getType());
        xla::XlaScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
        if (!fe_attrs.empty() && fe_attrs[num]) {
          builder->SetFrontendAttributes(*fe_attrs[num]);
        }
        xla::XlaScopedOpMetadataAssignment op_metadata(
            builder, GetOpNameMetadataFromLocation(arg));
        if (entry_args_same_across_replicas.empty()) {
          lowering[arg] = xla::Parameter(builder, num, shape,
                                         absl::StrCat(kArgPrefix, num));
        } else {
          lowering[arg] = xla::Parameter(
              builder, num, shape, absl::StrCat(kArgPrefix, num),
              std::vector<bool>(entry_args_same_across_replicas[num],
                                xla::ShapeUtil::GetLeafCount(shape)));
        }
        builder->ClearFrontendAttributes();
      }
    }
  }
  xla::XlaOp return_value;
  for (auto& inst : *block)
    if (failed(Lower(&inst, is_entry_function, ret_shardings, implicit_results,
                     builder, &lowering, &return_value)))
      return failure();
  auto computation_or =
      return_value.valid() ? builder->Build(return_value) : builder->Build();
  if (!computation_or.ok()) {
    block->back().emitError() << computation_or.status().message();
    return failure();
  }
  *result = std::move(computation_or.value());
  return success();
}
LogicalResult ConvertToHloModule::LowerRegionAsComputation(
    mlir::Region* region, xla::XlaComputation* func,
    llvm::ArrayRef<mlir::Value> implicit_operands,
    llvm::ArrayRef<mlir::Value> implicit_results, bool ensure_single_arg,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings) {
  std::unique_ptr<xla::XlaBuilder> builder = module_builder_.CreateSubBuilder(
      absl::StrCat(kRegionPrefix, region_id_++));
  return LowerBasicBlockAsFunction(
      &region->front(), builder.get(),
      false,
       ensure_single_arg,
      {}, arg_shardings, ret_shardings,
      {}, func, implicit_operands, implicit_results);
}
absl::Status PrepareForExport(mlir::ModuleOp module) {
  bool hasShapeOps = false;
  module.walk([&](Operation* op) {
    hasShapeOps |= isa<shape::ShapeDialect>(op->getDialect());
    return hasShapeOps ? WalkResult::interrupt() : WalkResult::advance();
  });
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(mhlo::createPrepareForExportPass());
  if (hasShapeOps) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mhlo::createSymbolicShapeOptimizationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createShapeLegalizeToHloPass());
  }
  mlir::BaseScopedDiagnosticHandler handler(module.getContext());
  (void)pm.run(module);
  absl::Status s = handler.ConsumeStatus();
  if (!s.ok()) {
    s = absl::Status(
        s.code(),
        absl::StrCat("Unable to prepare for XLA export: ", s.message()));
  }
  return s;
}
}  
absl::Status ConvertMlirHloToHlo(mlir::ModuleOp module,
                                 xla::HloProto* hlo_proto,
                                 MlirToHloConversionOptions options) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (failed(pm.run(module))) {
    return tsl::errors::Internal("Unable to convert StableHLO to MHLO");
  }
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  xla::XlaBuilder module_builder(kMain);
  ConvertToHloModule converter(module, module_builder, options);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  xla::HloModuleProto hlo_module = converter.ConsumeMainProto();
  StringRef module_name = module.getName() ? *module.getName() : kMain;
  hlo_module.set_name(module_name.str());
  if (auto cross_program_prefetches =
          module->getAttrOfType<mlir::ArrayAttr>(kMhloCrossProgramPrefetches)) {
    for (const auto& prefetch :
         Convert_cross_program_prefetches(cross_program_prefetches)) {
      *hlo_module.add_cross_program_prefetches() = std::move(prefetch);
    }
  }
  if (auto is_dynamic = module->getAttrOfType<mlir::BoolAttr>(kMhloIsDynamic)) {
    hlo_module.set_is_dynamic(is_dynamic.getValue());
  }
  if (auto frontend_attributes =
          module->getAttrOfType<DictionaryAttr>(kMhloFrontendAttributes)) {
    ConstructFrontendAttributesFromAttribute(
        frontend_attributes, *hlo_module.mutable_frontend_attributes());
  }
  if (auto use_auto_spmd_partitioning =
          module->getAttrOfType<mlir::BoolAttr>(kMhloUseAutoSpmdPartitioning)) {
    hlo_module.set_use_auto_spmd_partitioning(
        use_auto_spmd_partitioning.getValue());
  }
  if (auto spmd_output_sharding =
          module->getAttrOfType<mlir::StringAttr>(kMhloSpmdOutputSharding)) {
    *hlo_module.mutable_spmd_output_sharding() =
        *xla::ConvertSharding(spmd_output_sharding.getValue());
  }
  if (auto input_output_alias =
          module->getAttrOfType<mlir::ArrayAttr>(kMhloInputOutputAlias)) {
    if (std::optional<xla::HloInputOutputAliasProto> input_output_alias_proto =
            xla::ConvertInputOutputAlias(input_output_alias.getValue())) {
      *hlo_module.mutable_input_output_alias() = *input_output_alias_proto;
    }
  }
  if (auto spmd_parameters_sharding = module->getAttrOfType<mlir::ArrayAttr>(
          kMhloSpmdParametersShardings)) {
    for (const auto& sharding : spmd_parameters_sharding.getValue()) {
      *hlo_module.add_spmd_parameters_shardings() = *xla::ConvertSharding(
          mlir::cast<mlir::StringAttr>(sharding).getValue());
    }
  }
  if (auto xla_entry_computation_parameter_layout =
          module->getAttrOfType<mlir::ArrayAttr>(
              kMhloXlaEntryComputationParameterLayouts)) {
    auto status = mhlo::ExportModuleEntryComputationParameterLayouts(
        xla_entry_computation_parameter_layout, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto xla_entry_computation_parameter_tiles =
          module->getAttrOfType<mlir::ArrayAttr>(
              kMhloXlaEntryComputationParameterTiles)) {
    auto status = mhlo::ExportModuleEntryComputationParameterTiles(
        xla_entry_computation_parameter_tiles, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto xla_entry_computation_result_layout =
          module->getAttrOfType<mlir::ArrayAttr>(
              kMhloXlaEntryComputationResultLayout)) {
    auto status = mhlo::ExportModuleEntryComputationResultLayout(
        xla_entry_computation_result_layout, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto xla_entry_computation_result_tiles =
          module->getAttrOfType<mlir::ArrayAttr>(
              kMhloXlaEntryComputationResultTiles)) {
    auto status = mhlo::ExportModuleEntryComputationResultTiles(
        xla_entry_computation_result_tiles, hlo_module);
    if (!status.ok()) return status;
  }
  xla::StackFrameIndexProto stack_frame_index =
      converter.BuildStackFramesIndexProto();
  hlo_module.mutable_stack_frame_index()->Swap(&stack_frame_index);
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  return absl::OkStatus();
}
absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertMlirHloToHloModule(
    mlir::ModuleOp module, MlirToHloConversionOptions options) {
  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(module, &hlo_proto, options));
  const xla::HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(xla::HloModuleConfig config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          module_proto, xla::GetDebugOptionsFromFlags()));
  mhlo::ExportHloModuleConfig(config, module);
  return xla::HloModule::CreateFromProto(module_proto, config);
}
absl::Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                                 llvm::ArrayRef<xla::XlaOp> xla_params,
                                 std::vector<xla::XlaOp>& returns,
                                 MlirToHloConversionOptions options) {
  auto module = block.getParentOp()->getParentOfType<mlir::ModuleOp>();
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  options.return_tuple = false;
  options.use_tuple_args = false;
  ConvertToHloModule converter(module, builder, options);
  ConvertToHloModule::ValueLoweringMap lowering;
  if (xla_params.size() != block.getArguments().size())
    return tsl::errors::Internal("xla_params size (", xla_params.size(),
                                 ") != block arguments size (",
                                 block.getArguments().size(), ")");
  for (BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    lowering[arg] = xla_params[num];
  }
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  for (auto& inst : block) {
    if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
      returns.resize(inst.getNumOperands());
      for (OpOperand& ret : inst.getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::XlaOp operand;
        if (failed(GetXlaOp(ret.get(), lowering, &operand, &inst)))
          return diag_handler.ConsumeStatus();
        returns[index] = operand;
      }
    } else {
      xla::XlaOp return_value;
      if (failed(converter.Lower(&inst, true,
                                 {},
                                 {}, &builder, &lowering,
                                 &return_value)))
        return diag_handler.ConsumeStatus();
    }
  }
  return absl::OkStatus();
}
absl::Status ConvertMlirHloToHlo(mlir::ModuleOp module,
                                 ::xla::HloProto* hlo_proto,
                                 bool use_tuple_args, bool return_tuple,
                                 MlirToHloConversionOptions options) {
  options.use_tuple_args = use_tuple_args;
  options.return_tuple = return_tuple;
  return ConvertMlirHloToHlo(module, hlo_proto, options);
}
}  