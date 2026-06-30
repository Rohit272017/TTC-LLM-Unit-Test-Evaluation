#include "xla/service/gpu/stream_executor_util.h"
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/data_type.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
absl::StatusOr<se::dnn::VersionInfo> GetDnnVersionInfo(
    stream_executor::StreamExecutor* stream_exec) {
  if (!stream_exec) {
    return absl::InvalidArgumentError("StreamExecutor is null");
  }
  stream_executor::dnn::DnnSupport* dnn = stream_exec->AsDnn();
  if (!dnn) {
    return absl::FailedPreconditionError(
        "DNN library initialization failed. Look at the errors above for more "
        "details.");
  }
  return dnn->GetVersion();
}
se::dnn::VersionInfo GetDnnVersionInfoOrDefault(
    stream_executor::StreamExecutor* stream_exec,
    se::dnn::VersionInfo fallback_version) {
  return GetDnnVersionInfo(stream_exec).value_or(fallback_version);
}
namespace {
using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;
int64_t FindMissingDnum(absl::Span<const int64_t> vals) {
  for (int i = 0; i < vals.size(); i++) {
    if (!absl::c_linear_search(vals, i)) {
      return i;
    }
  }
  return vals.size();
}
absl::StatusOr<Layout> DataLayoutToXlaLayout(
    DataLayout data_layout, int64_t batch_dimension, int64_t feature_dimension,
    absl::Span<int64_t const> spatial_dimensions) {
  std::vector<int64_t> layout;
  switch (data_layout) {
    case DataLayout::kBatchDepthYX:  
      layout.push_back(batch_dimension);
      layout.push_back(feature_dimension);
      layout.insert(layout.end(), spatial_dimensions.begin(),
                    spatial_dimensions.end());
      break;
    case DataLayout::kBatchDepthYX4:   
    case DataLayout::kBatchDepthYX32:  
      layout.push_back(batch_dimension);
      layout.push_back(feature_dimension);
      layout.insert(layout.end(), spatial_dimensions.begin(),
                    spatial_dimensions.end());
      layout.push_back(FindMissingDnum(layout));
      break;
    case DataLayout::kBatchYXDepth:  
      layout.push_back(batch_dimension);
      layout.insert(layout.end(), spatial_dimensions.begin(),
                    spatial_dimensions.end());
      layout.push_back(feature_dimension);
      break;
    default:
      return Internal("Invalid layout %s", DataLayoutString(data_layout));
  }
  return LayoutUtil::MakeLayoutFromMajorToMinor(layout);
}
}  
absl::StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      DataLayout input, FilterLayout filter,
                                      DataLayout output) {
  TF_ASSIGN_OR_RETURN(
      Layout input_layout,
      DataLayoutToXlaLayout(input, dnums.input_batch_dimension(),
                            dnums.input_feature_dimension(),
                            dnums.input_spatial_dimensions()));
  TF_ASSIGN_OR_RETURN(
      Layout output_layout,
      DataLayoutToXlaLayout(input, dnums.output_batch_dimension(),
                            dnums.output_feature_dimension(),
                            dnums.output_spatial_dimensions()));
  std::vector<int64_t> filter_layout;
  switch (filter) {
    case FilterLayout::kOutputInputYX:  
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      break;
    case FilterLayout::kOutputInputYX4:  
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(FindMissingDnum(filter_layout));
      break;
    case FilterLayout::kOutputYXInput:  
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      break;
    default:
      return Internal("Invalid filter layout %s for conv with dnums %s,",
                      FilterLayoutString(filter),
                      ConvolutionDimensionNumbersToString(dnums));
  }
  return std::make_tuple(input_layout,
                         LayoutUtil::MakeLayoutFromMajorToMinor(filter_layout),
                         output_layout);
}
absl::StatusOr<std::tuple<DataLayout, FilterLayout, DataLayout>>
XlaConvShapesToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                     const Shape& input, const Shape& filter,
                                     const Shape& output) {
  CHECK(input.has_layout());
  CHECK(filter.has_layout());
  CHECK(output.has_layout());
  Layout nchw_input, nchw_filter, nchw_output;
  std::tie(nchw_input, nchw_filter, nchw_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX,
                                            FilterLayout::kOutputInputYX,
                                            DataLayout::kBatchDepthYX)
          .value();
  Layout nchw_vect_input, nchw_vect_filter, nchw_vect_output;
  std::tie(nchw_vect_input, nchw_vect_filter, nchw_vect_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX4,
                                            FilterLayout::kOutputInputYX4,
                                            DataLayout::kBatchDepthYX4)
          .value();
  Layout nhwc_input, nhwc_filter, nhwc_output;
  std::tie(nhwc_input, nhwc_filter, nhwc_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchYXDepth,
                                            FilterLayout::kOutputYXInput,
                                            DataLayout::kBatchYXDepth)
          .value();
  DataLayout input_layout;
  if (LayoutUtil::Equal(input.layout(), nchw_input)) {
    input_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(input.layout(), nchw_vect_input)) {
    int64_t vect_size = input.dimensions(input.layout().minor_to_major(0));
    if (vect_size == 4) {
      input_layout = DataLayout::kBatchDepthYX4;
    } else if (vect_size == 32) {
      input_layout = DataLayout::kBatchDepthYX32;
    } else {
      return Internal(
          "Invalid input shape %s for conv with dnums %s.  Most-minor dim "
          "should be 4 or 32, but was %d.",
          ShapeUtil::HumanStringWithLayout(input),
          ConvolutionDimensionNumbersToString(dnums), vect_size);
    }
  } else if (LayoutUtil::Equal(input.layout(), nhwc_input)) {
    input_layout = DataLayout::kBatchYXDepth;
  } else {
    return Internal(
        "Invalid input layout %s for conv with dnums %s; expected one of (%s, "
        "%s, %s)",
        LayoutUtil::HumanString(input.layout()),
        ConvolutionDimensionNumbersToString(dnums), nchw_input.ToString(),
        nchw_vect_input.ToString(), nhwc_input.ToString());
  }
  FilterLayout filter_layout;
  if (LayoutUtil::Equal(filter.layout(), nchw_filter)) {
    filter_layout = FilterLayout::kOutputInputYX;
  } else if (LayoutUtil::Equal(filter.layout(), nchw_vect_filter)) {
    int64_t vect_size = filter.dimensions(filter.layout().minor_to_major(0));
    if (vect_size == 4) {
      filter_layout = FilterLayout::kOutputInputYX4;
    } else if (vect_size == 32) {
      filter_layout = FilterLayout::kOutputInputYX32;
    } else {
      return Internal(
          "Invalid filter shape %s for conv with dnums %s.  Most-minor dim "
          "should be 4 or 32, but was %d.",
          ShapeUtil::HumanStringWithLayout(filter),
          ConvolutionDimensionNumbersToString(dnums), vect_size);
    }
  } else if (LayoutUtil::Equal(filter.layout(), nhwc_filter)) {
    filter_layout = FilterLayout::kOutputYXInput;
  } else {
    return Internal(
        "Invalid filter layout %s for conv with dnums %s, expected one of (%s, "
        "%s, %s)",
        LayoutUtil::HumanString(filter.layout()),
        ConvolutionDimensionNumbersToString(dnums), nchw_filter.ToString(),
        nchw_vect_filter.ToString(), nhwc_filter.ToString());
  }
  DataLayout output_layout;
  if (LayoutUtil::Equal(output.layout(), nchw_output)) {
    output_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(output.layout(), nchw_vect_output)) {
    int64_t vect_size = output.dimensions(output.layout().minor_to_major(0));
    if (vect_size == 4) {
      output_layout = DataLayout::kBatchDepthYX4;
    } else if (vect_size == 32) {
      output_layout = DataLayout::kBatchDepthYX32;
    } else {
      return Internal(
          "Invalid output shape %s for conv with dnums %s.  Most-minor dim "
          "should be 4 or 32, but was %d.",
          ShapeUtil::HumanStringWithLayout(output),
          ConvolutionDimensionNumbersToString(dnums), vect_size);
    }
  } else if (LayoutUtil::Equal(output.layout(), nhwc_output)) {
    output_layout = DataLayout::kBatchYXDepth;
  } else {
    return Internal("Invalid output layout %s for conv with dnums %s",
                    LayoutUtil::HumanString(output.layout()),
                    ConvolutionDimensionNumbersToString(dnums));
  }
  return std::make_tuple(input_layout, filter_layout, output_layout);
}
static std::optional<int64_t> FindVectorizedDim(int64_t rank, int64_t d0,
                                                int64_t d1,
                                                absl::Span<const int64_t> ds) {
  for (int64_t i = 0; i < rank; i++) {
    if (i == d0 || i == d1 || absl::c_linear_search(ds, i)) {
      continue;
    }
    return i;
  }
  return std::nullopt;
}
std::tuple<std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
FindVectorizedFeatureDims(const ConvolutionDimensionNumbers& dnums,
                          const Shape& input, const Shape& filter,
                          const Shape& output) {
  return {
      FindVectorizedDim(input.dimensions_size(), dnums.input_batch_dimension(),
                        dnums.input_feature_dimension(),
                        dnums.input_spatial_dimensions()),
      FindVectorizedDim(filter.dimensions_size(),
                        dnums.kernel_input_feature_dimension(),
                        dnums.kernel_output_feature_dimension(),
                        dnums.kernel_spatial_dimensions()),
      FindVectorizedDim(
          output.dimensions_size(), dnums.output_batch_dimension(),
          dnums.output_feature_dimension(), dnums.output_spatial_dimensions()),
  };
}
absl::Mutex& GetGpuMutex(const se::StreamExecutor* stream_exec) {
  static absl::Mutex mu(absl::kConstInit);
  static auto* mutexes =
      new std::map<std::pair<const se::Platform*,  int64_t>,
                   absl::Mutex>();
  absl::MutexLock global_lock(&mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->GetPlatform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;
  return it->second;
}
absl::StatusOr<std::unique_ptr<se::Kernel>> CreateKernel(
    absl::string_view kernel_name, uint64_t num_args, absl::string_view ptx,
    absl::Span<const uint8_t> cubin_data, se::StreamExecutor* stream_exec,
    uint32_t shared_mem_bytes) {
  se::MultiKernelLoaderSpec loader_spec(num_args);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);
  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(cubin_data, kernel_name);
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                      stream_exec->LoadKernel(loader_spec));
  se::KernelMetadata m;
  m.set_shared_memory_bytes(shared_mem_bytes);
  kernel->set_metadata(m);
  return kernel;
}
absl::Status ExecuteKernelOnStream(const se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::KernelArgsPackedArrayBase> kernel_args,
      se::PackKernelArgs(args, kernel.metadata()));
  return stream->Launch(dims.thread_counts_per_block(), dims.block_counts(),
                        kernel, *kernel_args);
}
absl::Status ExecuteKernelOnStream(const se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   const se::ClusterDim& cluster_dim,
                                   se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::KernelArgsPackedArrayBase> kernel_args,
      se::PackKernelArgs(args, kernel.metadata()));
  return stream->Launch(dims.thread_counts_per_block(), dims.block_counts(),
                        cluster_dim, kernel, *kernel_args);
}
template <typename T, typename Generator>
typename std::enable_if<std::is_integral<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) =
    delete;
template <typename T, typename Generator>
typename std::enable_if<std::is_floating_point<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) {
  return std::uniform_real_distribution<T>(lhs, rhs)(*gen);
}
namespace repeat_buffer_kernel {
void* kernel();
}
template <typename T>
static void InitializeTypedBuffer(se::Stream* stream,
                                  se::DeviceMemoryBase buffer,
                                  int64_t* rng_state) {
  constexpr int host_buffer_size = 10069;
  static std::vector<T>* host_buffer = [&] {
    auto* ret = new std::vector<T>(host_buffer_size);
    std::mt19937 gen;
    for (auto& element : *ret) {
      constexpr bool kIsIntegral = std::numeric_limits<T>::is_integer;
      constexpr bool kIsLowRange =
          !kIsIntegral && std::numeric_limits<T>::max_exponent <=
                              std::numeric_limits<Eigen::half>::max_exponent;
      using RandomType = typename std::conditional<std::is_same_v<T, double>,
                                                   double, float>::type;
      auto upper_bound = RandomType(kIsLowRange ? 0.1 : 1.0);
      auto rand_val = UniformDistribution(RandomType(0), upper_bound, &gen);
      element = T(kIsIntegral ? rand_val + 0.5 : rand_val);
    }
    return ret;
  }();
  CHECK_EQ(0, buffer.size() % sizeof(T));
  int64_t elements_to_fill = buffer.size() / sizeof(T);
  int64_t host_index = *rng_state;
  CHECK_LT(host_index, host_buffer_size);
  *rng_state = (*rng_state + elements_to_fill) % host_buffer_size;
  int64_t first_size =
      std::min<int64_t>(host_buffer_size - host_index, elements_to_fill);
  TF_CHECK_OK(stream->Memcpy(&buffer, host_buffer->data() + host_index,
                             first_size * sizeof(T)));
  elements_to_fill -= first_size;
  if (elements_to_fill == 0) {
    return;
  }
  int64_t second_size = std::min<int64_t>(host_index, elements_to_fill);
  CHECK_LE(first_size + second_size, host_buffer_size);
  se::DeviceMemoryBase mem =
      buffer.GetByteSlice(first_size * sizeof(T), second_size * sizeof(T));
  TF_CHECK_OK(stream->Memcpy(&mem, host_buffer->data(), mem.size()));
  elements_to_fill -= second_size;
  if (elements_to_fill == 0) {
    return;
  }
#ifdef GOOGLE_CUDA
  CHECK_EQ(elements_to_fill, buffer.size() / sizeof(T) - host_buffer_size);
  se::StreamExecutor* executor = stream->parent();
  auto kernel =
      se::TypedKernelFactory<se::DeviceMemoryBase, int64_t, int64_t>::Create(
          executor, "RepeatBufferKernel", repeat_buffer_kernel::kernel());
  if (!kernel.ok()) {
    LOG(FATAL) << "Could not create RepeatBufferKernel: " << kernel.status();
  }
  constexpr int64_t host_buffer_bytes = host_buffer_size * sizeof(T);
  constexpr int threads_per_block = 256;
  constexpr int blocks_per_grid =
      (host_buffer_bytes + threads_per_block - 1) / threads_per_block;
  TF_CHECK_OK(stream->ThenLaunch(se::ThreadDim(threads_per_block, 1, 1),
                                 se::BlockDim(blocks_per_grid, 1, 1), *kernel,
                                 buffer, host_buffer_bytes,
                                 static_cast<int64_t>(buffer.size())));
#endif
}
void InitializeBuffer(se::Stream* stream, PrimitiveType buffer_type,
                      int64_t* rng_state, se::DeviceMemoryBase buffer) {
  return primitive_util::PrimitiveTypeSwitch<void>(
      [&](auto primitive_type_constant) -> void {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant) ||
                      primitive_util::IsIntegralType(primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return InitializeTypedBuffer<NativeT>(stream, buffer, rng_state);
        }
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return InitializeTypedBuffer<typename NativeT::value_type>(
              stream, buffer, rng_state);
        }
        if constexpr (primitive_type_constant == PRED) {
          return InitializeTypedBuffer<int8_t>(stream, buffer, rng_state);
        }
        LOG(FATAL) << "Unexpected type: "
                   << primitive_util::LowercasePrimitiveTypeName(buffer_type);
      },
      buffer_type);
}
absl::StatusOr<se::dnn::ConvolutionKind> GetDNNConvKindFromCudnnConvKind(
    CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kBackwardFilter:
      return se::dnn::BACKWARD_FILTER;
    case CudnnConvKind::kBackwardInput:
      return se::dnn::BACKWARD_DATA;
    case CudnnConvKind::kForward:
      return se::dnn::FORWARD;
    case CudnnConvKind::kForwardActivation:
      return se::dnn::FORWARD_BIAS_ACTIVATION;
    case CudnnConvKind::kForwardGraph:
      return se::dnn::FORWARD_GRAPH;
    default:
      break;
  }
  return Internal("Unexpected convolution kind");
}
absl::StatusOr<se::dnn::NormKind> GetDNNNormKindFromCudnnNormKind(
    CudnnNormKind kind) {
  switch (kind) {
    case CudnnNormKind::kLayerForwardInfer:
      return se::dnn::LAYER_FWD_INFER;
    case CudnnNormKind::kLayerForwardTrain:
      return se::dnn::LAYER_FWD_TRAIN;
    case CudnnNormKind::kLayerBackward:
      return se::dnn::LAYER_BWD;
    default:
      return Internal("Unexpected norm kind");
  }
}
absl::StatusOr<se::dnn::FMHAMaskKind> GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(
    CudnnfMHAMaskKind kind) {
  switch (kind) {
    case CudnnfMHAMaskKind::kNoMask:
      return se::dnn::NO_MASK;
    case CudnnfMHAMaskKind::kPadding:
      return se::dnn::PADDING;
    case CudnnfMHAMaskKind::kCausal:
      return se::dnn::CAUSAL;
    case CudnnfMHAMaskKind::kPaddingCausal:
      return se::dnn::PADDING_CAUSAL;
    case CudnnfMHAMaskKind::kAlibi:
      return se::dnn::ALIBI;
    default:
      return Internal("Unexpected fmha mask kind");
  }
}
absl::StatusOr<se::dnn::DataType> GetDNNDataTypeFromPrimitiveType(
    PrimitiveType type) {
  switch (type) {
    case F16:
      return se::dnn::ToDataType<Eigen::half>::value;
    case F32:
      return se::dnn::ToDataType<float>::value;
    case F64:
      return se::dnn::ToDataType<double>::value;
    case S8:
      return se::dnn::ToDataType<int8_t>::value;
    case S32:
      return se::dnn::ToDataType<int32_t>::value;
    case BF16:
      return se::dnn::ToDataType<Eigen::bfloat16>::value;
    case F8E4M3FN:
      return se::dnn::ToDataType<tsl::float8_e4m3fn>::value;
    case F8E5M2:
      return se::dnn::ToDataType<tsl::float8_e5m2>::value;
    default:
      break;
  }
  return Internal("Unsupported datatype");
}
bool RequireDeterminism(const HloModuleConfig& config) {
  return config.debug_options().xla_gpu_deterministic_ops() ||
         config.debug_options().xla_gpu_exclude_nondeterministic_ops();
}
namespace {
std::vector<AutotuneResult> KeepNonFailures(
    absl::Span<AutotuneResult const> profile_results) {
  std::vector<AutotuneResult> filtered_results;
  absl::c_copy_if(profile_results, std::back_inserter(filtered_results),
                  [](const AutotuneResult& r) {
                    return !r.has_failure() ||
                           r.failure().kind() == AutotuneResult::WRONG_RESULT;
                  });
  return filtered_results;
}
absl::Status AllAlgorithmsFailedInternalError(
    std::optional<std::string_view> instr_str,
    absl::Span<AutotuneResult const> profile_results) {
  std::ostringstream msg;
  if (instr_str.has_value()) {
    msg << "All algorithms tried for " << instr_str.value()
        << " failed. Falling back to default algorithm.  Per-algorithm "
           "errors:";
  } else {
    msg << "All algorithms failed. Falling back to the default algorithm. "
        << "Per-algorithm errors:";
  }
  for (const auto& result : profile_results) {
    msg << "\n  " << result.failure().msg();
  }
  return Internal("%s", msg.str());
}
absl::Status NoAlgorithmSuppliedInternalError(
    std::optional<std::string_view> instr_str) {
  std::ostringstream msg;
  if (instr_str.has_value()) {
    msg << "There are no algorithm candidates for computing: \n  "
        << instr_str.value()
        << "\nThis likely means that the instruction shape is not supported by "
           "the target GPU library.";
  } else {
    msg << "There are no algorithm candidates for computing the instruction.\n"
           "This likely means that the instruction shape is not supported by "
           "the target GPU library.";
  }
  return Internal("%s", msg.str());
}
void SortAutotuningResultsByRunTime(std::vector<AutotuneResult>& results) {
  absl::c_sort(results,
               [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
                 return tsl::proto_utils::FromDurationProto(lhs.run_time()) <
                        tsl::proto_utils::FromDurationProto(rhs.run_time());
               });
}
absl::Span<AutotuneResult const> TopResultsWithinMeasurementError(
    std::vector<AutotuneResult>& results_sorted_by_runtime) {
  constexpr absl::Duration kMeasurementError = absl::Microseconds(4);
  absl::Duration min_time = tsl::proto_utils::FromDurationProto(
      results_sorted_by_runtime.front().run_time());
  absl::Duration limit_time = min_time + kMeasurementError;
  auto limit_time_it = absl::c_find_if(
      results_sorted_by_runtime, [limit_time](const AutotuneResult& x) {
        return tsl::proto_utils::FromDurationProto(x.run_time()) > limit_time;
      });
  return absl::MakeSpan(&*results_sorted_by_runtime.begin(), &*limit_time_it);
}
}  
absl::StatusOr<AutotuneResult> PickBestResult(
    absl::Span<AutotuneResult const> profile_results,
    std::optional<std::string_view> instr_str,
    HloModuleConfig hlo_module_config) {
  if (profile_results.empty()) {
    return NoAlgorithmSuppliedInternalError(instr_str);
  }
  std::vector<AutotuneResult> filtered_results =
      KeepNonFailures(profile_results);
  if (filtered_results.empty()) {
    return AllAlgorithmsFailedInternalError(instr_str, profile_results);
  }
  if (RequireDeterminism(hlo_module_config)) {
    return *filtered_results.begin();
  }
  SortAutotuningResultsByRunTime(filtered_results);
  auto top_within_error = TopResultsWithinMeasurementError(filtered_results);
  return *absl::c_min_element(top_within_error, [](const AutotuneResult& lhs,
                                                   const AutotuneResult& rhs) {
    return lhs.scratch_bytes() < rhs.scratch_bytes();
  });
}
}  
}  