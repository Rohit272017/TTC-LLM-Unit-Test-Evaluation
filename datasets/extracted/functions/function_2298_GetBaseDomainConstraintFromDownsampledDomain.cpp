#include "tensorstore/driver/downsample/downsample.h"
#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <mutex>
#include <utility>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/downsample/downsample_array.h"
#include "tensorstore/driver/downsample/downsample_method_json_binder.h"  
#include "tensorstore/driver/downsample/downsample_nditerable.h"
#include "tensorstore/driver/downsample/downsample_util.h"
#include "tensorstore/driver/downsample/grid_occupancy_map.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/read.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/std_vector.h"  
#include "tensorstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/std_vector.h"  
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_downsample {
namespace {
using ::tensorstore::internal::DriverPtr;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::LockCollection;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::OpenTransactionPtr;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::TransformedDriverSpec;
namespace jb = tensorstore::internal_json_binding;
Result<IndexDomain<>> GetBaseDomainConstraintFromDownsampledDomain(
    IndexDomain<> downsampled_domain,
    tensorstore::span<const Index> downsample_factors) {
  assert(downsampled_domain.valid());
  const DimensionIndex rank = downsampled_domain.rank();
  assert(rank == downsample_factors.size());
  IndexDomainBuilder builder(rank);
  builder.labels(downsampled_domain.labels());
  auto& implicit_lower_bounds = builder.implicit_lower_bounds();
  auto& implicit_upper_bounds = builder.implicit_upper_bounds();
  auto origin = builder.origin();
  auto shape = builder.shape();
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (downsample_factors[i] != 1) {
      implicit_lower_bounds[i] = true;
      implicit_upper_bounds[i] = true;
      origin[i] = -kInfIndex;
      shape[i] = kInfSize;
    } else {
      implicit_lower_bounds[i] = downsampled_domain.implicit_lower_bounds()[i];
      implicit_upper_bounds[i] = downsampled_domain.implicit_upper_bounds()[i];
      origin[i] = downsampled_domain.origin()[i];
      shape[i] = downsampled_domain.shape()[i];
    }
  }
  return builder.Finalize();
}
Result<IndexTransform<>> GetBaseTransformForDownsampledTransform(
    IndexTransformView<> base_transform,
    IndexTransformView<> downsampled_transform,
    tensorstore::span<const Index> downsample_factors,
    DownsampleMethod downsample_method) {
  if (downsample_method == DownsampleMethod::kStride) {
    return base_transform | tensorstore::AllDims().Stride(downsample_factors) |
           downsampled_transform;
  }
  PropagatedIndexTransformDownsampling propagated;
  TENSORSTORE_RETURN_IF_ERROR(
      internal_downsample::PropagateAndComposeIndexTransformDownsampling(
          downsampled_transform, base_transform, downsample_factors,
          propagated));
  return std::move(propagated.transform);
}
class DownsampleDriverSpec
    : public internal::RegisteredDriverSpec<DownsampleDriverSpec,
                                            internal::DriverSpec> {
 public:
  constexpr static char id[] = "downsample";
  TransformedDriverSpec base;
  std::vector<Index> downsample_factors;
  DownsampleMethod downsample_method;
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x), x.base,
             x.downsample_factors, x.downsample_method);
  };
  absl::Status InitializeFromBase() {
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(RankConstraint{internal::GetRank(this->base)}));
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(this->base.driver_spec->schema.dtype()));
    return absl::OkStatus();
  }
  absl::Status ValidateDownsampleFactors() {
    TENSORSTORE_RETURN_IF_ERROR(
        this->schema.Set(RankConstraint(this->downsample_factors.size())));
    return absl::OkStatus();
  }
  absl::Status ValidateDownsampleMethod() {
    auto dtype = this->schema.dtype();
    if (!dtype.valid()) return absl::OkStatus();
    return internal_downsample::ValidateDownsampleMethod(
        dtype, this->downsample_method);
  }
  OpenMode open_mode() const override { return base.driver_spec->open_mode(); }
  absl::Status ApplyOptions(SpecOptions&& options) override {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.dtype()));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.rank()));
    auto transform = base.transform;
    if (!transform.valid()) {
      transform = tensorstore::IdentityTransform(downsample_factors.size());
    }
    if (options.domain().valid()) {
      TENSORSTORE_RETURN_IF_ERROR(schema.Set(options.domain()));
      TENSORSTORE_ASSIGN_OR_RETURN(auto base_domain,
                                   GetBaseDomainConstraintFromDownsampledDomain(
                                       options.domain(), downsample_factors));
      TENSORSTORE_RETURN_IF_ERROR(options.Override(std::move(base_domain)));
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, transform | AllDims().Stride(downsample_factors));
    TENSORSTORE_RETURN_IF_ERROR(options.TransformInputSpaceSchema(transform));
    return internal::TransformAndApplyOptions(base, std::move(options));
  }
  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection<&DownsampleDriverSpec::base>()(
                       is_loading,
                       JsonSerializationOptions(options, obj->schema.dtype(),
                                                obj->schema.rank()),
                       obj, j);
                 }),
      jb::Initialize([](auto* obj) { return obj->InitializeFromBase(); }),
      jb::Member("downsample_factors",
                 jb::Validate(
                     [](const auto& options, auto* obj) {
                       return obj->ValidateDownsampleFactors();
                     },
                     jb::Projection<&DownsampleDriverSpec::downsample_factors>(
                         jb::Array(jb::Integer<Index>(1))))),
      jb::Member(
          "downsample_method",
          jb::Validate(
              [](const auto& options, auto* obj) {
                return obj->ValidateDownsampleMethod();
              },
              jb::Projection<&DownsampleDriverSpec::downsample_method>())),
      jb::Initialize([](auto* obj) {
        SpecOptions base_options;
        static_cast<Schema&>(base_options) = std::exchange(obj->schema, {});
        return obj->ApplyOptions(std::move(base_options));
      }));
  Result<IndexDomain<>> GetDomain() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain,
                                 internal::GetEffectiveDomain(base));
    if (!domain.valid()) {
      return schema.domain();
    }
    if (domain.rank() != downsample_factors.size()) {
      return absl::InternalError(tensorstore::StrCat(
          "Domain of base TensorStore has rank (", domain.rank(),
          ") but expected ", downsample_factors.size()));
    }
    auto downsampled_domain = internal_downsample::DownsampleDomain(
        domain, downsample_factors, downsample_method);
    return MergeIndexDomains(std::move(downsampled_domain), schema.domain());
  }
  Result<ChunkLayout> GetChunkLayout() const override {
    return internal::GetEffectiveChunkLayout(base) |
           AllDims().Stride(downsample_factors);
  }
  Result<CodecSpec> GetCodec() const override {
    return internal::GetEffectiveCodec(base);
  }
  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return {std::in_place};
  }
  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 internal::GetEffectiveDimensionUnits(base));
    if (!dimension_units.empty()) {
      tensorstore::span<const Index> downsample_factors =
          this->downsample_factors;
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto transform,
          tensorstore::IdentityTransform(downsample_factors.size()) |
              tensorstore::AllDims().Stride(downsample_factors));
      dimension_units =
          TransformOutputDimensionUnits(transform, std::move(dimension_units));
    }
    return dimension_units;
  }
  kvstore::Spec GetKvstore() const override {
    return base.driver_spec->GetKvstore();
  }
  Result<TransformedDriverSpec> GetBase(
      IndexTransformView<> transform) const override {
    TransformedDriverSpec new_base;
    new_base.driver_spec = base.driver_spec;
    if (transform.valid()) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          new_base.transform,
          GetBaseTransformForDownsampledTransform(
              base.transform.valid()
                  ? base.transform
                  : tensorstore::IdentityTransform(downsample_factors.size()),
              transform, downsample_factors, downsample_method));
    }
    return new_base;
  }
  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override {
    if (!!(request.read_write_mode & ReadWriteMode::write)) {
      return absl::InvalidArgumentError("only reading is supported");
    }
    request.read_write_mode = ReadWriteMode::read;
    return MapFutureValue(
        InlineExecutor{},
        [spec = internal::DriverSpec::PtrT<const DownsampleDriverSpec>(this)](
            internal::Driver::Handle handle)
            -> Result<internal::Driver::Handle> {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto downsampled_handle,
              MakeDownsampleDriver(std::move(handle), spec->downsample_factors,
                                   spec->downsample_method));
          if (auto domain = spec->schema.domain(); domain.valid()) {
            TENSORSTORE_RETURN_IF_ERROR(
                MergeIndexDomains(domain,
                                  downsampled_handle.transform.domain()),
                tensorstore::MaybeAnnotateStatus(
                    _, "downsampled domain does not match domain in schema"));
          }
          return downsampled_handle;
        },
        internal::OpenDriver(base, std::move(request)));
  }
};
class DownsampleDriver
    : public internal::RegisteredDriver<DownsampleDriver,
                                        internal::Driver> {
 public:
  Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    auto driver_spec = internal::DriverSpec::Make<DownsampleDriverSpec>();
    driver_spec->context_binding_state_ = ContextBindingState::bound;
    TENSORSTORE_ASSIGN_OR_RETURN(
        driver_spec->base,
        base_driver_->GetBoundSpec(std::move(transaction), base_transform_));
    driver_spec->downsample_factors = downsample_factors_;
    driver_spec->downsample_method = downsample_method_;
    TENSORSTORE_RETURN_IF_ERROR(driver_spec->InitializeFromBase());
    TransformedDriverSpec spec;
    spec.transform = transform;
    spec.driver_spec = std::move(driver_spec);
    return spec;
  }
  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto strided_base_transform,
                                 GetStridedBaseTransform());
    return base_driver_->GetChunkLayout(strided_base_transform) | transform;
  }
  Result<CodecSpec> GetCodec() override { return base_driver_->GetCodec(); }
  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    if (downsample_method_ == DownsampleMethod::kStride) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto strided_transform,
                                   GetStridedBaseTransform() | transform);
      return base_driver_->GetFillValue(strided_transform);
    }
    PropagatedIndexTransformDownsampling propagated;
    TENSORSTORE_RETURN_IF_ERROR(
        internal_downsample::PropagateAndComposeIndexTransformDownsampling(
            transform, base_transform_, downsample_factors_, propagated));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto fill_value, base_driver_->GetFillValue(propagated.transform));
    if (!fill_value.valid()) return {std::in_place};
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto broadcast_fill_value,
        BroadcastArray(std::move(fill_value),
                       propagated.transform.domain().box()));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto downsampled_fill_value,
        internal_downsample::DownsampleArray(
            broadcast_fill_value, propagated.input_downsample_factors,
            downsample_method_));
    return UnbroadcastArray(downsampled_fill_value);
  }
  Result<DimensionUnitsVector> GetDimensionUnits() override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto dimension_units,
                                 base_driver_->GetDimensionUnits());
    TENSORSTORE_ASSIGN_OR_RETURN(auto strided_base_transform,
                                 GetStridedBaseTransform());
    return TransformOutputDimensionUnits(strided_base_transform,
                                         std::move(dimension_units));
  }
  KvStore GetKvstore(const Transaction& transaction) override {
    return base_driver_->GetKvstore(transaction);
  }
  Result<internal::DriverHandle> GetBase(
      ReadWriteMode read_write_mode, IndexTransformView<> transform,
      const Transaction& transaction) override {
    internal::DriverHandle base_handle;
    base_handle.driver = base_driver_;
    base_handle.driver.set_read_write_mode(read_write_mode);
    base_handle.transaction = transaction;
    TENSORSTORE_ASSIGN_OR_RETURN(base_handle.transform,
                                 GetBaseTransformForDownsampledTransform(
                                     base_transform_, transform,
                                     downsample_factors_, downsample_method_));
    return base_handle;
  }
  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override;
  explicit DownsampleDriver(DriverPtr base, IndexTransform<> base_transform,
                            tensorstore::span<const Index> downsample_factors,
                            DownsampleMethod downsample_method)
      : base_driver_(std::move(base)),
        base_transform_(std::move(base_transform)),
        downsample_factors_(downsample_factors.begin(),
                            downsample_factors.end()),
        downsample_method_(downsample_method) {}
  DataType dtype() override { return base_driver_->dtype(); }
  DimensionIndex rank() override { return base_transform_.input_rank(); }
  Executor data_copy_executor() override {
    return base_driver_->data_copy_executor();
  }
  void Read(ReadRequest request, ReadChunkReceiver receiver) override;
  Result<IndexTransform<>> GetStridedBaseTransform() {
    return base_transform_ | tensorstore::AllDims().Stride(downsample_factors_);
  }
  Future<IndexTransform<>> ResolveBounds(ResolveBoundsRequest request) override;
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base_driver_, x.base_transform_, x.downsample_factors_,
             x.downsample_method_);
  };
  DriverPtr base_driver_;
  IndexTransform<> base_transform_;
  std::vector<Index> downsample_factors_;
  DownsampleMethod downsample_method_;
};
Future<IndexTransform<>> DownsampleDriver::ResolveBounds(
    ResolveBoundsRequest request) {
  return MapFutureValue(
      InlineExecutor{},
      [self = IntrusivePtr<DownsampleDriver>(this),
       transform = std::move(request.transform)](
          IndexTransform<> base_transform) -> Result<IndexTransform<>> {
        Box<dynamic_rank(internal::kNumInlinedDims)> downsampled_bounds(
            base_transform.input_rank());
        internal_downsample::DownsampleBounds(
            base_transform.domain().box(), downsampled_bounds,
            self->downsample_factors_, self->downsample_method_);
        return tensorstore::PropagateBoundsToTransform(
            downsampled_bounds, base_transform.implicit_lower_bounds(),
            base_transform.implicit_upper_bounds(), std::move(transform));
      },
      base_driver_->ResolveBounds({std::move(request.transaction),
                                   base_transform_,
                                   std::move(request.options)}));
}
struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  IntrusivePtr<DownsampleDriver> self_;
  internal::ReadChunkReceiver receiver_;
  absl::Mutex mutex_;
  SharedOffsetArray<void> data_buffer_;
  Index remaining_elements_;
  internal_downsample::GridOccupancyTracker independently_emitted_chunks_;
  absl::InlinedVector<Index, internal::kNumInlinedDims> downsample_factors_;
  DimensionIndex original_input_rank_;
  IndexDomain<> base_transform_domain_;
  AnyCancelReceiver on_cancel_;
  absl::Status error_;
  bool done_signal_received_ = false;
  bool done_sent_ = false;
  bool canceled_ = false;
  size_t chunks_in_progress_ = 0;
  void Cancel() {
    std::lock_guard<ReadState> guard(*this);
    canceled_ = true;
  }
  void lock() ABSL_NO_THREAD_SAFETY_ANALYSIS { mutex_.Lock(); }
  void unlock() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    bool has_error = !error_.ok();
    bool send_done = !done_sent_ && chunks_in_progress_ == 0 &&
                     (done_signal_received_ || has_error);
    if (send_done) done_sent_ = true;
    AnyCancelReceiver on_cancel;
    if (canceled_ && on_cancel_) {
      on_cancel = std::move(on_cancel_);
    }
    mutex_.Unlock();
    if (on_cancel) on_cancel();
    if (!send_done) return;
    if (has_error) {
      execution::set_error(receiver_, error_);
    } else {
      execution::set_done(receiver_);
    }
    execution::set_stopping(receiver_);
  }
  void SetError(absl::Status error, size_t decrement_chunks_in_progress = 0) {
    std::lock_guard<ReadState> guard(*this);
    chunks_in_progress_ -= decrement_chunks_in_progress;
    if (!error_.ok()) return;
    error_ = std::move(error);
    canceled_ = true;
  }
  void EmitBufferedChunkForBox(BoxView<> base_domain);
  void EmitBufferedChunks();
};
struct BufferedReadChunkImpl {
  internal::IntrusivePtr<ReadState> state_;
  absl::Status operator()(LockCollection& lock_collection) const {
    return absl::OkStatus();
  }
  Result<NDIterable::Ptr> operator()(internal::ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     internal::Arena* arena) const {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated,
        internal_downsample::PropagateIndexTransformDownsampling(
            chunk_transform, state_->data_buffer_.domain(),
            state_->downsample_factors_));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transformed_array,
        MakeTransformedArray(state_->data_buffer_,
                             std::move(propagated.transform)));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_nditerable,
        GetTransformedArrayNDIterable(transformed_array, arena));
    return internal_downsample::DownsampleNDIterable(
        std::move(base_nditerable), transformed_array.domain().box(),
        propagated.input_downsample_factors, state_->self_->downsample_method_,
        chunk_transform.input_rank(), arena);
  }
};
IndexTransform<> GetDownsampledRequestIdentityTransform(
    BoxView<> base_domain, tensorstore::span<const Index> downsample_factors,
    DownsampleMethod downsample_method, DimensionIndex request_rank) {
  assert(base_domain.rank() == downsample_factors.size());
  assert(request_rank <= base_domain.rank());
  IndexTransformBuilder builder(base_domain.rank(), request_rank);
  internal_downsample::DownsampleBounds(base_domain, builder.input_bounds(),
                                        downsample_factors, downsample_method);
  builder.output_identity_transform();
  return builder.Finalize().value();
}
void ReadState::EmitBufferedChunkForBox(BoxView<> base_domain) {
  auto request_transform = GetDownsampledRequestIdentityTransform(
      base_domain, downsample_factors_, self_->downsample_method_,
      original_input_rank_);
  ReadChunk downsampled_chunk;
  downsampled_chunk.transform =
      IdentityTransform(request_transform.domain().box());
  downsampled_chunk.impl = BufferedReadChunkImpl{IntrusivePtr<ReadState>(this)};
  execution::set_value(receiver_, std::move(downsampled_chunk),
                       std::move(request_transform));
}
void ReadState::EmitBufferedChunks() {
  if (independently_emitted_chunks_.occupied_chunks.empty()) {
    EmitBufferedChunkForBox(base_transform_domain_.box());
  } else {
    internal_downsample::GridOccupancyMap emitted_chunk_map(
        std::move(independently_emitted_chunks_), base_transform_domain_.box());
    const DimensionIndex rank = emitted_chunk_map.rank();
    Index grid_cell[kMaxRank];
    tensorstore::span<Index> grid_cell_span(&grid_cell[0], rank);
    Box<dynamic_rank(internal::kNumInlinedDims)> grid_cell_domain;
    grid_cell_domain.set_rank(rank);
    emitted_chunk_map.InitializeCellIterator(grid_cell_span);
    do {
      if (!emitted_chunk_map.GetGridCellDomain(grid_cell_span,
                                               grid_cell_domain)) {
        continue;
      }
      EmitBufferedChunkForBox(grid_cell_domain);
    } while (emitted_chunk_map.AdvanceCellIterator(grid_cell_span));
  }
  {
    std::lock_guard<ReadState> guard(*this);
    --chunks_in_progress_;
  }
}
struct IndependentReadChunkImpl {
  internal::IntrusivePtr<ReadState> state_;
  internal::ReadChunk base_chunk_;
  absl::Status operator()(LockCollection& lock_collection) {
    return base_chunk_.impl(lock_collection);
  }
  Result<NDIterable::Ptr> operator()(internal::ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     internal::Arena* arena) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto propagated,
        internal_downsample::PropagateIndexTransformDownsampling(
            chunk_transform, state_->base_transform_domain_.box(),
            state_->downsample_factors_));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_transform,
        ComposeTransforms(base_chunk_.transform, propagated.transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_nditerable,
        base_chunk_.impl(internal::ReadChunk::BeginRead{},
                         std::move(base_transform), arena));
    return internal_downsample::DownsampleNDIterable(
        std::move(base_nditerable), propagated.transform.domain().box(),
        propagated.input_downsample_factors, state_->self_->downsample_method_,
        chunk_transform.input_rank(), arena);
  }
};
bool MaybeEmitIndependentReadChunk(
    ReadState& state, ReadChunk& base_chunk,
    IndexTransformView<> base_request_transform) {
  if (!internal_downsample::CanDownsampleIndexTransform(
          base_request_transform, state.base_transform_domain_.box(),
          state.downsample_factors_)) {
    return false;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto inverse_request_transform,
                               InverseTransform(base_request_transform), false);
  TENSORSTORE_ASSIGN_OR_RETURN(
      base_chunk.transform,
      ComposeTransforms(base_chunk.transform, inverse_request_transform),
      false);
  const Index num_elements = base_chunk.transform.domain().num_elements();
  bool emit_buffered_chunk;
  {
    absl::MutexLock lock(&state.mutex_);
    bool has_data_buffer =
        state.data_buffer_.byte_strided_origin_pointer() != nullptr;
    bool remaining_data = (state.remaining_elements_ -= num_elements) != 0;
    emit_buffered_chunk = (!remaining_data && has_data_buffer);
    if (has_data_buffer || remaining_data) {
      state.independently_emitted_chunks_.MarkOccupied(
          base_chunk.transform.domain().box());
    }
  }
  internal::ReadChunk downsampled_chunk;
  auto request_transform = GetDownsampledRequestIdentityTransform(
      base_chunk.transform.domain().box(), state.downsample_factors_,
      state.self_->downsample_method_, state.original_input_rank_);
  downsampled_chunk.impl = IndependentReadChunkImpl{
      internal::IntrusivePtr<ReadState>(&state), std::move(base_chunk)};
  downsampled_chunk.transform =
      IdentityTransform(request_transform.domain().box());
  execution::set_value(state.receiver_, std::move(downsampled_chunk),
                       request_transform);
  if (emit_buffered_chunk) {
    state.self_->data_copy_executor()(
        [state = internal::IntrusivePtr<ReadState>(&state)] {
          state->EmitBufferedChunks();
        });
  } else {
    std::lock_guard<ReadState> guard(state);
    --state.chunks_in_progress_;
  }
  return true;
}
struct ReadReceiverImpl {
  internal::IntrusivePtr<ReadState> state_;
  void set_starting(AnyCancelReceiver on_cancel) {
    {
      absl::MutexLock lock(&state_->mutex_);
      if (!state_->canceled_) {
        state_->on_cancel_ = std::move(on_cancel);
        return;
      }
    }
    if (on_cancel) on_cancel();
  }
  void set_value(ReadChunk chunk, IndexTransform<> cell_transform) {
    if (cell_transform.domain().box().is_empty()) return;
    {
      absl::MutexLock lock(&state_->mutex_);
      if (state_->canceled_) return;
      ++state_->chunks_in_progress_;
    }
    if (MaybeEmitIndependentReadChunk(*state_, chunk, cell_transform)) return;
    state_->self_->data_copy_executor()([state = state_,
                                         chunk = std::move(chunk),
                                         cell_transform = std::move(
                                             cell_transform)]() mutable {
      const Index num_elements = cell_transform.domain().num_elements();
      {
        std::lock_guard<ReadState> guard(*state);
        if (state->canceled_) {
          --state->chunks_in_progress_;
          return;
        }
        if (state->data_buffer_.byte_strided_origin_pointer() == nullptr) {
          state->data_buffer_ =
              AllocateArray(state->base_transform_domain_.box(), c_order,
                            default_init, state->self_->base_driver_->dtype());
        }
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto transformed_data_buffer,
          MakeTransformedArray(state->data_buffer_, std::move(cell_transform)),
          state->SetError(_, 1));
      TENSORSTORE_RETURN_IF_ERROR(
          internal::CopyReadChunk(chunk.impl, chunk.transform,
                                  transformed_data_buffer),
          state->SetError(_, 1));
      {
        std::lock_guard<ReadState> guard(*state);
        bool elements_done = (state->remaining_elements_ -= num_elements) == 0;
        if (state->canceled_ || !elements_done) {
          --state->chunks_in_progress_;
          return;
        }
      }
      state->EmitBufferedChunks();
    });
  }
  void set_error(absl::Status status) { state_->SetError(std::move(status)); }
  void set_done() {
    std::lock_guard<ReadState> guard(*state_);
    state_->done_signal_received_ = true;
  }
  void set_stopping() {
    absl::MutexLock lock(&state_->mutex_);
    state_->on_cancel_ = {};
  }
};
void DownsampleDriver::Read(ReadRequest request, ReadChunkReceiver receiver) {
  if (downsample_method_ == DownsampleMethod::kStride) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        request.transform, GetStridedBaseTransform() | request.transform,
        execution::set_error(FlowSingleReceiver{std::move(receiver)}, _));
    base_driver_->Read(std::move(request), std::move(receiver));
    return;
  }
  auto base_resolve_future = base_driver_->ResolveBounds(
      {request.transaction, base_transform_, {fix_resizable_bounds}});
  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->self_.reset(this);
  state->original_input_rank_ = request.transform.input_rank();
  state->receiver_ = std::move(receiver);
  execution::set_starting(state->receiver_,
                          [state = state.get()] { state->Cancel(); });
  std::move(base_resolve_future)
      .ExecuteWhenReady([state = std::move(state),
                         request = std::move(request)](
                            ReadyFuture<IndexTransform<>> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          state->SetError(std::move(r.status()));
          return;
        }
        IndexTransform<> base_transform = std::move(*r);
        PropagatedIndexTransformDownsampling propagated;
        TENSORSTORE_RETURN_IF_ERROR(
            internal_downsample::PropagateAndComposeIndexTransformDownsampling(
                request.transform, base_transform,
                state->self_->downsample_factors_, propagated),
            state->SetError(_));
        state->remaining_elements_ =
            propagated.transform.domain().num_elements();
        state->downsample_factors_ =
            std::move(propagated.input_downsample_factors);
        state->base_transform_domain_ = propagated.transform.domain();
        auto* state_ptr = state.get();
        request.transform = std::move(propagated.transform);
        state_ptr->self_->base_driver_->Read(
            std::move(request), ReadReceiverImpl{std::move(state)});
      });
}
Future<ArrayStorageStatistics> DownsampleDriver::GetStorageStatistics(
    GetStorageStatisticsRequest request) {
  if (downsample_method_ == DownsampleMethod::kStride) {
    TENSORSTORE_ASSIGN_OR_RETURN(request.transform,
                                 GetStridedBaseTransform() | request.transform);
    return base_driver_->GetStorageStatistics(std::move(request));
  }
  auto [promise, future] = PromiseFuturePair<ArrayStorageStatistics>::Make();
  auto base_resolve_future = base_driver_->ResolveBounds(
      {request.transaction, base_transform_, {fix_resizable_bounds}});
  LinkValue(WithExecutor(
                data_copy_executor(),
                [self = IntrusivePtr<DownsampleDriver>(this),
                 request = std::move(request)](
                    Promise<ArrayStorageStatistics> promise,
                    ReadyFuture<IndexTransform<>> future) mutable {
                  IndexTransform<> base_transform = std::move(future.value());
                  PropagatedIndexTransformDownsampling propagated;
                  TENSORSTORE_RETURN_IF_ERROR(
                      internal_downsample::
                          PropagateAndComposeIndexTransformDownsampling(
                              request.transform, base_transform,
                              self->downsample_factors_, propagated),
                      static_cast<void>(promise.SetResult(_)));
                  request.transform = std::move(propagated.transform);
                  LinkResult(std::move(promise),
                             self->base_driver_->GetStorageStatistics(
                                 std::move(request)));
                }),
            std::move(promise), std::move(base_resolve_future));
  return std::move(future);
}
const internal::DriverRegistration<DownsampleDriverSpec> driver_registration;
}  
}  
namespace internal {
Result<Driver::Handle> MakeDownsampleDriver(
    Driver::Handle base, tensorstore::span<const Index> downsample_factors,
    DownsampleMethod downsample_method) {
  if (downsample_factors.size() != base.transform.input_rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of downsample factors (", downsample_factors.size(),
        ") does not match TensorStore rank (", base.transform.input_rank(),
        ")"));
  }
  if (!(base.driver.read_write_mode() & ReadWriteMode::read)) {
    return absl::InvalidArgumentError(
        "Cannot downsample write-only TensorStore");
  }
  if (std::any_of(downsample_factors.begin(), downsample_factors.end(),
                  [](Index factor) { return factor < 1; })) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Downsample factors ", downsample_factors, " are not all positive"));
  }
  TENSORSTORE_RETURN_IF_ERROR(internal_downsample::ValidateDownsampleMethod(
      base.driver->dtype(), downsample_method));
  auto downsampled_domain =
      internal_downsample::GetDownsampledDomainIdentityTransform(
          base.transform.domain(), downsample_factors, downsample_method);
  base.driver =
      internal::MakeReadWritePtr<internal_downsample::DownsampleDriver>(
          ReadWriteMode::read, std::move(base.driver),
          std::move(base.transform), downsample_factors, downsample_method);
  base.transform = std::move(downsampled_domain);
  return base;
}
}  
Result<Spec> Downsample(const Spec& base_spec,
                        tensorstore::span<const Index> downsample_factors,
                        DownsampleMethod downsample_method) {
  using internal_spec::SpecAccess;
  Spec downsampled_spec;
  auto& impl = SpecAccess::impl(downsampled_spec);
  auto driver_spec =
      internal::DriverSpec::Make<internal_downsample::DownsampleDriverSpec>();
  driver_spec->context_binding_state_ = base_spec.context_binding_state();
  driver_spec->base = SpecAccess::impl(base_spec);
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->InitializeFromBase());
  driver_spec->downsample_factors.assign(downsample_factors.begin(),
                                         downsample_factors.end());
  driver_spec->downsample_method = downsample_method;
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->ValidateDownsampleFactors());
  TENSORSTORE_RETURN_IF_ERROR(driver_spec->ValidateDownsampleMethod());
  impl.driver_spec = std::move(driver_spec);
  if (base_spec.transform().valid()) {
    impl.transform = internal_downsample::GetDownsampledDomainIdentityTransform(
        base_spec.transform().domain(), downsample_factors, downsample_method);
  }
  return downsampled_spec;
}
}  