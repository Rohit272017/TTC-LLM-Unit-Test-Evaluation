#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include "absl/status/status.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/prefetch_autotuner.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tsl/platform/mutex.h"
namespace tensorflow {
namespace data {
 constexpr const char* const PrefetchDatasetOp::kDatasetType;
 constexpr const char* const PrefetchDatasetOp::kInputDataset;
 constexpr const char* const PrefetchDatasetOp::kBufferSize;
 constexpr const char* const PrefetchDatasetOp::kOutputTypes;
 constexpr const char* const PrefetchDatasetOp::kOutputShapes;
 constexpr const char* const PrefetchDatasetOp::kSlackPeriod;
 constexpr const char* const PrefetchDatasetOp::kLegacyAutotune;
 constexpr const char* const PrefetchDatasetOp::kBufferSizeMin;
namespace {
constexpr double kSleepFactor = 0.2;
constexpr char kBuffer[] = "buffer";
constexpr char kStatus[] = "status";
constexpr char kSizeSuffix[] = ".size";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessageSuffix[] = ".error_message";
}  
class PrefetchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
          int64_t slack_period, bool legacy_autotune, int64_t buffer_size_min)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        slack_period_(slack_period),
        legacy_autotune_(legacy_autotune),
        buffer_size_min_(buffer_size_min) {
    input_->Ref();
    random_indexing_compatible_ = absl::OkStatus();
    if (input_ != nullptr) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    }
  }
  ~Dataset() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }
  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }
  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }
  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return input_->Cardinality(options);
  }
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }
  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }
  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    return input_->Get(ctx, index, out_tensors);
  }
  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    AttrValue slack_period_attr;
    b->BuildAttrValue(slack_period_, &slack_period_attr);
    AttrValue legacy_autotune_attr;
    b->BuildAttrValue(legacy_autotune_, &legacy_autotune_attr);
    AttrValue buffer_size_min_attr;
    b->BuildAttrValue(buffer_size_min_, &buffer_size_min_attr);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, buffer_size},
                      {std::make_pair(kSlackPeriod, slack_period_attr),
                       std::make_pair(kLegacyAutotune, legacy_autotune_attr),
                       std::make_pair(kBufferSizeMin, buffer_size_min_attr)},
                      output));
    return absl::OkStatus();
  }
 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          buffer_size_min_(params.dataset->buffer_size_min_),
          legacy_autotune_(params.dataset->legacy_autotune_),
          buffer_size_(std::make_shared<model::SharedState>(
              legacy_autotune_ ? 0 : params.dataset->buffer_size_, mu_,
              cond_var_)) {
      slack_us_ = 0;
    }
    ~Iterator() override {
      CancelThreads();
      if (deregister_fn_) deregister_fn_();
    }
    bool SymbolicCheckpointCompatible() const override { return true; }
    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
      auto_tuner_ = std::make_unique<PrefetchAutotuner>(
          dataset()->buffer_size_, dataset()->buffer_size_min_,
          ctx->ram_budget_manager());
      interleave_depth_ = ctx->interleave_depth();
      if (buffer_size_->value == model::kAutotune) {
        buffer_size_->value = buffer_size_min_;
      }
      cancellation_manager_ = std::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(), [this]() { CancelThreads(); },
          &deregister_fn_));
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      IteratorContext iter_ctx(params);
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          &iter_ctx, this, prefix(), &input_impl_));
      if (ctx->warm_start() && !ctx->is_restoring()) {
        TF_RETURN_IF_ERROR(EnsureThreadsStarted(ctx));
      }
      ctx->MergeCheckpoint(iter_ctx.checkpoint());
      return absl::OkStatus();
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      const auto& stats_aggregator = ctx->stats_aggregator();
      {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(EnsureThreadsStarted(ctx));
        while (buffer_.empty() && !prefetch_thread_finished_ &&
               buffer_limit() != 0) {
          if (legacy_autotune_) {
            auto_tuner_->RecordEmpty();
            buffer_size_->value = auto_tuner_->buffer_limit();
          }
          RecordStop(ctx);
          cond_var_->wait(l);
          RecordStart(ctx);
        }
        if (!buffer_.empty()) {
          return Consume(ctx, out_tensors, end_of_sequence);
        }
        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        DCHECK_EQ(buffer_limit(), 0);
      }
      mutex_lock input_l(input_mu_);
      {
        mutex_lock l(*mu_);
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              stats_utils::BufferSizeScalarName(dataset()->node_name()),
              static_cast<float>(buffer_.size()), num_elements());
          stats_aggregator->AddScalar(
              stats_utils::BufferCapacityScalarName(dataset()->node_name()),
              static_cast<float>(buffer_limit()), num_elements());
        }
      }
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      double buffer_size_min = buffer_size_min_;
      double buffer_size_max = std::numeric_limits<int64_t>::max();
      if (buffer_size_->value != model::kAutotune && buffer_size_->value != 0) {
        buffer_size_min = buffer_size_->value;
        buffer_size_max = buffer_size_->value;
      }
      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          1,
          {model::MakeParameter(kBufferSize, buffer_size_, buffer_size_min,
                                buffer_size_max)},
          legacy_autotune_);
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      if (ctx->symbolic_checkpoint()) {
        return absl::OkStatus();
      }
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kBufferSize, buffer_.size()));
      for (size_t i = 0; i < buffer_.size(); i++) {
        auto& buffer_element = buffer_[i];
        TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
        if (buffer_element.status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              absl::StrCat(prefix(), "::", i),
              absl::StrCat(kBuffer, kSizeSuffix), buffer_element.value.size()));
          for (size_t j = 0; j < buffer_element.value.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                absl::StrCat(prefix(), "::", i),
                absl::StrCat(kBuffer, "[", j, "]"), buffer_element.value[j]));
          }
        }
      }
      return absl::OkStatus();
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      DCHECK(!prefetch_thread_);
      DCHECK(buffer_.empty());
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      if (!ctx->symbolic_checkpoint()) {
        TF_RETURN_IF_ERROR(RestoreBuffer(ctx, reader));
      }
      if (ctx->warm_start()) {
        TF_RETURN_IF_ERROR(EnsureThreadsStarted(ctx));
      }
      cond_var_->notify_all();
      return absl::OkStatus();
    }
    data::TraceMeMetadata GetTraceMeMetadata() const override {
      int64_t limit = -1, size = -1;
      data::TraceMeMetadata result;
      if (mu_->try_lock()) {
        limit = buffer_limit();
        size = buffer_.size();
        if (!buffer_.empty()) {
          std::vector<std::string> shapes(buffer_.front().value.size());
          for (const auto& component : buffer_.front().value) {
            shapes.push_back(component.shape().DebugString());
          }
          result.push_back(std::make_pair("next_element_shapes",
                                          absl::StrJoin(shapes, ",")));
        }
        mu_->unlock();
      }
      result.push_back(std::make_pair(
          "buffer_limit",
          limit == -1
              ? kTraceInfoUnavailable
              : strings::Printf("%lld", static_cast<long long>(limit))));
      result.push_back(std::make_pair(
          "autotune",
          dataset()->buffer_size_ == model::kAutotune ? "true" : "false"));
      result.push_back(std::make_pair(
          "autotune_mode", legacy_autotune_ ? "legacy" : "performance"));
      if (dataset()->slack_period_ > 0) {
        result.push_back(std::make_pair(
            "slack",
            strings::Printf("%lld", static_cast<long long>(slack_us_.load()))));
      }
      result.push_back(std::make_pair(
          "interleave_depth",
          strings::Printf("%lld", static_cast<long long>(interleave_depth_))));
      return result;
    }
   private:
    struct BufferElement {
      explicit BufferElement(IteratorContext* ctx)
          : uid(tensorflow::EnvTime::NowNanos()),
            checkpoint(MemoryCheckpoint{ctx->id_registry()}) {}
      Status status;
      std::vector<Tensor> value;
      int64_t created_us;
      const uint64 uid;
      MemoryCheckpoint checkpoint;
    };
    Status RestoreBuffer(IteratorContext* const ctx,
                         IteratorStateReader* const reader)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      size_t buffer_size;
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kBufferSize, &temp));
        buffer_size = static_cast<size_t>(temp);
      }
      for (size_t i = 0; i < buffer_size; i++) {
        buffer_.emplace_back(ctx);
        auto& buffer_element = buffer_.back();
        TF_RETURN_IF_ERROR(ReadStatus(reader, i, &buffer_element.status));
        if (buffer_element.status.ok()) {
          size_t value_size;
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, kSizeSuffix), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            TF_RETURN_IF_ERROR(
                reader->ReadTensor(ctx->flr(), absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, "[", j, "]"),
                                   &buffer_element.value.back()));
          }
        }
        RecordBufferEnqueue(ctx, buffer_element.value);
      }
      return absl::OkStatus();
    }
    int64_t buffer_limit() const TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (legacy_autotune_) {
        return auto_tuner_->buffer_limit();
      }
      return buffer_size_->value;
    }
    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
    }
    Status Consume(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        double buffer_limit_ = buffer_limit();
        stats_aggregator->AddToHistogram(
            stats_utils::BufferUtilizationHistogramName(dataset()->node_name()),
            {static_cast<float>(buffer_.size()) /
             static_cast<float>(buffer_limit_)},
            num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(buffer_limit_), num_elements());
      }
      Status s = buffer_.front().status;
      if (s.ok()) {
        int64_t buffer_element_id = buffer_.front().uid;
        tsl::profiler::TraceMe traceme(
            [&] {
              return tsl::profiler::TraceMeEncode(
                  "PrefetchConsume", {{"element_id", buffer_element_id}});
            },
            profiler::kInfo);
        if (dataset()->slack_period_ > 0 &&
            (num_elements() + 1) % dataset()->slack_period_ == 0) {
          int64_t slack_us = EnvTime::NowMicros() - buffer_.front().created_us;
          slack_us_ = kSleepFactor * slack_us_ + slack_us;
          VLOG(2) << "Setting slack_us_: " << slack_us_;
        }
        *out_tensors = std::move(buffer_.front().value);
        ctx->MergeCheckpoint(&buffer_.front().checkpoint);
        RecordBufferDequeue(ctx, *out_tensors);
        if (legacy_autotune_ && !auto_tuner_->HasElementSize()) {
          auto_tuner_->SetElementSize(GetAllocatedBytes(*out_tensors));
        }
      } else {
        RecordBufferDequeue(ctx, buffer_.front().value);
      }
      if (legacy_autotune_) {
        auto_tuner_->RecordConsumption(buffer_.size());
        buffer_size_->value = auto_tuner_->buffer_limit();
      }
      buffer_.pop_front();
      *end_of_sequence = false;
      cond_var_->notify_all();
      return s;
    }
    Status EnsureThreadsStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            "tf_data_prefetch", [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return absl::OkStatus();
    }
    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      int num_produced = 0;
      while (true) {
        {
          mutex_lock l(*mu_);
          while (!cancelled_ && buffer_.size() >= buffer_limit()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }
          if (cancelled_) {
            prefetch_thread_finished_ = true;
            cond_var_->notify_all();
            return;
          }
        }
        if (dataset()->slack_period_ > 0 &&
            num_produced % dataset()->slack_period_ == 0) {
          VLOG(2) << "Sleeping for: " << slack_us_ * kSleepFactor;
          ctx->env()->SleepForMicroseconds(slack_us_ * kSleepFactor);
        }
        mutex_lock input_l(input_mu_);
        bool end_of_sequence = false;
        BufferElement buffer_element(ctx.get());
        {
          tsl::profiler::TraceMe traceme(
              [&] {
                return tsl::profiler::TraceMeEncode(
                    "PrefetchProduce", {{"element_id", buffer_element.uid}});
              },
              profiler::kInfo);
          buffer_element.status = input_impl_->GetNext(
              ctx.get(), &buffer_element.value, &end_of_sequence);
          buffer_element.checkpoint.Merge(ctx->checkpoint());
        }
        if (buffer_element.status.ok() && end_of_sequence) {
          mutex_lock l(*mu_);
          prefetch_thread_finished_ = true;
          cond_var_->notify_all();
          return;
        }
        {
          mutex_lock l(*mu_);
          RecordBufferEnqueue(ctx.get(), buffer_element.value);
          buffer_element.created_us = EnvTime::NowMicros();
          buffer_.push_back(std::move(buffer_element));
          cond_var_->notify_all();
        }
        ++num_produced;
      }
    }
    Status WriteStatus(IteratorStateWriter* writer, size_t index,
                       const Status& status) TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(absl::StrCat(prefix(), "::", index), CodeKey(),
                              static_cast<int64_t>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            absl::StrCat(prefix(), "::", index), ErrorMessageKey(),
            std::string(status.message())));
      }
      return absl::OkStatus();
    }
    Status ReadStatus(IteratorStateReader* reader, size_t index, Status* status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64_t code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                                            CodeKey(), &code_int));
      absl::StatusCode code = static_cast<absl::StatusCode>(code_int);
      if (code != absl::StatusCode::kOk) {
        tstring error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                               ErrorMessageKey(), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = absl::OkStatus();
      }
      return absl::OkStatus();
    }
    string CodeKey() { return absl::StrCat(kStatus, kCodeSuffix); }
    string ErrorMessageKey() {
      return absl::StrCat(kStatus, kErrorMessageSuffix);
    }
    const std::shared_ptr<mutex> mu_;
    mutex input_mu_ TF_ACQUIRED_BEFORE(*mu_);
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(input_mu_);
    const std::shared_ptr<condition_variable> cond_var_;
    const int64_t buffer_size_min_;
    std::unique_ptr<PrefetchAutotuner> auto_tuner_ TF_GUARDED_BY(*mu_);
    std::deque<BufferElement> buffer_ TF_GUARDED_BY(*mu_);
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;
    bool prefetch_thread_finished_ TF_GUARDED_BY(*mu_) = false;
    const bool legacy_autotune_;
    std::atomic<int64_t> slack_us_;
    const std::shared_ptr<model::SharedState> buffer_size_;
    std::function<void()> deregister_fn_;
    int64 interleave_depth_ = -1;
    std::unique_ptr<Thread> prefetch_thread_ TF_GUARDED_BY(*mu_);
  };
  const DatasetBase* const input_;
  const int64_t buffer_size_;
  const int64_t slack_period_;
  const bool legacy_autotune_ = true;
  const int64_t buffer_size_min_ = 0;
  absl::Status random_indexing_compatible_;
  TraceMeMetadata traceme_metadata_;
};
PrefetchDatasetOp::PrefetchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kSlackPeriod)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kSlackPeriod, &slack_period_));
  }
  if (ctx->HasAttr(kLegacyAutotune)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kLegacyAutotune, &legacy_autotune_));
  }
  if (ctx->HasAttr(kBufferSizeMin)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kBufferSizeMin, &buffer_size_min_));
  }
  if (GetExperiments().contains("autotune_buffer_optimization")) {
    legacy_autotune_ = false;
    buffer_size_min_ = std::max(static_cast<int64_t>(1), buffer_size_min_);
  }
}
void PrefetchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  int64_t buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0 || buffer_size == model::kAutotune,
              errors::InvalidArgument("buffer_size must be >= 0 or set "
                                      "buffer_size to be ",
                                      model::kAutotune, " for auto-tuning"));
  if (buffer_size == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }
  *output = new Dataset(ctx, input, buffer_size, slack_period_,
                        legacy_autotune_, buffer_size_min_);
}
namespace {
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset").Device(DEVICE_CPU).Priority(2),
                        PrefetchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("buffer_size")
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        PrefetchDatasetOp);
}  
}  
}  