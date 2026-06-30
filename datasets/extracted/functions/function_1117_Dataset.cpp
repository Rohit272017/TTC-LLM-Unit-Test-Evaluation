#include "tensorflow/core/kernels/data/flat_map_dataset_op.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/flat_map_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/thread_annotations.h"
namespace tensorflow {
namespace data {
 constexpr const char* const FlatMapDatasetOp::kDatasetType;
 constexpr const char* const FlatMapDatasetOp::kInputDataset;
 constexpr const char* const FlatMapDatasetOp::kOtherArguments;
 constexpr const char* const FlatMapDatasetOp::kFunc;
 constexpr const char* const FlatMapDatasetOp::kTarguments;
 constexpr const char* const FlatMapDatasetOp::kOutputTypes;
 constexpr const char* const FlatMapDatasetOp::kOutputShapes;
constexpr int64_t kMaxRandomIndexingCardinality = 100;
constexpr char kCycleLength[] = "cycle_length";
constexpr char kElementIndex[] = "element_index";
constexpr char kInputsSize[] = "inputs_size";
constexpr char kInputs[] = "inputs";
constexpr char kCurrentElementIteratorUninitialized[] =
    "current_element_iterator_uninitialized";
constexpr char kExhausted[] = "exhausted";
class FlatMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        output_types_(output_types),
        output_shapes_(output_shapes),
        random_access_handler_(ctx, input, *captured_func_) {
    input_->Ref();
    random_indexing_compatible_ = input_->RandomIndexingCompatible();
    if (random_indexing_compatible_.ok() &&
        input_->Cardinality() > kMaxRandomIndexingCardinality) {
      random_indexing_compatible_ = absl::FailedPreconditionError(
          absl::StrCat("The cardinality of the input to ", type_string(),
                       " is too large to support global shuffling. It is ",
                       input_->Cardinality(), ", which is greater than ",
                       kMaxRandomIndexingCardinality));
    }
  }
  ~Dataset() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }
  const DataTypeVector& output_dtypes() const override { return output_types_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }
  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }
  int64_t CardinalityInternal(CardinalityOptions options) const override {
    if (options.compute_level() <
        CardinalityOptions::CARDINALITY_COMPUTE_MODERATE) {
      return kUnknownCardinality;
    }
    absl::StatusOr<int64_t> cardinality = random_access_handler_.Cardinality();
    if (!cardinality.ok()) {
      LOG(ERROR) << "Unable to compute cardinality for dataset "
                 << DebugString() << " due to error: " << cardinality.status();
      return kUnknownCardinality;
    }
    return *cardinality;
  }
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }
  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }
  absl::Status RandomIndexingCompatible() const override {
    return absl::UnimplementedError(
        "Please consider applying maps on each dataset, concatenating them "
        "into "
        "one dataset and apply global shuffle dataset op onto the "
        "dataset to achieve the same result as flat map with global "
        "shuffling.");
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, input_graph_node)},  
        {std::make_pair(1, other_arguments)},         
        {std::make_pair(kFunc, f),
         std::make_pair(kTarguments, other_arguments_types_attr)},  
        output));
    return absl::OkStatus();
  }
 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}
    bool SymbolicCheckpointCompatible() const override { return true; }
    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      input_ckpt_ = std::make_unique<MemoryCheckpoint>(ctx->id_registry());
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (ctx->index_mapper()) {
        return Get(ctx, out_tensors, end_of_sequence);
      }
      mutex_lock l(mu_);
      do {
        if (!input_impl_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        if (current_element_iterator_) {
          bool end_of_element;
          auto nested_ctx = MakeNestedIteratorContext(ctx);
          TF_RETURN_IF_ERROR(current_element_iterator_->GetNext(
              &nested_ctx, out_tensors, &end_of_element));
          ctx->MergeCheckpoint(nested_ctx.checkpoint());
          if (!end_of_element) {
            *end_of_sequence = false;
            return absl::OkStatus();
          }
          ctx->MergeCheckpoint(input_ckpt_.get());
          ctx->PurgeCheckpoint(current_element_iterator_->prefix());
          current_element_iterator_.reset();
        }
        inputs_.clear();
        auto input_ctx = std::make_unique<IteratorContext>(*ctx);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(input_ctx.get(), &inputs_, end_of_sequence));
        input_ckpt_->Merge(input_ctx->checkpoint());
        if (*end_of_sequence) {
          input_impl_.reset();
          return absl::OkStatus();
        }
        TF_RETURN_IF_ERROR(
            BuildCurrentElementIteratorLocked(ctx, true));
      } while (true);
    }
    Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                        bool* end_of_sequence, int* num_skipped) override {
      mutex_lock l(mu_);
      *num_skipped = 0;
      while (*num_skipped < num_to_skip) {
        if (!input_impl_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        if (current_element_iterator_) {
          bool end_of_element;
          auto nested_ctx = MakeNestedIteratorContext(ctx);
          int last_num_skipped;
          TF_RETURN_IF_ERROR(current_element_iterator_->Skip(
              &nested_ctx, num_to_skip - *num_skipped, &end_of_element,
              &last_num_skipped));
          *num_skipped += last_num_skipped;
          ctx->MergeCheckpoint(nested_ctx.checkpoint());
          if (!end_of_element) {
            if (*num_skipped != num_to_skip) {
              return absl::InternalError(absl::StrFormat(
                  "Expected `num_skipped` and `num_to_skip` to be the same. Got"
                  " %d(num_skipped) and %d(num_to_skip)",
                  *num_skipped, num_to_skip));
            }
            continue;
          }
          ctx->MergeCheckpoint(input_ckpt_.get());
          ctx->PurgeCheckpoint(current_element_iterator_->prefix());
          current_element_iterator_.reset();
        }
        inputs_.clear();
        auto input_ctx = std::make_unique<IteratorContext>(*ctx);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(input_ctx.get(), &inputs_, end_of_sequence));
        input_ckpt_->Merge(input_ctx->checkpoint());
        if (*end_of_sequence) {
          input_impl_.reset();
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        TF_RETURN_IF_ERROR(
            BuildCurrentElementIteratorLocked(ctx, false));
      }
      *end_of_sequence = false;
      return absl::OkStatus();
    }
    absl::Status Get(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) TF_LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      TF_ASSIGN_OR_RETURN(size_t parent_index,
                          ctx->index_mapper()(element_count_));
      FlatMapRandomAccessHandler& random_access =
          dataset()->random_access_handler_;
      absl::StatusOr<int64_t> dataset_index =
          random_access.GetDatasetIndex(parent_index);
      if (absl::IsOutOfRange(dataset_index.status())) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
      TF_RETURN_IF_ERROR(dataset_index.status());
      if (dataset_iterators_.empty()) {
        TF_ASSIGN_OR_RETURN(
            dataset_iterators_,
            random_access.MakeInputIterators(ctx, this, prefix()));
        next_positions_.resize(dataset_iterators_.size(), 0);
        input_element_counts_.resize(dataset_iterators_.size(), 0);
      }
      IteratorContext::Params params(ctx);
      params.index_mapper =
          GetFlatMapIndexMapper(ctx->index_mapper(), *dataset_index);
      IteratorContext global_shuffle_ctx(std::move(params));
      TF_RETURN_IF_ERROR(dataset_iterators_[*dataset_index]->GetNext(
          &global_shuffle_ctx, out_tensors, end_of_sequence));
      ctx->MergeCheckpoint(global_shuffle_ctx.checkpoint());
      ++element_count_;
      ++input_element_counts_[*dataset_index];
      return absl::OkStatus();
    }
    IndexMapperFn GetFlatMapIndexMapper(IndexMapperFn parent_index_mapper,
                                        size_t input_dataset_index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      absl::StatusOr<int64_t> cardinality =
          dataset()->random_access_handler_.Cardinality();
      return [this, parent_index_mapper = std::move(parent_index_mapper),
              input_dataset_index, cardinality = std::move(cardinality)](
                 size_t element_position) -> absl::StatusOr<size_t> {
        if (!cardinality.ok() || *cardinality < 0) {
          return absl::FailedPreconditionError(
              "Global shuffling requires finite cardinalities.");
        }
        FlatMapRandomAccessHandler& random_access =
            dataset()->random_access_handler_;
        while (next_positions_[input_dataset_index] < *cardinality) {
          size_t index = next_positions_[input_dataset_index];
          if (parent_index_mapper != nullptr) {
            TF_ASSIGN_OR_RETURN(index, parent_index_mapper(index));
          }
          ++next_positions_[input_dataset_index];
          TF_ASSIGN_OR_RETURN(int64_t shuffled_dataset_index,
                              random_access.GetDatasetIndex(index));
          if (input_dataset_index == shuffled_dataset_index) {
            if (input_dataset_index > 0) {
              TF_ASSIGN_OR_RETURN(
                  int64_t cumulative_cardinality,
                  random_access.CumulativeCardinality(input_dataset_index - 1));
              index -= cumulative_cardinality;
            }
            return index;
          }
        }
        return *cardinality;
      };
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(
          std::move(args),
          {model::MakeNonTunableParameter(kCycleLength, 1)});
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override
        TF_LOCKS_EXCLUDED(mu_) {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), kExhausted, static_cast<int64_t>(!input_impl_)));
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kElementIndex, element_index_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(), kCurrentElementIteratorUninitialized,
            static_cast<int64_t>(!current_element_iterator_)));
        if (current_element_iterator_ && !ctx->symbolic_checkpoint()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(prefix(), kInputsSize, inputs_.size()));
          for (int i = 0; i < inputs_.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                prefix(), strings::StrCat(kInputs, "[", i, "]"), inputs_[i]));
          }
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, current_element_iterator_));
        }
      }
      return absl::OkStatus();
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override
        TF_LOCKS_EXCLUDED(mu_) {
      if (ctx->restored_element_count().has_value()) {
        return RestoreForGlobalShuffle(ctx, reader);
      }
      mutex_lock l(mu_);
      input_impl_.reset();
      element_index_ = 0;
      current_element_iterator_.reset();
      inputs_.clear();
      int64_t input_exhausted;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kExhausted, &input_exhausted));
      if (!static_cast<bool>(input_exhausted)) {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        {
          int64_t temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(prefix(), kElementIndex, &temp));
          element_index_ = temp;
        }
        int64_t current_element_iterator_uninitialized;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(prefix(), kCurrentElementIteratorUninitialized,
                               &current_element_iterator_uninitialized));
        if (!static_cast<bool>(current_element_iterator_uninitialized)) {
          TF_RETURN_IF_ERROR(RestoreCurrentElementIterator(ctx, reader));
        }
      }
      return absl::OkStatus();
    }
    Status RestoreForGlobalShuffle(IteratorContext* ctx,
                                   IteratorStateReader* reader)
        TF_LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      element_count_ = *ctx->restored_element_count();
      FlatMapRandomAccessHandler& random_access =
          dataset()->random_access_handler_;
      TF_ASSIGN_OR_RETURN(int64_t cardinality, random_access.Cardinality());
      if (dataset_iterators_.empty()) {
        TF_ASSIGN_OR_RETURN(
            dataset_iterators_,
            random_access.MakeInputIterators(ctx, this, prefix()));
      }
      input_element_counts_.resize(dataset_iterators_.size(), 0);
      next_positions_.resize(dataset_iterators_.size(), 0);
      std::fill(input_element_counts_.begin(), input_element_counts_.end(), 0);
      std::fill(next_positions_.begin(), next_positions_.end(), 0);
      for (size_t count = 0; count < element_count_ && count < cardinality;
           ++count) {
        TF_ASSIGN_OR_RETURN(size_t parent_index, ctx->index_mapper()(count));
        absl::StatusOr<size_t> dataset_index =
            random_access.GetDatasetIndex(parent_index);
        if (absl::IsOutOfRange(dataset_index.status())) {
          break;
        }
        TF_RETURN_IF_ERROR(dataset_index.status());
        ++input_element_counts_[*dataset_index];
        next_positions_[*dataset_index] = count + 1;
      }
      for (size_t i = 0; i < dataset_iterators_.size(); ++i) {
        IteratorContext::Params params(ctx);
        params.restored_element_count = input_element_counts_[i];
        IteratorContext ctx_copy(std::move(params));
        TF_RETURN_IF_ERROR(
            RestoreInput(&ctx_copy, reader, dataset_iterators_[i]));
        ctx->MergeCheckpoint(ctx_copy.checkpoint());
      }
      return absl::OkStatus();
    }
   private:
    Status BuildCurrentElementIteratorLocked(IteratorContext* ctx,
                                             bool is_get_next)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      std::shared_ptr<model::Node> node = is_get_next ? model_node() : nullptr;
      return MakeIteratorFromInputElement(
          ctx, this, inputs_, element_index_++, *instantiated_captured_func_,
          prefix(), &current_element_iterator_, node);
    }
    Status RestoreCurrentElementIterator(IteratorContext* ctx,
                                         IteratorStateReader* reader)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (ctx->symbolic_checkpoint()) {
        return RestoreCurrentElementIteratorSymbolic(ctx, reader);
      }
      size_t inputs_size;
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kInputsSize, &temp));
        inputs_size = static_cast<size_t>(temp);
      }
      inputs_.reserve(inputs_size);
      for (int i = 0; i < inputs_size; i++) {
        inputs_.emplace_back();
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            ctx->flr(), prefix(), strings::StrCat(kInputs, "[", i, "]"),
            &inputs_.back()));
      }
      element_index_--;
      TF_RETURN_IF_ERROR(
          BuildCurrentElementIteratorLocked(ctx, false));
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_element_iterator_));
      return absl::OkStatus();
    }
    Status RestoreCurrentElementIteratorSymbolic(IteratorContext* ctx,
                                                 IteratorStateReader* reader)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      bool end_of_sequence;
      auto input_ctx = std::make_unique<IteratorContext>(*ctx);
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(input_ctx.get(), &inputs_, &end_of_sequence));
      if (end_of_sequence) {
        return absl::FailedPreconditionError(
            "Unexpected end of sequence while symbolically restoring "
            "FlatMapDataset. Please verify that the input produces data "
            "deterministically.");
      }
      input_ckpt_->Merge(input_ctx->checkpoint());
      element_index_--;
      TF_RETURN_IF_ERROR(
          BuildCurrentElementIteratorLocked(ctx, false));
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_element_iterator_));
      return absl::OkStatus();
    }
    mutex mu_;
    size_t element_index_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<MemoryCheckpoint> input_ckpt_ TF_GUARDED_BY(mu_);
    std::vector<Tensor> inputs_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    size_t element_count_ TF_GUARDED_BY(mu_) = 0;
    std::vector<int64_t> input_element_counts_ TF_GUARDED_BY(mu_);
    std::vector<size_t> next_positions_;
    std::vector<std::unique_ptr<IteratorBase>> dataset_iterators_
        TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> current_element_iterator_ TF_GUARDED_BY(mu_);
  };
  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_ = absl::OkStatus();
  mutable FlatMapRandomAccessHandler random_access_handler_;
};
FlatMapDatasetOp::FlatMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, {},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}
void FlatMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));
  *output = new Dataset(ctx, input, std::move(captured_func), output_types_,
                        output_shapes_);
}
namespace {
REGISTER_KERNEL_BUILDER(Name("FlatMapDataset").Device(DEVICE_CPU),
                        FlatMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("FlatMapDataset");
}  
}  
}  