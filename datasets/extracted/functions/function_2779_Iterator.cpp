#include "tensorflow/core/kernels/data/interleave_dataset_op.h"
#include <algorithm>
#include <memory>
#include <optional>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/thread_annotations.h"
namespace tensorflow {
namespace data {
 constexpr const char* const InterleaveDatasetOp::kDatasetType;
 constexpr const char* const InterleaveDatasetOp::kInputDataset;
 constexpr const char* const InterleaveDatasetOp::kOtherArguments;
 constexpr const char* const InterleaveDatasetOp::kCycleLength;
 constexpr const char* const InterleaveDatasetOp::kBlockLength;
 constexpr const char* const InterleaveDatasetOp::kFunc;
 constexpr const char* const InterleaveDatasetOp::kTarguments;
 constexpr const char* const InterleaveDatasetOp::kOutputTypes;
 constexpr const char* const InterleaveDatasetOp::kOutputShapes;
constexpr char kCycleIndex[] = "cycle_index";
constexpr char kBlockIndex[] = "block_index";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumOpen[] = "num_open";
constexpr char kArgsSize[] = "args_size";
constexpr char kArgsList[] = "args_list_";
constexpr char kCurrentElementsUninitialized[] =
    "current_elements_uninitialized";
constexpr char kNextInputElementIndex[] = "next_input_element_index";
constexpr char kLastCheckpointedInputElementIndex[] =
    "last_checkpointed_input_element_index";
constexpr char kInputElementIndices[] = "input_element_indices";
class InterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func, int64_t cycle_length,
          int64_t block_length, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        output_types_(output_types),
        output_shapes_(output_shapes),
        traceme_metadata_(
            {{"block_length",
              strings::Printf("%lld", static_cast<long long>(block_length))},
             {"cycle_length",
              strings::Printf("%lld", static_cast<long long>(cycle_length))}}) {
    input_->Ref();
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
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }
  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* cycle_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
    Node* block_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {{0, input_node}, {2, cycle_length_node}, {3, block_length_node}},
        {{1, other_arguments}},
        {{kFunc, f}, {kTarguments, other_arguments_types_attr}}, output));
    return absl::OkStatus();
  }
 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          current_elements_(params.dataset->cycle_length_) {}
    bool SymbolicCheckpointCompatible() const override { return true; }
    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      input_ckpt_ = std::make_unique<MemoryCheckpoint>(ctx->id_registry());
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }
    void AdvanceToNextInCycle() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      block_index_ = 0;
      cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
    }
    Status AdvancePosition(int num_elements) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      block_index_ += num_elements;
      if (block_index_ == dataset()->block_length_) {
        AdvanceToNextInCycle();
        return absl::OkStatus();
      } else if (block_index_ < dataset()->block_length_) {
        return absl::OkStatus();
      }
      return absl::InternalError(
          "Something went wrong as `block_index_` should never be larger than "
          "`dataset()->block_length_`");
    }
    void AdvancePosition() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      ++block_index_;
      if (block_index_ == dataset()->block_length_) {
        AdvanceToNextInCycle();
      }
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      while (!end_of_input_ || num_open_ > 0) {
        if (current_elements_[cycle_index_]) {
          bool end_of_element;
          auto nested_ctx = MakeNestedIteratorContext(ctx);
          CurrentElement& current_element = *current_elements_[cycle_index_];
          TF_RETURN_IF_ERROR(current_element.iterator->GetNext(
              &nested_ctx, out_tensors, &end_of_element));
          ctx->MergeCheckpoint(nested_ctx.checkpoint());
          if (!end_of_element) {
            AdvancePosition();
            *end_of_sequence = false;
            return absl::OkStatus();
          } else {
            ctx->PurgeCheckpoint(current_element.iterator->prefix());
            UpdateSymbolicCheckpointAfterCurrentElementFinished(
                *ctx, *current_elements_[cycle_index_]);
            current_elements_[cycle_index_].reset();
            --num_open_;
            AdvanceToNextInCycle();
          }
        } else {
          TF_RETURN_IF_ERROR(MoveToNextElement(ctx));
        }
      }
      ctx->MergeCheckpoint(input_ckpt_.get());
      *end_of_sequence = true;
      return absl::OkStatus();
    }
    Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                        bool* end_of_sequence, int* num_skipped) override {
      mutex_lock l(mu_);
      *num_skipped = 0;
      while (!end_of_input_ || num_open_ > 0) {
        if (current_elements_[cycle_index_]) {
          CurrentElement& current_element = *current_elements_[cycle_index_];
          int element_num_to_skip = num_to_skip - *num_skipped;
          if (element_num_to_skip > dataset()->block_length_ - block_index_) {
            element_num_to_skip = dataset()->block_length_ - block_index_;
          }
          bool end_of_element = false;
          int element_num_skipped = 0;
          auto nested_ctx = MakeNestedIteratorContext(ctx);
          TF_RETURN_IF_ERROR(current_element.iterator->Skip(
              &nested_ctx, element_num_to_skip, &end_of_element,
              &element_num_skipped));
          *num_skipped += element_num_skipped;
          ctx->MergeCheckpoint(nested_ctx.checkpoint());
          if (!end_of_element) {
            TF_RETURN_IF_ERROR(AdvancePosition(element_num_skipped));
          } else {
            ctx->PurgeCheckpoint(current_element.iterator->prefix());
            UpdateSymbolicCheckpointAfterCurrentElementFinished(
                *ctx, *current_elements_[cycle_index_]);
            current_elements_[cycle_index_].reset();
            --num_open_;
            AdvanceToNextInCycle();
          }
          if (num_to_skip == *num_skipped) {
            *end_of_sequence = false;
            return absl::OkStatus();
          }
        } else {
          TF_RETURN_IF_ERROR(MoveToNextElement(ctx));
        }
      }
      ctx->MergeCheckpoint(input_ckpt_.get());
      *end_of_sequence = true;
      return absl::OkStatus();
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(
          std::move(args), {model::MakeNonTunableParameter(
                               kCycleLength, dataset()->cycle_length_)});
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kCycleIndex, cycle_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kBlockIndex, block_index_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), kEndOfInput, static_cast<int64_t>(end_of_input_)));
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kNumOpen, num_open_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kNextInputElementIndex,
                                             next_input_element_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kLastCheckpointedInputElementIndex,
                              last_checkpointed_input_element_index_));
      TF_RETURN_IF_ERROR(SaveCurrentElements(ctx, writer));
      return absl::OkStatus();
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      int64_t cycle_index;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kCycleIndex, &cycle_index));
      cycle_index_ = size_t(cycle_index);
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kBlockIndex, &block_index_));
      int64_t end_of_input;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kEndOfInput, &end_of_input));
      end_of_input_ = static_cast<bool>(end_of_input);
      int64_t num_open;
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kNumOpen, &num_open));
      num_open_ = size_t(num_open);
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kNextInputElementIndex,
                                            &next_input_element_index_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kLastCheckpointedInputElementIndex,
                             &last_checkpointed_input_element_index_));
      int64_t cycle_length = dataset()->cycle_length_;
      std::vector<InputOffset> input_element_indices(cycle_length, -1);
      std::vector<std::optional<MemoryCheckpoint>> checkpoints(cycle_length);
      std::vector<std::vector<Tensor>> args(cycle_length);
      if (ctx->symbolic_checkpoint()) {
        auto status_or = RestoreInputOffsets(*reader);
        if (!status_or.ok()) {
          return status_or.status();
        }
        auto& input_offset_w_cycle_idxs = status_or.value();
        TF_RETURN_IF_ERROR(RestoreArgsListAndInputOffsetCycleIdxMap(
            *ctx, input_element_indices, checkpoints, args,
            input_offset_w_cycle_idxs));
      }
      TF_RETURN_IF_ERROR(
          RestoreCurrentElements(ctx, reader, input_element_indices,
                                 std::move(checkpoints), std::move(args)));
      return absl::OkStatus();
    }
    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }
   private:
    using InputOffset = int64_t;
    using CycleIdx = int;
    struct CurrentElement;
    struct InputOffsetWithCycleIdx;
    int64_t GetSubIteratorIndexForPrefix(bool symbolic_checkpoint)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return GetSubIteratorIndexForPrefix(symbolic_checkpoint, cycle_index_,
                                          next_input_element_index_);
    }
    int64_t GetSubIteratorIndexForPrefix(
        bool symbolic_checkpoint, int64_t cycle_index,
        std::optional<int64_t> input_element_index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return (symbolic_checkpoint) ? (input_element_index.value())
                                   : (cycle_index);
    }
    Status SaveCurrentElements(SerializationContext* ctx,
                               IteratorStateWriter* writer)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            prefix(),
            strings::StrCat(kCurrentElementsUninitialized, "[", idx, "]"),
            !current_elements_[idx]));
        if (!current_elements_[idx]) {
          continue;
        }
        if (!ctx->symbolic_checkpoint()) {
          TF_RETURN_IF_ERROR(
              SaveInput(ctx, writer, current_elements_[idx]->iterator));
          const auto& args = current_elements_[idx]->args;
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              prefix(), strings::StrCat(kArgsSize, "[", idx, "]"),
              args.size()));
          for (int i = 0; i < args.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                prefix(), strings::StrCat(kArgsList, "[", idx, "][", i, "]"),
                args[i]));
          }
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              prefix(), strings::StrCat(kInputElementIndices, "[", idx, "]"),
              current_elements_[idx]->input_element_index));
        }
      }
      return absl::OkStatus();
    }
    absl::StatusOr<std::vector<InputOffsetWithCycleIdx>> RestoreInputOffsets(
        IteratorStateReader& reader) {
      std::vector<InputOffsetWithCycleIdx> input_offsets;
      int64_t cycle_length = dataset()->cycle_length_;
      for (int cycle_idx = 0; cycle_idx < cycle_length; cycle_idx++) {
        int64_t current_element_uninitialized;
        TF_RETURN_IF_ERROR(reader.ReadScalar(
            prefix(),
            strings::StrCat(kCurrentElementsUninitialized, "[", cycle_idx, "]"),
            &current_element_uninitialized));
        if (!current_element_uninitialized) {
          int64_t input_element_index;
          TF_RETURN_IF_ERROR(reader.ReadScalar(
              prefix(),
              strings::StrCat(kInputElementIndices, "[", cycle_idx, "]"),
              &input_element_index));
          input_offsets.push_back(
              InputOffsetWithCycleIdx{input_element_index, cycle_idx});
        }
      }
      return std::move(input_offsets);
    }
    Status RestoreArgsListAndInputOffsetCycleIdxMap(
        IteratorContext& ctx, std::vector<InputOffset>& input_element_indices,
        std::vector<std::optional<MemoryCheckpoint>>& checkpoints,
        std::vector<std::vector<Tensor>>& args,
        std::vector<InputOffsetWithCycleIdx>& input_offset_w_cycle_idxs)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (input_element_indices.size() != dataset()->cycle_length_ ||
          checkpoints.size() != dataset()->cycle_length_ ||
          args.size() != dataset()->cycle_length_) {
        return absl::FailedPreconditionError(
            "input_element_indices, checkpoints and args should be of same "
            "length");
      }
      std::sort(input_offset_w_cycle_idxs.begin(),
                input_offset_w_cycle_idxs.end(),
                [](const InputOffsetWithCycleIdx& lhs,
                   const InputOffsetWithCycleIdx& rhs) {
                  return lhs.input_element_index < rhs.input_element_index;
                });
      bool end_of_sequence = false;
      int num_to_skip;
      int num_actually_skip;
      InputOffset prev_input_element_index =
          last_checkpointed_input_element_index_;
      auto input_ctx = std::make_unique<IteratorContext>(ctx);
      for (const auto& input_offset_w_cycle_idx : input_offset_w_cycle_idxs) {
        InputOffset input_element_index =
            input_offset_w_cycle_idx.input_element_index;
        CycleIdx cycle_idx = input_offset_w_cycle_idx.cycle_idx;
        if (input_element_index >= next_input_element_index_) {
          return absl::FailedPreconditionError(
              "input_element_index < next_input_element_index_ must be "
              "met.");
        }
        num_to_skip = input_element_index - prev_input_element_index - 1;
        TF_RETURN_IF_ERROR(input_impl_->Skip(input_ctx.get(), num_to_skip,
                                             &end_of_sequence,
                                             &num_actually_skip));
        if (end_of_sequence || num_actually_skip != num_to_skip) {
          return absl::InternalError(
              "Unexpected end of sequence while symbolically restoring "
              "InterleaveDataset. Please verify that the input produces data "
              "deterministically.");
        }
        std::vector<Tensor> current_element_args;
        TF_RETURN_IF_ERROR(input_impl_->GetNext(
            input_ctx.get(), &current_element_args, &end_of_sequence));
        prev_input_element_index = input_element_index;
        checkpoints[cycle_idx].emplace(*input_ctx->checkpoint());
        args[cycle_idx] = std::move(current_element_args);
        input_element_indices[cycle_idx] = input_element_index;
      }
      num_to_skip = next_input_element_index_ - prev_input_element_index - 1;
      TF_RETURN_IF_ERROR(input_impl_->Skip(
          input_ctx.get(), num_to_skip, &end_of_sequence, &num_actually_skip));
      if (end_of_sequence || num_actually_skip != num_to_skip) {
        return absl::InternalError(
            "Unexpected end of sequence while symbolically restoring "
            "InterleaveDataset. Please verify that the input produces data "
            "deterministically.");
      }
      input_ckpt_->Merge(input_ctx->checkpoint());
      return absl::OkStatus();
    }
    Status RestoreCurrentElements(
        IteratorContext* ctx, IteratorStateReader* reader,
        std::vector<InputOffset>& input_element_indices,
        std::vector<std::optional<MemoryCheckpoint>>&& checkpoints,
        std::vector<std::vector<Tensor>>&& args)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        int64_t current_element_uninitialized;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            prefix(),
            strings::StrCat(kCurrentElementsUninitialized, "[", idx, "]"),
            &current_element_uninitialized));
        if (!current_element_uninitialized) {
          if (!ctx->symbolic_checkpoint()) {
            int64_t args_size;
            std::vector<Tensor> current_element_args;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                prefix(), strings::StrCat(kArgsSize, "[", idx, "]"),
                &args_size));
            current_element_args.resize(args_size);
            for (int i = 0; i < args_size; i++) {
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  ctx->flr(), prefix(),
                  strings::StrCat(kArgsList, "[", idx, "][", i, "]"),
                  &current_element_args[i]));
            }
            args[idx] = std::move(current_element_args);
          }
          std::unique_ptr<IteratorBase> iterator;
          TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
              ctx, this, args[idx],
              GetSubIteratorIndexForPrefix(ctx->symbolic_checkpoint(), idx,
                                           input_element_indices[idx]),
              *instantiated_captured_func_, prefix(), &iterator,
              nullptr));
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, iterator));
          current_elements_[idx].emplace(
              std::move(checkpoints[idx]), std::move(args[idx]),
              input_element_indices[idx], std::move(iterator));
        } else {
          current_elements_[idx].reset();
        }
      }
      return absl::OkStatus();
    }
    Status MoveToNextElement(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!end_of_input_) {
        IteratorContext input_ctx = MakeNestedIteratorContext(ctx);
        std::vector<Tensor> args;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(&input_ctx, &args, &end_of_input_));
        input_ckpt_->Merge(input_ctx.checkpoint());
        if (!end_of_input_) {
          std::unique_ptr<IteratorBase> iterator;
          TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
              ctx, this, args,
              GetSubIteratorIndexForPrefix(ctx->symbolic_checkpoint()),
              *instantiated_captured_func_, prefix(), &iterator, model_node()));
          ++num_open_;
          std::optional<MemoryCheckpoint> checkpoint;
          if (ctx->symbolic_checkpoint()) {
            checkpoint.emplace(*input_ckpt_);
          }
          current_elements_[cycle_index_].emplace(
              std::move(checkpoint), std::move(args), next_input_element_index_,
              std::move(iterator));
          next_input_element_index_++;
        }
      } else {
        AdvanceToNextInCycle();
      }
      return absl::OkStatus();
    }
    InputOffset IsEarliestInputElementIndex(InputOffset input_element_index) {
      InputOffset min_input_element_index = input_element_index;
      for (int i = 0; i < current_elements_.size(); ++i) {
        if (!current_elements_[i]) continue;
        if (current_elements_[i]->input_element_index <
            min_input_element_index) {
          min_input_element_index = current_elements_[i]->input_element_index;
        }
      }
      return (min_input_element_index == input_element_index);
    }
    void UpdateSymbolicCheckpointAfterCurrentElementFinished(
        IteratorContext& ctx, CurrentElement& current_element)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!ctx.symbolic_checkpoint()) {
        return;
      }
      InputOffset input_element_index = current_element.input_element_index;
      if (IsEarliestInputElementIndex(input_element_index)) {
        MemoryCheckpoint& checkpoint =
            const_cast<MemoryCheckpoint&>(current_element.checkpoint.value());
        ctx.MergeCheckpoint(&checkpoint);
        last_checkpointed_input_element_index_ = input_element_index;
      }
    }
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    struct CurrentElement {
      const std::optional<MemoryCheckpoint> checkpoint = std::nullopt;
      const InputOffset input_element_index = -1;
      const std::vector<Tensor> args;  
      std::unique_ptr<IteratorBase> iterator = nullptr;
      explicit CurrentElement(std::optional<MemoryCheckpoint>&& checkpoint,
                              std::vector<Tensor>&& args,
                              InputOffset input_element_index,
                              std::unique_ptr<IteratorBase> iterator)
          : checkpoint(std::move(checkpoint)),
            input_element_index(input_element_index),
            args(std::move(args)),
            iterator(std::move(iterator)) {}
      CurrentElement(CurrentElement&& other) = default;
    };
    struct InputOffsetWithCycleIdx {
      InputOffset input_element_index;
      CycleIdx cycle_idx;
    };
    std::vector<std::optional<CurrentElement>> current_elements_;
    InputOffset last_checkpointed_input_element_index_ TF_GUARDED_BY(mu_) = -1;
    InputOffset next_input_element_index_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<MemoryCheckpoint> input_ckpt_ TF_GUARDED_BY(mu_);
    size_t cycle_index_ TF_GUARDED_BY(mu_) = 0;
    int64_t block_index_ TF_GUARDED_BY(mu_) = 0;
    bool end_of_input_ TF_GUARDED_BY(mu_) = false;
    size_t num_open_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };
  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int64_t cycle_length_;
  const int64_t block_length_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const TraceMeMetadata traceme_metadata_;
};
InterleaveDatasetOp::InterleaveDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, {},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}
void InterleaveDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  int64_t cycle_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCycleLength, &cycle_length));
  if (cycle_length == model::kAutotune) {
    cycle_length = port::MaxParallelism();
  }
  OP_REQUIRES(
      ctx, cycle_length > 0,
      errors::InvalidArgument("cycle_length must be greater than zero."));
  int64_t block_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBlockLength, &block_length));
  OP_REQUIRES(
      ctx, block_length > 0,
      errors::InvalidArgument("block_length must be greater than zero."));
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));
  *output = new Dataset(ctx, input, std::move(captured_func), cycle_length,
                        block_length, output_types_, output_shapes_);
}
namespace {
REGISTER_KERNEL_BUILDER(Name("InterleaveDataset").Device(DEVICE_CPU),
                        InterleaveDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("InterleaveDataset");
}  
}  
}  