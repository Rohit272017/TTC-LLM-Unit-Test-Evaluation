#include "tensorflow/core/kernels/data/experimental/sampling_dataset_op.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
namespace tensorflow {
namespace data {
namespace experimental {
 constexpr const char* const SamplingDatasetOp::kDatasetType;
 constexpr const char* const SamplingDatasetOp::kInputDataset;
 constexpr const char* const SamplingDatasetOp::kRate;
 constexpr const char* const SamplingDatasetOp::kSeed;
 constexpr const char* const SamplingDatasetOp::kSeed2;
 constexpr const char* const SamplingDatasetOp::kOutputTypes;
 constexpr const char* const SamplingDatasetOp::kOutputShapes;
class SamplingDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, float rate, int64_t seed, int64_t seed2,
          const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)),
        rate_(rate),
        seeds_(seed, seed2),
        input_(input) {
    input_->Ref();
  }
  ~Dataset() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::unique_ptr<IteratorBase>(
        new Iterator({this, name_utils::IteratorPrefix(kDatasetType, prefix)},
                     seeds_.first, seeds_.second));
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
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }
  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* rate = nullptr;
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(rate_, &rate));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.first, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.second, &seed2));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, rate, seed, seed2}, output));
    return absl::OkStatus();
  }
 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params, int64_t seed, int64_t seed2)
        : DatasetIterator<Dataset>(params),
          seeds_(MaybeOverrideSeeds({seed, seed2})),
          parent_generator_(seeds_.first, seeds_.second),
          generator_(&parent_generator_) {}
    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      bool rand_val_hit;
      do {
        {
          tf_shared_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return absl::OkStatus();
          }
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        }
        if (*end_of_sequence) {
          mutex_lock l(mu_);
          input_impl_.reset();
          return absl::OkStatus();
        }
        float rand_val = Random();
        rand_val_hit = rand_val < dataset()->rate_;
        if (!rand_val_hit) {
          out_tensors->clear();
        }
      } while (!rand_val_hit);
      *end_of_sequence = false;
      return absl::OkStatus();
    }
   protected:
    void ResetRngs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      parent_generator_ = random::PhiloxRandom(seeds_.first, seeds_.second);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          this->full_name("num_random_samples"), num_random_samples_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name("seed"), seeds_.first));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name("seed2"), seeds_.second));
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("input_impl_empty"), ""));
      }
      return absl::OkStatus();
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          this->full_name("num_random_samples"), &num_random_samples_));
      int64_t seed;
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name("seed"), &seed));
      int64_t seed2;
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name("seed2"), &seed2));
      seeds_ = {seed, seed2};
      ResetRngs();
      if (!reader->Contains(full_name("input_impl_empty"))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return absl::OkStatus();
    }
    mutex mu_;
    std::pair<int64_t, int64_t> seeds_ TF_GUARDED_BY(mu_);
   private:
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    float Random() {
      mutex_lock l(mu_);
      num_random_samples_++;
      uint32 random_uint = generator_();
      return random::Uint32ToFloat(random_uint);
    }
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
  };
  const float rate_;
  const std::pair<int64_t, int64_t> seeds_;
  const DatasetBase* const input_;
};  
SamplingDatasetOp::SamplingDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}
void SamplingDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  float rate;
  int64_t seed;
  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, kRate, &rate));
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
  *output = new Dataset(ctx, rate, seed, seed2, input);
}
namespace {
REGISTER_KERNEL_BUILDER(Name("SamplingDataset").Device(DEVICE_CPU),
                        SamplingDatasetOp);
}  
}  
}  
}  