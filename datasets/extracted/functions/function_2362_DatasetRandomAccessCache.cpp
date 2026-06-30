#include "tensorflow/core/kernels/data/cache_dataset_ops.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "tensorflow/core/data/global_shuffle_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/cache_ops.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
namespace tensorflow {
namespace data {
 constexpr const char* const CacheDatasetOp::kDatasetType;
 constexpr const char* const CacheDatasetOp::kInputDataset;
 constexpr const char* const CacheDatasetOp::kFileName;
 constexpr const char* const CacheDatasetOp::kOutputTypes;
 constexpr const char* const CacheDatasetOp::kOutputShapes;
namespace {
constexpr char kKeyStrFormat[] = "%%%zuzu_%%%zuzu";
constexpr char kPaddingSizeStrFormat[] = "%zu";
constexpr char kFileDatasetPrefix[] = "File";
constexpr char kMode[] = "Mode";
constexpr char kLockFileSuffix[] = ".lockfile";
constexpr char kIterationCompleted[] = "iteration_completed";
constexpr char kCurIndex[] = "cur_index";
constexpr char kShardId[] = "shard_id";
constexpr char kCreatedAt[] = "Created at";
constexpr char kMemoryDatasetPrefix[] = "Memory";
constexpr char kMemoryCache[] = "MemoryCache";
constexpr char kCacheCompleted[] = "cache_completed";
constexpr char kIndex[] = "index";
constexpr char kImpl[] = "Impl";
constexpr char kCacheDataset[] = "CacheDataset";
constexpr char kIncompleteCacheErrorMessage[] =
    "The calling iterator did not fully read the dataset being cached. In "
    "order to avoid unexpected truncation of the dataset, the partially cached "
    "contents of the dataset  will be discarded. This can happen if you have "
    "an input pipeline similar to `dataset.cache().take(k).repeat()`. You "
    "should use `dataset.take(k).cache().repeat()` instead.";
}  
class DatasetRandomAccessCache {
 public:
  explicit DatasetRandomAccessCache(const DatasetBase* dataset)
      : input_(dataset) {}
  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) {
    if (!iter_resource_) {
      TF_ASSIGN_OR_RETURN(iter_resource_,
                          GetIteratorResourceFromDataset(ctx, input_));
      TF_RETURN_IF_ERROR(iter_resource_->SetIteratorFromDataset(ctx, input_));
    }
    if (index >= cache_.size()) {
      TF_RETURN_IF_ERROR(ExtendTempCacheToIndex(index, ctx));
    }
    *out_tensors = cache_.at(index);
    return absl::OkStatus();
  }
  std::vector<std::vector<Tensor>> GetCacheData() { return cache_; }
 private:
  Status ExtendTempCacheToIndex(int64 index, OpKernelContext* ctx) {
    bool end_of_sequence;
    while (cache_.size() <= index) {
      std::vector<Tensor> out_tensors;
      TF_RETURN_IF_ERROR(
          iter_resource_->GetNext(ctx, &out_tensors, &end_of_sequence));
      if (end_of_sequence) {
        return tensorflow::errors::OutOfRange("Index out of range [0, ",
                                              cache_.size(), "):", index);
      }
      cache_.push_back(out_tensors);
    }
    return absl::OkStatus();
  }
  absl::StatusOr<core::RefCountPtr<IteratorResource>>
  GetIteratorResourceFromDataset(OpKernelContext* ctx,
                                 const DatasetBase* dataset) {
    FunctionLibraryRuntime* flr;
    std::unique_ptr<DeviceMgr> device_mgr(nullptr);
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> plfr(nullptr);
    TF_RETURN_IF_ERROR(
        ctx->function_library()->Clone(&flib_def, &plfr, &flr, true));
    core::RefCountPtr<IteratorResource> iter_resource(new IteratorResource(
        ctx->env(), dataset->output_dtypes(), dataset->output_shapes(),
        std::move(device_mgr), std::move(flib_def), std::move(plfr), flr));
    return iter_resource;
  }
  const DatasetBase* input_;  
  core::RefCountPtr<IteratorResource> iter_resource_;
  std::vector<std::vector<Tensor>> cache_;
};
class IteratorRandomAccessCache {
 public:
  explicit IteratorRandomAccessCache(const DatasetBase* input)
      : input_(input) {}
  absl::Status Get(AnyContext ctx, size_t element_position,
                   std::vector<Tensor>* out_tensors) {
    if (element_position < cache_.size() && !cache_[element_position].empty()) {
      *out_tensors = cache_[element_position];
      return absl::OkStatus();
    }
    TF_RETURN_IF_ERROR(input_->Get(ctx, element_position, out_tensors));
    if (element_position >= cache_.size()) {
      cache_.resize(element_position + 1);
    }
    cache_[element_position] = *out_tensors;
    return absl::OkStatus();
  }
 private:
  const DatasetBase* input_ = nullptr;
  std::vector<std::vector<Tensor>> cache_;
};
class CacheDatasetOp::FileDatasetBase : public DatasetBase {
 public:
  FileDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                  string filename, Env* env)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        filename_(std::move(filename)),
        env_(env),
        num_tensors_(input->output_dtypes().size()),
        tensor_index_padding_size_(StringPaddingSize(num_tensors_)),
        item_index_padding_size_(StringPaddingSize(kMaxItems)),
        tensor_format_string_(strings::Printf(kKeyStrFormat,
                                              item_index_padding_size_,
                                              tensor_index_padding_size_)) {
    input_->Ref();
    DCHECK_EQ(item_index_padding_size_, 7);
  }
  ~FileDatasetBase() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return std::make_unique<FileIterator>(FileIterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }
  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }
  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kFileDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
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
 protected:
  const DatasetBase* const input_;
  const tstring filename_;
 private:
  static size_t StringPaddingSize(size_t num_tensors) {
    return strings::Printf(kPaddingSizeStrFormat, num_tensors - 1).size();
  }
  string FormatName(size_t item_index, size_t tensor_index) const {
    return strings::Printf(tensor_format_string_.c_str(), item_index,
                           tensor_index);
  }
  class FileIterator : public DatasetIterator<FileDatasetBase> {
   public:
    explicit FileIterator(const Params& params)
        : DatasetIterator<FileDatasetBase>(params) {
      if (params.dataset->env_
              ->FileExists(MetaFilename(params.dataset->filename_))
              .ok()) {
        mode_ = Mode::read;
      } else {
        mode_ = Mode::write;
      }
    }
    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      return InitializeIterator(ctx);
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       1);
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kMode, mode_));
      return SaveInput(ctx, writer, iterator_);
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kMode, &temp));
        mode_ = static_cast<Mode>(temp);
      }
      if (mode_ == Mode::write &&
          dataset()
              ->env_->FileExists(MetaFilename(dataset()->filename_))
              .ok()) {
        LOG(WARNING)
            << "It looks like the cache was already completely written("
            << MetaFilename(dataset()->filename_)
            << ") after the last checkpoint was saved. Attempting to read "
            << "the cache instead of continuing to write. If this is a "
            << "mistake, please remove the above file and try running again.";
        mode_ = Mode::read;
      }
      TF_RETURN_IF_ERROR(InitializeIterator(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }
   private:
    class FileWriterIterator : public DatasetIterator<FileDatasetBase> {
     public:
      explicit FileWriterIterator(const Params& params)
          : DatasetIterator<FileDatasetBase>(params),
            cur_index_(0),
            shard_id_(0),
            filename_(
                strings::StrCat(params.dataset->filename_, "_", shard_id_)),
            lockfile_(strings::StrCat(filename_, kLockFileSuffix)),
            lockfile_created_(false),
            iteration_completed_(false) {}
      ~FileWriterIterator() override {
        if (!dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          LOG(WARNING) << kIncompleteCacheErrorMessage;
          std::vector<string> cache_files;
          Status s = dataset()->env_->GetMatchingPaths(
              strings::StrCat(filename_, "*"), &cache_files);
          if (!s.ok()) {
            LOG(WARNING) << "Failed to get matching files on " << filename_
                         << "* : " << s.ToString();
          }
          for (const string& path : cache_files) {
            s = dataset()->env_->DeleteFile(path);
            if (!s.ok()) {
              LOG(WARNING) << "Failed to delete " << path << " : "
                           << s.ToString();
            }
          }
        }
      }
      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(EnsureLockFileExists(end_of_sequence));
        if (*end_of_sequence) {
          return absl::OkStatus();
        }
        TF_RETURN_IF_ERROR(writer_->status());
        if (cur_index_ >= kMaxItems) {
          Status s = Finish();
          if (!s.ok()) {
            LOG(ERROR) << s;
          }
          return errors::InvalidArgument(
              "Upstream iterator is producing more than ", kMaxItems,
              " items, which is more than the cache limit.");
        }
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence && out_tensors->empty()) {
          TF_RETURN_IF_ERROR(Finish());
          cur_index_++;
          return absl::OkStatus();
        }
        if (out_tensors->size() != dataset()->num_tensors_) {
          return errors::Internal(
              "Upstream iterator returned invalid number of tensors. "
              "Expected ",
              dataset()->num_tensors_, " got: ", out_tensors->size());
        }
        size_t tensor_index = 0;
        for (const Tensor& t : *out_tensors) {
          DCHECK_LT(tensor_index, dataset()->num_tensors_);
          string key = dataset()->FormatName(cur_index_, tensor_index++);
          TF_RETURN_IF_ERROR(writer_->Add(key, t));
        }
        if (*end_of_sequence) {
          TF_RETURN_IF_ERROR(Finish());
        }
        cur_index_++;
        return absl::OkStatus();
      }
     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         1);
      }
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kCurIndex, cur_index_));
        if (iteration_completed_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(prefix(), kIterationCompleted, ""));
          return absl::OkStatus();
        }
        if (lockfile_created_) {
          TF_RETURN_IF_ERROR(writer_->Finish());
          shard_id_++;
          filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
          lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
          lockfile_created_ = false;
        }
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kShardId, shard_id_));
        return absl::OkStatus();
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64_t temp;
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kCurIndex, &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }
        if (reader->Contains(prefix(), kIterationCompleted)) {
          iteration_completed_ = true;
          return absl::OkStatus();
        }
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        {
          TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kShardId, &temp));
          shard_id_ = static_cast<size_t>(temp);
          if (shard_id_ != temp) {
            return errors::Internal("Invalid value for shard_id ", temp);
          }
        }
        filename_ = strings::StrCat(dataset()->filename_, "_", shard_id_);
        lockfile_ = strings::StrCat(filename_, kLockFileSuffix);
        writer_ = std::make_unique<BundleWriter>(dataset()->env_, filename_);
        return absl::OkStatus();
      }
     private:
      Status EnsureLockFileExists(bool* end_of_sequence)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (iteration_completed_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        if (lockfile_created_) {
          return absl::OkStatus();
        }
        if (dataset()->env_->FileExists(MetaFilename(filename_)).ok()) {
          return errors::AlreadyExists("Existing cache files found: \n",
                                       MetaFilename(filename_), "\n",
                                       DataFilename(filename_, 0, 1), "\n",
                                       "To continue delete the above files.");
        }
        if (dataset()->env_->FileExists(lockfile_).ok()) {
          char contents_scratch[151] = {0};  
          StringPiece contents;
          std::unique_ptr<RandomAccessFile> file;
          if (dataset()->env_->NewRandomAccessFile(lockfile_, &file).ok()) {
            file->Read(0, 150, &contents, contents_scratch).IgnoreError();
          }
          return errors::AlreadyExists(
              "There appears to be a concurrent caching iterator running - "
              "cache lockfile already exists ('",
              lockfile_,
              "'). If you are sure no other running TF computations are "
              "using this cache prefix, delete the lockfile and "
              "re-initialize the iterator. Lockfile contents: ",
              contents);
        }
        std::unique_ptr<WritableFile> lockfile;
        TF_RETURN_IF_ERROR(
            dataset()->env_->NewWritableFile(lockfile_, &lockfile));
        TF_RETURN_IF_ERROR(lockfile->Append(
            strings::StrCat(kCreatedAt, ": ", EnvTime::NowSeconds())));
        writer_ = std::make_unique<BundleWriter>(dataset()->env_, filename_);
        lockfile_created_ = true;
        return absl::OkStatus();
      }
      Status Finish() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        iteration_completed_ = true;
        TF_RETURN_IF_ERROR(writer_->Finish());
        {
          std::vector<tstring> prefixes;
          prefixes.reserve(shard_id_ + 1);
          for (size_t i = 0; i <= shard_id_; ++i) {
            prefixes.emplace_back(
                strings::StrCat(dataset()->filename_, "_", i));
          }
          TF_RETURN_IF_ERROR(
              MergeBundles(dataset()->env_, prefixes, dataset()->filename_));
        }
        for (size_t i = 0; i <= shard_id_; ++i) {
          TF_RETURN_IF_ERROR(dataset()->env_->DeleteFile(
              strings::StrCat(dataset()->filename_, "_", i, kLockFileSuffix)));
        }
        return absl::OkStatus();
      }
      mutex mu_;
      size_t cur_index_ TF_GUARDED_BY(mu_);
      size_t shard_id_ TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      string filename_;
      std::unique_ptr<BundleWriter> writer_ TF_GUARDED_BY(mu_);
      string lockfile_ TF_GUARDED_BY(mu_);
      bool lockfile_created_ TF_GUARDED_BY(mu_);
      bool iteration_completed_ TF_GUARDED_BY(mu_);
    };  
    class FileReaderIterator : public DatasetIterator<FileDatasetBase> {
     public:
      explicit FileReaderIterator(const Params& params)
          : DatasetIterator<FileDatasetBase>(params),
            cur_index_(0),
            reader_(dataset()->env_, dataset()->filename_),
            iterator_restored_(false) {}
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(reader_.status());
        if (!reader_.Valid()) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        out_tensors->clear();
        out_tensors->resize(dataset()->num_tensors_);
        for (size_t i = 0; i < dataset()->num_tensors_; ++i) {
          if (!iterator_restored_) {
            reader_.Next();  
          } else {
            iterator_restored_ = false;
          }
          if (!reader_.Valid()) {
            out_tensors->clear();
            *end_of_sequence = true;
            return absl::OkStatus();
          }
          StringPiece key = reader_.key();
          DCHECK_EQ(key, dataset()->FormatName(cur_index_, i));
          TF_RETURN_IF_ERROR(reader_.ReadCurrent(&(*out_tensors)[i]));
          TF_RETURN_IF_ERROR(reader_.status());
        }
        cur_index_++;
        return absl::OkStatus();
      }
     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         1);
      }
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kCurIndex, cur_index_));
        return absl::OkStatus();
      }
      Status RestoreInternal(
          IteratorContext* ctx,
          IteratorStateReader* iterator_state_reader) override {
        mutex_lock l(mu_);
        {
          int64_t temp;
          TF_RETURN_IF_ERROR(
              iterator_state_reader->ReadScalar(prefix(), kCurIndex, &temp));
          cur_index_ = static_cast<size_t>(temp);
          if (cur_index_ != temp) {
            return errors::Internal("Invalid value for cur_index ", temp);
          }
        }
        if (!reader_.Valid()) {
          return errors::Internal("Error initializing BundleReader.");
        }
        reader_.Seek(dataset()->FormatName(cur_index_, 0));
        iterator_restored_ = true;
        return absl::OkStatus();
      }
     private:
      mutex mu_;
      size_t cur_index_ TF_GUARDED_BY(mu_);
      BundleReader reader_ TF_GUARDED_BY(mu_);
      bool iterator_restored_ TF_GUARDED_BY(mu_);
    };  
    Status InitializeIterator(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      switch (mode_) {
        case Mode::read:
          iterator_ =
              std::make_unique<FileReaderIterator>(FileReaderIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
          break;
        case Mode::write:
          iterator_ =
              std::make_unique<FileWriterIterator>(FileWriterIterator::Params{
                  dataset(), strings::StrCat(prefix(), kImpl)});
      }
      TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
      return iterator_->Initialize(ctx);
    }
    mutex mu_;
    enum Mode { read, write };
    Mode mode_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);
  };  
  Env* const env_;
  const size_t num_tensors_;
  const size_t tensor_index_padding_size_;
  static constexpr size_t kMaxItems = 10000000;  
  const size_t item_index_padding_size_;
  const string tensor_format_string_;
};  
class CacheDatasetOp::FileDataset : public CacheDatasetOp::FileDatasetBase {
 public:
  using FileDatasetBase::FileDatasetBase;
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
    Node* filename = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph, filename}, output));
    return absl::OkStatus();
  }
};
class CacheDatasetOp::FileDatasetV2 : public CacheDatasetOp::FileDatasetBase {
 public:
  explicit FileDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                         string filename, Env* env,
                         const Tensor& resource_handle)
      : FileDatasetBase(ctx, input, filename, env),
        resource_handle_(resource_handle) {}
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename_node));
    Node* resource_handle_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return absl::OkStatus();
  }
 private:
  const Tensor resource_handle_;
};
class CacheDatasetOp::MemoryDatasetBase : public DatasetBase {
 public:
  explicit MemoryDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                             std::shared_ptr<MemoryCache> cache)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        cache_(std::move(cache)) {
    input_->Ref();
    random_indexing_compatible_ = input_->RandomIndexingCompatible();
  }
  ~MemoryDatasetBase() override { input_->Unref(); }
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return std::make_unique<MemoryIterator>(
        MemoryIterator::Params{
            this, name_utils::IteratorPrefix(kDatasetType, prefix, params)},
        cache_.get());
  }
  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }
  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.dataset_prefix = kMemoryDatasetPrefix;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }
  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return input_->Cardinality(options);
  };
  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    mutex_lock l(mu_);
    CardinalityOptions options;
    options.set_compute_level(CardinalityOptions::CARDINALITY_COMPUTE_LOW);
    int64_t cardinality = Cardinality(options);
    if (cardinality != kUnknownCardinality &&
        cardinality != kInfiniteCardinality && index >= cardinality) {
      return errors::OutOfRange("Index out of range [0, ", cardinality,
                                "):", index);
    }
    if (!dataset_random_access_cache_) {
      dataset_random_access_cache_ =
          std::make_unique<DatasetRandomAccessCache>(input_);
    }
    return dataset_random_access_cache_->Get(ctx, index, out_tensors);
  }
  Status Get(AnyContext ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    mutex_lock l(mu_);
    if (!iterator_random_access_cache_) {
      iterator_random_access_cache_ =
          std::make_unique<IteratorRandomAccessCache>(input_);
    }
    return iterator_random_access_cache_->Get(ctx, index, out_tensors);
  }
  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }
  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }
  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }
 protected:
  class MemoryIterator : public DatasetIterator<MemoryDatasetBase> {
   public:
    explicit MemoryIterator(const Params& params, MemoryCache* cache)
        : DatasetIterator<MemoryDatasetBase>(params),
          cache_(cache),
          global_shuffle_iterator_(dataset()) {}
    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      return InitializeIterator(ctx);
    }
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (ctx->index_mapper() != nullptr) {
        return global_shuffle_iterator_.GetNext(ctx, out_tensors,
                                                end_of_sequence);
      }
      mutex_lock l(mu_);
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }
   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       1);
    }
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (cache_->IsCompleted()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCacheCompleted, ""));
        TF_RETURN_IF_ERROR(
            WriteElementsToCheckpoint(writer, prefix(), cache_->data()));
      }
      TF_RETURN_IF_ERROR(global_shuffle_iterator_.Save(prefix(), ctx, writer));
      return SaveInput(ctx, writer, iterator_);
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      if (ctx->restored_element_count().has_value()) {
        return global_shuffle_iterator_.Restore(prefix(), ctx, reader);
      }
      mutex_lock l(mu_);
      iterator_.reset();
      cache_->Reset();
      if (reader->Contains(prefix(), kCacheCompleted)) {
        std::vector<std::vector<Tensor>> temp_cache;
        TF_RETURN_IF_ERROR(
            ReadElementsFromCheckpoint(ctx, reader, prefix(), &temp_cache));
        cache_->Complete(std::move(temp_cache));
      }
      TF_RETURN_IF_ERROR(InitializeIterator(ctx));
      return RestoreInput(ctx, reader, iterator_);
    }
   private:
    class MemoryWriterIterator : public DatasetIterator<MemoryDatasetBase> {
     public:
      explicit MemoryWriterIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDatasetBase>(params), cache_(cache) {}
      ~MemoryWriterIterator() override {
        mutex_lock l(mu_);
        if (!temp_cache_.empty() && !cache_->IsCompleted()) {
          LOG(WARNING) << kIncompleteCacheErrorMessage;
          cache_->Reset();
        }
      }
      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          if (!cache_->IsCompleted()) {
            VLOG(2) << "Finalizing the cache because EOF has been reached.";
            cache_->Complete(std::move(temp_cache_));
          }
          return absl::OkStatus();
        }
        RecordBufferEnqueue(ctx, *out_tensors);
        temp_cache_.emplace_back(*out_tensors);
        if (temp_cache_.size() == dataset()->input_->Cardinality()) {
          VLOG(2) << "Finalizing the cache because its size matches the "
                     "expected input cardinality.";
          cache_->Complete(std::move(temp_cache_));
        }
        return absl::OkStatus();
      }
     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         1);
      }
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (!cache_->IsCompleted()) {
          TF_RETURN_IF_ERROR(
              WriteElementsToCheckpoint(writer, prefix(), temp_cache_));
        }
        return SaveInput(ctx, writer, input_impl_);
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(prefix(), kCacheCompleted)) {
          TF_RETURN_IF_ERROR(
              ReadElementsFromCheckpoint(ctx, reader, prefix(), &temp_cache_));
        }
        return RestoreInput(ctx, reader, input_impl_);
      }
     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      MemoryCache* const cache_ TF_GUARDED_BY(mu_);  
      std::vector<std::vector<Tensor>> temp_cache_ TF_GUARDED_BY(mu_);
    };  
    class MemoryReaderIterator : public DatasetIterator<MemoryDatasetBase> {
     public:
      explicit MemoryReaderIterator(const Params& params, MemoryCache* cache)
          : DatasetIterator<MemoryDatasetBase>(params),
            cache_(cache),
            index_(0) {}
      Status Initialize(IteratorContext* ctx) override {
        tf_shared_lock l(mu_);
        for (size_t i = 0; i < cache_->size(); ++i) {
          RecordBufferEnqueue(ctx, cache_->at(i));
        }
        return absl::OkStatus();
      }
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (index_ < cache_->size()) {
          const std::vector<Tensor>& cache_tensors = cache_->at(index_);
          out_tensors->insert(out_tensors->begin(), cache_tensors.begin(),
                              cache_tensors.end());
          index_++;
          *end_of_sequence = false;
          return absl::OkStatus();
        } else {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
      }
     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         1);
      }
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kIndex, index_));
        return absl::OkStatus();
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        {
          int64_t temp = cache_->size();
          if (reader->Contains(prefix(), kIndex)) {
            TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kIndex, &temp));
          }
          index_ = static_cast<size_t>(temp);
        }
        return absl::OkStatus();
      }
     private:
      mutex mu_;
      MemoryCache* const cache_ TF_GUARDED_BY(mu_);  
      size_t index_ TF_GUARDED_BY(mu_);
    };  
    Status InitializeIterator(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (cache_->IsCompleted()) {
        iterator_ = std::make_unique<MemoryReaderIterator>(
            MemoryReaderIterator::Params{dataset(),
                                         strings::StrCat(prefix(), kImpl)},
            cache_);
      } else {
        iterator_ = std::make_unique<MemoryWriterIterator>(
            MemoryWriterIterator::Params{dataset(),
                                         strings::StrCat(prefix(), kImpl)},
            cache_);
      }
      TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
      return iterator_->Initialize(ctx);
    }
    mutex mu_;
    MemoryCache* cache_ TF_GUARDED_BY(mu_);  
    std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);
    GlobalShuffleIterator global_shuffle_iterator_;
  };  
  mutable mutex mu_;
  const DatasetBase* const input_;
  const std::shared_ptr<MemoryCache> cache_;
  mutable std::unique_ptr<DatasetRandomAccessCache> dataset_random_access_cache_
      TF_GUARDED_BY(mu_);
  mutable std::unique_ptr<IteratorRandomAccessCache>
      iterator_random_access_cache_;
  absl::Status random_indexing_compatible_ = absl::OkStatus();
};  
class CacheDatasetOp::MemoryDataset : public CacheDatasetOp::MemoryDatasetBase {
 public:
  MemoryDataset(OpKernelContext* ctx, const DatasetBase* input,
                MemoryCacheManager* manager, ResourceHandle&& resource_handle)
      : MemoryDatasetBase(ctx, input, manager->get()),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}
  ~MemoryDataset() override {
    manager_->Unref();
    Status s = resource_mgr_->Delete<MemoryCacheManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete cache resource: " << s.ToString();
    }
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_node, filename_node}, output));
    return absl::OkStatus();
  }
 private:
  MemoryCacheManager* const manager_;  
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  
};
class CacheDatasetOp::MemoryDatasetV2
    : public CacheDatasetOp::MemoryDatasetBase {
 public:
  MemoryDatasetV2(OpKernelContext* ctx, const DatasetBase* input,
                  MemoryCacheManager* manager, ResourceHandle&& resource_handle,
                  bool owns_resource)
      : MemoryDatasetBase(ctx, input, manager->get()),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}
  ~MemoryDatasetV2() override {
    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<MemoryCacheManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete cache resource: " << s.ToString();
      }
    }
  }
 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* filename_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(tstring(""), &filename_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_node, filename_node, resource_handle_node}, output));
    return absl::OkStatus();
  }
 private:
  MemoryCacheManager* const manager_;  
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  
};
CacheDatasetOp::CacheDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kCacheDataset ? 1 : 2) {}
void CacheDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  tstring filename;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kFileName, &filename));
  if (filename.empty()) {
    static std::atomic<int64_t> resource_id_counter(0);
    const string& container = ctx->resource_manager()->default_container();
    auto name = strings::StrCat(ctx->op_kernel().name(), "/", kMemoryCache, "_",
                                resource_id_counter.fetch_add(1));
    if (op_version_ == 2) {
      bool owns_resource = false;
      MemoryCacheManager* manager = nullptr;
      auto handle = HandleFromInput(ctx, 2);
      Status s = ctx->resource_manager()->Lookup<MemoryCacheManager>(
          handle.container(), handle.name(), &manager);
      if (errors::IsNotFound(s)) {
        owns_resource = true;
        OP_REQUIRES_OK(
            ctx,
            ctx->resource_manager()->LookupOrCreate<MemoryCacheManager>(
                container, name, &manager, [](MemoryCacheManager** manager) {
                  *manager = new MemoryCacheManager();
                  return absl::OkStatus();
                }));
        handle = MakeResourceHandle<MemoryCacheManager>(ctx, container, name);
      } else {
        OP_REQUIRES_OK(ctx, s);
      }
      *output = new MemoryDatasetV2(ctx, input, manager, std::move(handle),
                                    owns_resource);
    } else {
      MemoryCacheManager* manager;
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<MemoryCacheManager>(
                   container, name, &manager, [](MemoryCacheManager** manager) {
                     *manager = new MemoryCacheManager();
                     return absl::OkStatus();
                   }));
      auto handle =
          MakeResourceHandle<MemoryCacheManager>(ctx, container, name);
      *output = new MemoryDataset(ctx, input, manager, std::move(handle));
    }
  } else {
    if (op_version_ == 2) {
      *output =
          new FileDatasetV2(ctx, input, filename, ctx->env(), ctx->input(2));
    } else {
      *output = new FileDataset(ctx, input, filename, ctx->env());
    }
  }
}
namespace {
REGISTER_KERNEL_BUILDER(Name("CacheDataset").Device(DEVICE_CPU),
                        CacheDatasetOp);
REGISTER_KERNEL_BUILDER(Name("CacheDatasetV2").Device(DEVICE_CPU),
                        CacheDatasetOp);
}  
}  
}  