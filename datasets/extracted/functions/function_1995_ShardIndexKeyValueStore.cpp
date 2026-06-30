#include "tensorstore/kvstore/zarr3_sharding_indexed/zarr3_sharding_indexed.h"
#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/bit_vec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/internal/cache_key/std_vector.h"  
#include "tensorstore/internal/estimate_heap_usage/std_optional.h"  
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  
#include "tensorstore/serialization/std_vector.h"  
#include "tensorstore/util/execution/result_sender.h"  
#include "tensorstore/util/garbage_collection/std_vector.h"  
namespace tensorstore {
namespace zarr3_sharding_indexed {
namespace {
using ::tensorstore::internal_kvstore::DeleteRangeEntry;
using ::tensorstore::internal_kvstore::kReadModifyWrite;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
class ShardIndexKeyValueStore : public kvstore::Driver {
 public:
  explicit ShardIndexKeyValueStore(kvstore::DriverPtr base,
                                   ShardIndexLocation index_location,
                                   int64_t index_size_in_bytes)
      : base_(std::move(base)),
        index_location_(index_location),
        index_size_in_bytes_(index_size_in_bytes) {}
  Future<kvstore::ReadResult> Read(kvstore::Key key,
                                   kvstore::ReadOptions options) override {
    assert(options.byte_range == OptionalByteRangeRequest{});
    switch (index_location_) {
      case ShardIndexLocation::kStart:
        options.byte_range =
            OptionalByteRangeRequest::Range(0, index_size_in_bytes_);
        break;
      case ShardIndexLocation::kEnd:
        options.byte_range =
            OptionalByteRangeRequest::SuffixLength(index_size_in_bytes_);
        break;
    }
    return MapFutureError(
        InlineExecutor{},
        [](const absl::Status& status) {
          return internal::ConvertInvalidArgumentToFailedPrecondition(status);
        },
        base_->Read(std::move(key), std::move(options)));
  }
  std::string DescribeKey(std::string_view key) override {
    return tensorstore::StrCat("shard index in ", base_->DescribeKey(key));
  }
  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
  }
  kvstore::Driver* base() { return base_.get(); }
 private:
  kvstore::DriverPtr base_;
  ShardIndexLocation index_location_;
  int64_t index_size_in_bytes_;
};
class ShardIndexCache
    : public internal::KvsBackedCache<ShardIndexCache, internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ShardIndexCache, internal::AsyncCache>;
 public:
  using ReadData = ShardIndex;
  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardIndexCache;
    size_t ComputeReadDataSizeInBytes(const void* read_data) override {
      const auto& cache = GetOwningCache(*this);
      return read_data
                 ? cache.shard_index_params().num_entries * sizeof(uint64_t) * 2
                 : 0;
    }
    std::string GetKeyValueStoreKey() override {
      return GetOwningCache(*this).base_kvstore_path_;
    }
    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            std::shared_ptr<ReadData> read_data;
            if (value) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  auto shard_index,
                  DecodeShardIndex(*value,
                                   GetOwningCache(*this).shard_index_params()),
                  static_cast<void>(execution::set_error(receiver, _)));
              read_data = std::make_shared<ReadData>(std::move(shard_index));
            }
            execution::set_value(receiver, std::move(read_data));
          });
    }
  };
  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    ABSL_UNREACHABLE();
  }
  explicit ShardIndexCache(kvstore::DriverPtr base_kvstore,
                           std::string base_kvstore_path, Executor executor,
                           ShardIndexParameters&& params)
      : Base(kvstore::DriverPtr(new ShardIndexKeyValueStore(
            std::move(base_kvstore), params.index_location,
            params.index_codec_state->encoded_size()))),
        base_kvstore_path_(std::move(base_kvstore_path)),
        executor_(std::move(executor)),
        shard_index_params_(std::move(params)) {}
  ShardIndexKeyValueStore* shard_index_kvstore_driver() {
    return static_cast<ShardIndexKeyValueStore*>(this->Base::kvstore_driver());
  }
  kvstore::Driver* base_kvstore_driver() {
    return shard_index_kvstore_driver()->base();
  }
  const std::string& base_kvstore_path() const { return base_kvstore_path_; }
  const Executor& executor() { return executor_; }
  span<const Index> grid_shape() const {
    return span<const Index>(shard_index_params_.index_shape.data(),
                             shard_index_params_.index_shape.size() - 1);
  }
  const ShardIndexParameters& shard_index_params() const {
    return shard_index_params_;
  }
  std::string base_kvstore_path_;
  Executor executor_;
  ShardIndexParameters shard_index_params_;
};
class ShardedKeyValueStoreWriteCache
    : public internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                      internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                        internal::AsyncCache>;
 public:
  using ReadData = ShardEntries;
  explicit ShardedKeyValueStoreWriteCache(
      internal::CachePtr<ShardIndexCache> shard_index_cache)
      : Base(kvstore::DriverPtr(shard_index_cache->base_kvstore_driver())),
        shard_index_cache_(std::move(shard_index_cache)) {}
  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;
    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return internal::EstimateHeapUsage(*static_cast<const ReadData*>(data));
    }
    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            ShardEntries entries;
            const auto& shard_index_params =
                GetOwningCache(*this).shard_index_params();
            if (value) {
              TENSORSTORE_ASSIGN_OR_RETURN(
                  entries, DecodeShard(*value, shard_index_params),
                  static_cast<void>(execution::set_error(receiver, _)));
            } else {
              entries.entries.resize(shard_index_params.num_entries);
            }
            execution::set_value(
                receiver, std::make_shared<ShardEntries>(std::move(entries)));
          });
    }
    void DoEncode(std::shared_ptr<const ShardEntries> data,
                  EncodeReceiver receiver) override {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto encoded_shard,
          EncodeShard(*data, GetOwningCache(*this).shard_index_params()),
          static_cast<void>(execution::set_error(receiver, _)));
      execution::set_value(receiver, std::move(encoded_shard));
    }
    std::string GetKeyValueStoreKey() override {
      return GetOwningCache(*this).base_kvstore_path();
    }
  };
  class TransactionNode : public Base::TransactionNode,
                          public internal_kvstore::AtomicMultiPhaseMutation {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;
    using Base::TransactionNode::TransactionNode;
    absl::Mutex& mutex() override { return this->mutex_; }
    void PhaseCommitDone(size_t next_phase) override {}
    internal::TransactionState::Node& GetTransactionNode() override {
      return *this;
    }
    void Abort() override {
      this->AbortRemainingPhases();
      Base::TransactionNode::Abort();
    }
    std::string DescribeKey(std::string_view key) override {
      auto& cache = GetOwningCache(*this);
      return tensorstore::StrCat(
          DescribeInternalKey(key, cache.shard_index_params().grid_shape()),
          " in ",
          cache.kvstore_driver()->DescribeKey(cache.base_kvstore_path()));
    }
    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;
    void StartApply();
    void AllEntriesDone(
        internal_kvstore::SinglePhaseMutation& single_phase_mutation) override;
    void MergeForWriteback(bool conditional);
    void RecordEntryWritebackError(
        internal_kvstore::ReadModifyWriteEntry& entry,
        absl::Status error) override {
      absl::MutexLock lock(&mutex_);
      if (apply_status_.ok()) {
        apply_status_ = std::move(error);
      }
    }
    void Revoke() override {
      Base::TransactionNode::Revoke();
      { UniqueWriterLock(*this); }
      this->RevokeAllEntries();
    }
    void WritebackSuccess(ReadState&& read_state) override;
    void WritebackError() override;
    void InvalidateReadState() override;
    bool MultiPhaseReadsCommitted() override { return this->reads_committed_; }
    void Read(
        internal_kvstore::ReadModifyWriteEntry& entry,
        kvstore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
        kvstore::ReadModifyWriteTarget::ReadReceiver&& receiver) override {
      this->AsyncCache::TransactionNode::Read({options.staleness_bound})
          .ExecuteWhenReady(WithExecutor(
              GetOwningCache(*this).executor(),
              [&entry,
               if_not_equal =
                   std::move(options.generation_conditions.if_not_equal),
               receiver = std::move(receiver)](
                  ReadyFuture<const void> future) mutable {
                if (!future.result().ok()) {
                  execution::set_error(receiver, future.result().status());
                  return;
                }
                execution::submit(HandleShardReadSuccess(entry, if_not_equal),
                                  receiver);
              }));
    }
    static Result<kvstore::ReadResult> HandleShardReadSuccess(
        internal_kvstore::ReadModifyWriteEntry& entry,
        const StorageGeneration& if_not_equal) {
      auto& self = static_cast<TransactionNode&>(entry.multi_phase());
      TimestampedStorageGeneration stamp;
      std::shared_ptr<const ShardEntries> entries;
      {
        AsyncCache::ReadLock<ShardEntries> lock{self};
        stamp = lock.stamp();
        entries = lock.shared_data();
      }
      if (!StorageGeneration::IsUnknown(stamp.generation) &&
          stamp.generation == if_not_equal) {
        return kvstore::ReadResult::Unspecified(std::move(stamp));
      }
      if (StorageGeneration::IsDirty(stamp.generation)) {
        stamp.generation =
            StorageGeneration::AddLayer(std::move(stamp.generation));
      }
      auto entry_id = InternalKeyToEntryId(entry.key_);
      const auto& shard_entry = entries->entries[entry_id];
      if (!shard_entry) {
        return kvstore::ReadResult::Missing(std::move(stamp));
      } else {
        return kvstore::ReadResult::Value(*shard_entry, std::move(stamp));
      }
    }
    ApplyReceiver apply_receiver_;
    ApplyOptions apply_options_;
    absl::Status apply_status_;
  };
  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }
  const internal::CachePtr<ShardIndexCache>& shard_index_cache() const {
    return shard_index_cache_;
  }
  const Executor& executor() { return shard_index_cache()->executor(); }
  const ShardIndexParameters& shard_index_params() const {
    return shard_index_cache_->shard_index_params();
  }
  int64_t num_entries_per_shard() const {
    return shard_index_cache_->shard_index_params().num_entries;
  }
  const std::string& base_kvstore_path() const {
    return shard_index_cache_->base_kvstore_path();
  }
  internal::CachePtr<ShardIndexCache> shard_index_cache_;
};
void ShardedKeyValueStoreWriteCache::TransactionNode::InvalidateReadState() {
  Base::TransactionNode::InvalidateReadState();
  internal_kvstore::InvalidateReadState(phases_);
}
void ShardedKeyValueStoreWriteCache::TransactionNode::DoApply(
    ApplyOptions options, ApplyReceiver receiver) {
  apply_receiver_ = std::move(receiver);
  apply_options_ = options;
  apply_status_ = absl::OkStatus();
  GetOwningCache(*this).executor()([this] { this->StartApply(); });
}
void ShardedKeyValueStoreWriteCache::TransactionNode::StartApply() {
  RetryAtomicWriteback(apply_options_.staleness_bound);
}
void ShardedKeyValueStoreWriteCache::TransactionNode::AllEntriesDone(
    internal_kvstore::SinglePhaseMutation& single_phase_mutation) {
  if (!apply_status_.ok()) {
    execution::set_error(std::exchange(apply_receiver_, {}),
                         std::exchange(apply_status_, {}));
    return;
  }
  auto& self = *this;
  GetOwningCache(*this).executor()([&self] {
    TimestampedStorageGeneration stamp;
    bool mismatch = false;
    bool modified = false;
    int64_t num_entries = 0;
    auto& cache = GetOwningCache(self);
    const int64_t num_entries_per_shard = cache.num_entries_per_shard();
    for (auto& entry : self.phases_.entries_) {
      if (entry.entry_type() != kReadModifyWrite) {
        auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
        auto [begin_id, end_id] = InternalKeyRangeToEntryRange(
            dr_entry.key_, dr_entry.exclusive_max_, num_entries_per_shard);
        modified = true;
        num_entries += end_id - begin_id;
        continue;
      }
      auto& buffered_entry =
          static_cast<AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry&>(
              entry);
      if (buffered_entry.value_state_ != kvstore::ReadResult::kUnspecified) {
        modified = true;
        ++num_entries;
      }
      auto& entry_stamp = buffered_entry.stamp();
      if (StorageGeneration::IsConditional(entry_stamp.generation)) {
        if (!StorageGeneration::IsUnknown(stamp.generation) &&
            StorageGeneration::Clean(stamp.generation) !=
                StorageGeneration::Clean(entry_stamp.generation)) {
          mismatch = true;
          break;
        } else {
          stamp = entry_stamp;
        }
      }
    }
    if (mismatch) {
      self.apply_options_.staleness_bound = absl::Now();
      self.StartApply();
      return;
    }
    if (!modified && StorageGeneration::IsUnknown(stamp.generation) &&
        self.apply_options_.apply_mode !=
            ApplyOptions::ApplyMode::kSpecifyUnchanged) {
      internal::AsyncCache::ReadState update;
      update.stamp = TimestampedStorageGeneration::Unconditional();
      execution::set_value(std::exchange(self.apply_receiver_, {}),
                           std::move(update));
      return;
    }
    if (!StorageGeneration::IsUnknown(stamp.generation) ||
        num_entries != num_entries_per_shard) {
      self.internal::AsyncCache::TransactionNode::Read(
              {self.apply_options_.staleness_bound})
          .ExecuteWhenReady([&self](ReadyFuture<const void> future) {
            if (!future.result().ok()) {
              execution::set_error(std::exchange(self.apply_receiver_, {}),
                                   future.result().status());
              return;
            }
            GetOwningCache(self).executor()(
                [&self] { self.MergeForWriteback(true); });
          });
      return;
    }
    self.MergeForWriteback(false);
  });
}
void ShardedKeyValueStoreWriteCache::TransactionNode::MergeForWriteback(
    bool conditional) {
  TimestampedStorageGeneration stamp;
  ShardEntries new_entries;
  if (conditional) {
    auto lock = internal::AsyncCache::ReadLock<ShardEntries>{*this};
    stamp = lock.stamp();
    new_entries = *lock.shared_data();
  } else {
    stamp = TimestampedStorageGeneration::Unconditional();
  }
  auto& cache = GetOwningCache(*this);
  const int64_t num_entries_per_shard = cache.num_entries_per_shard();
  const bool has_existing_entries = !new_entries.entries.empty();
  new_entries.entries.resize(num_entries_per_shard);
  bool mismatch = false;
  bool changed = false;
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      auto [begin_id, end_id] = InternalKeyRangeToEntryRange(
          dr_entry.key_, dr_entry.exclusive_max_, num_entries_per_shard);
      if (has_existing_entries) {
        for (EntryId id = begin_id; id < end_id; ++id) {
          new_entries.entries[id] = std::nullopt;
        }
      }
      changed = true;
      continue;
    }
    auto& buffered_entry =
        static_cast<internal_kvstore::AtomicMultiPhaseMutation::
                        BufferedReadModifyWriteEntry&>(entry);
    auto& entry_stamp = buffered_entry.stamp();
    if (StorageGeneration::IsConditional(entry_stamp.generation) &&
        StorageGeneration::Clean(entry_stamp.generation) !=
            StorageGeneration::Clean(stamp.generation)) {
      mismatch = true;
      break;
    }
    if (buffered_entry.value_state_ == kvstore::ReadResult::kUnspecified ||
        !StorageGeneration::IsInnerLayerDirty(entry_stamp.generation)) {
      continue;
    }
    auto entry_id = InternalKeyToEntryId(buffered_entry.key_);
    auto& new_entry = new_entries.entries[entry_id];
    if (buffered_entry.value_state_ == kvstore::ReadResult::kValue) {
      new_entry = buffered_entry.value_;
      changed = true;
    } else if (new_entry) {
      new_entry = std::nullopt;
      changed = true;
    } else if (!conditional) {
      changed = true;
    }
  }
  if (mismatch) {
    apply_options_.staleness_bound = absl::Now();
    this->StartApply();
    return;
  }
  internal::AsyncCache::ReadState update;
  update.stamp = std::move(stamp);
  if (changed) {
    update.stamp.generation.MarkDirty();
  }
  update.data = std::make_shared<ShardEntries>(std::move(new_entries));
  execution::set_value(std::exchange(apply_receiver_, {}), std::move(update));
}
void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackSuccess(
    ReadState&& read_state) {
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) {
      internal_kvstore::WritebackSuccess(static_cast<DeleteRangeEntry&>(entry));
    } else {
      internal_kvstore::WritebackSuccess(
          static_cast<internal_kvstore::ReadModifyWriteEntry&>(entry),
          read_state.stamp);
    }
  }
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
}
void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackError() {
  internal_kvstore::WritebackError(phases_);
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackError();
}
struct ShardedKeyValueStoreSpecData {
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  kvstore::Spec base;
  std::vector<Index> grid_shape;
  internal_zarr3::ZarrCodecChainSpec index_codecs;
  ShardIndexLocation index_location;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ShardedKeyValueStoreSpecData,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults,
                                          ::nlohmann::json::object_t)
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.cache_pool, x.data_copy_concurrency, x.base, x.grid_shape,
             x.index_codecs, x.index_location);
  };
};
namespace jb = ::tensorstore::internal_json_binding;
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ShardedKeyValueStoreSpecData,
    jb::Object(
        jb::Member("base",
                   jb::Projection<&ShardedKeyValueStoreSpecData::base>()),
        jb::Member(
            "grid_shape",
            jb::Projection<&ShardedKeyValueStoreSpecData::grid_shape>(
                jb::Validate([](const auto& options,
                                auto* obj) { return ValidateGridShape(*obj); },
                             jb::ChunkShapeVector(nullptr)))),
        jb::Member("index_codecs",
                   jb::Projection<&ShardedKeyValueStoreSpecData::index_codecs>(
                       internal_zarr3::ZarrCodecChainJsonBinder<
                           false>)),
        jb::Member(
            "index_location",
            jb::Projection<&ShardedKeyValueStoreSpecData::index_location>(
                jb::DefaultValue<jb::kAlwaysIncludeDefaults>([](auto* x) {
                  *x = ShardIndexLocation::kEnd;
                }))),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection<&ShardedKeyValueStoreSpecData::cache_pool>()),
        jb::Member(
            internal::DataCopyConcurrencyResource::id,
            jb::Projection<
                &ShardedKeyValueStoreSpecData::data_copy_concurrency>())));
class ShardedKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          ShardedKeyValueStoreSpec, ShardedKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "zarr3_sharding_indexed";
  Future<kvstore::DriverPtr> DoOpen() const override;
  Result<kvstore::Spec> GetBase(std::string_view path) const override {
    return data_.base;
  }
};
class ShardedKeyValueStore
    : public internal_kvstore::RegisteredDriver<ShardedKeyValueStore,
                                                ShardedKeyValueStoreSpec> {
 public:
  explicit ShardedKeyValueStore(ShardedKeyValueStoreParameters&& params,
                                std::string_view shared_cache_key = {});
  Future<ReadResult> Read(Key key, ReadOptions options) override;
  void ListImpl(ListOptions options, ListReceiver receiver) override;
  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;
  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override;
  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range) override;
  Future<const void> DeleteRange(KeyRange range) override;
  std::string DescribeKey(std::string_view key) override;
  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final;
  Result<KvStore> GetBase(std::string_view path,
                          const Transaction& transaction) const override;
  kvstore::Driver* base_kvstore_driver() const {
    return shard_index_cache()->base_kvstore_driver();
  }
  const ShardIndexParameters& shard_index_params() const {
    return shard_index_cache()->shard_index_params();
  }
  const Executor& executor() const { return shard_index_cache()->executor(); }
  const std::string& base_kvstore_path() const {
    return shard_index_cache()->base_kvstore_path();
  }
  const internal::CachePtr<ShardIndexCache>& shard_index_cache() const {
    return write_cache_->shard_index_cache_;
  }
  absl::Status GetBoundSpecData(ShardedKeyValueStoreSpecData& spec) const;
  internal::CachePtr<ShardedKeyValueStoreWriteCache> write_cache_;
  struct DataForSpec {
    Context::Resource<internal::CachePoolResource> cache_pool_resource;
    Context::Resource<internal::DataCopyConcurrencyResource>
        data_copy_concurrency_resource;
    ZarrCodecChainSpec index_codecs;
  };
  std::unique_ptr<DataForSpec> data_for_spec_;
};
ShardedKeyValueStore::ShardedKeyValueStore(
    ShardedKeyValueStoreParameters&& params,
    std::string_view shared_cache_key) {
  write_cache_ = internal::GetCache<ShardedKeyValueStoreWriteCache>(
      params.cache_pool.get(), shared_cache_key, [&] {
        return std::make_unique<ShardedKeyValueStoreWriteCache>(
            internal::GetCache<ShardIndexCache>(
                params.cache_pool.get(), "", [&] {
                  return std::make_unique<ShardIndexCache>(
                      std::move(params.base_kvstore),
                      std::move(params.base_kvstore_path),
                      std::move(params.executor),
                      std::move(params.index_params));
                }));
      });
  this->SetBatchNestingDepth(
      this->base_kvstore_driver()->BatchNestingDepth() +
      1 +  
      1    
  );
}
class ReadOperationState;
using ReadOperationStateBase = internal_kvstore_batch::BatchReadEntry<
    ShardedKeyValueStore, internal_kvstore_batch::ReadRequest<
                              EntryId, kvstore::ReadGenerationConditions>>;
class ReadOperationState
    : public ReadOperationStateBase,
      public internal::AtomicReferenceCount<ReadOperationState> {
 public:
  explicit ReadOperationState(BatchEntryKey&& batch_entry_key_)
      : ReadOperationStateBase(std::move(batch_entry_key_)),
        internal::AtomicReferenceCount<ReadOperationState>(
            1) {}
 private:
  internal::PinnedCacheEntry<ShardIndexCache> shard_index_cache_entry_;
  Batch successor_batch_{no_batch};
  void Submit(Batch::View batch) override {
    const auto& executor = driver().executor();
    executor(
        [this, batch = Batch(batch)] { this->ProcessBatch(std::move(batch)); });
  }
  void ProcessBatch(Batch batch) {
    internal::IntrusivePtr<ReadOperationState> self(this,
                                                    internal::adopt_object_ref);
    if (ShouldReadEntireShard()) {
      ReadEntireShard(std::move(self), std::move(batch));
      return;
    }
    shard_index_cache_entry_ =
        GetCacheEntry(driver().shard_index_cache(), std::string_view{});
    auto shard_index_read_future = shard_index_cache_entry_->Read(
        {this->request_batch.staleness_bound, batch});
    if (batch) {
      if (!shard_index_read_future.ready()) {
        successor_batch_ = Batch::New();
      } else {
        successor_batch_ = std::move(batch);
      }
    }
    std::move(shard_index_read_future)
        .ExecuteWhenReady(
            [self = std::move(self)](ReadyFuture<const void> future) mutable {
              const auto& executor = self->driver().executor();
              executor([self = std::move(self), status = future.status()] {
                if (!status.ok()) {
                  internal_kvstore_batch::SetCommonResult<Request>(
                      self->request_batch.requests, {status});
                  return;
                }
                OnShardIndexReady(std::move(self));
              });
            });
  }
  bool ShouldReadEntireShard() {
    const int64_t num_entries_per_shard =
        driver().shard_index_params().num_entries;
    if (request_batch.requests.size() < num_entries_per_shard) {
      return false;
    }
    const auto& first_request = request_batch.requests[0];
    BitVec<> covered_entries(num_entries_per_shard);
    int64_t num_covered = 0;
    for (const auto& request : request_batch.requests) {
      if (std::get<kvstore::ReadGenerationConditions>(request) !=
          std::get<kvstore::ReadGenerationConditions>(first_request)) {
        return false;
      }
      if (std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
              .byte_range.IsFull()) {
        auto ref = covered_entries[std::get<EntryId>(request)];
        if (!ref) ++num_covered;
        ref = true;
      }
    }
    if (num_covered != num_entries_per_shard) {
      return false;
    }
    return true;
  }
  static void ReadEntireShard(internal::IntrusivePtr<ReadOperationState> self,
                              Batch batch) {
    auto& first_request = self->request_batch.requests[0];
    kvstore::ReadOptions read_options;
    read_options.batch = std::move(batch);
    read_options.generation_conditions =
        std::move(std::get<kvstore::ReadGenerationConditions>(first_request));
    read_options.staleness_bound = self->request_batch.staleness_bound;
    auto& driver = self->driver();
    driver.base_kvstore_driver()
        ->Read(driver.base_kvstore_path(), std::move(read_options))
        .ExecuteWhenReady([self = std::move(self)](
                              ReadyFuture<kvstore::ReadResult> future) mutable {
          const auto& executor = self->driver().executor();
          executor([self = std::move(self), future = std::move(future)] {
            OnFullShardReady(std::move(self), std::move(future.result()));
          });
        });
  }
  static void OnFullShardReady(internal::IntrusivePtr<ReadOperationState> self,
                               Result<kvstore::ReadResult>&& result) {
    if (!result.ok() || !result->has_value()) {
      internal_kvstore_batch::SetCommonResult(self->request_batch.requests,
                                              std::move(result));
      return;
    }
    auto& read_result = *result;
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto shard_index,
        DecodeShardIndexFromFullShard(read_result.value,
                                      self->driver().shard_index_params()),
        internal_kvstore_batch::SetCommonResult(self->request_batch.requests,
                                                _));
    const auto complete_request = [&](Request& request) {
      auto& byte_range_request =
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
      const auto index_entry = shard_index[std::get<EntryId>(request)];
      if (index_entry.IsMissing()) {
        byte_range_request.promise.SetResult(
            kvstore::ReadResult::Missing(read_result.stamp));
        return;
      }
      TENSORSTORE_RETURN_IF_ERROR(
          index_entry.Validate(std::get<EntryId>(request),
                               read_result.value.size()),
          static_cast<void>(byte_range_request.promise.SetResult(_)));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto validated_byte_range,
          byte_range_request.byte_range.Validate(index_entry.length),
          static_cast<void>(byte_range_request.promise.SetResult(_)));
      validated_byte_range.inclusive_min += index_entry.offset;
      validated_byte_range.exclusive_max += index_entry.offset;
      kvstore::ReadResult request_read_result;
      request_read_result.stamp = read_result.stamp;
      request_read_result.state = kvstore::ReadResult::kValue;
      request_read_result.value =
          internal::GetSubCord(read_result.value, validated_byte_range);
      byte_range_request.promise.SetResult(std::move(request_read_result));
    };
    for (auto& request : self->request_batch.requests) {
      complete_request(request);
    }
  }
  static void OnShardIndexReady(
      internal::IntrusivePtr<ReadOperationState> self) {
    std::shared_ptr<const ShardIndex> shard_index;
    TimestampedStorageGeneration stamp;
    {
      auto lock = internal::AsyncCache::ReadLock<ShardIndexCache::ReadData>(
          *self->shard_index_cache_entry_);
      stamp = lock.stamp();
      shard_index = lock.shared_data();
    }
    assert(!StorageGeneration::IsUnknown(stamp.generation));
    if (!shard_index) {
      internal_kvstore_batch::SetCommonResult(
          self->request_batch.requests,
          kvstore::ReadResult::Missing(std::move(stamp)));
      return;
    }
    auto successor_batch = std::move(self->successor_batch_);
    if (successor_batch) {
      self->successor_batch_ = Batch::New();
    }
    const auto process_request = [&](Request& request) {
      ShardIndexEntry index_entry = ShardIndexEntry::Missing();
      kvstore::ReadResult::State state;
      if (!std::get<kvstore::ReadGenerationConditions>(request).Matches(
              stamp.generation)) {
        state = kvstore::ReadResult::kUnspecified;
      } else {
        index_entry = (*shard_index)[std::get<EntryId>(request)];
        state = kvstore::ReadResult::kMissing;
      }
      auto& byte_range_request =
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
      if (index_entry.IsMissing()) {
        byte_range_request.promise.SetResult(
            kvstore::ReadResult{state, {}, stamp});
        return;
      }
      TENSORSTORE_RETURN_IF_ERROR(
          index_entry.Validate(std::get<EntryId>(request)),
          static_cast<void>(byte_range_request.promise.SetResult(
              self->shard_index_cache_entry_->AnnotateError(
                  _,
                  true))));
      assert(byte_range_request.byte_range.SatisfiesInvariants());
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto validated_byte_range,
          byte_range_request.byte_range.Validate(index_entry.length),
          static_cast<void>(byte_range_request.promise.SetResult(_)));
      if (validated_byte_range.inclusive_min ==
          validated_byte_range.exclusive_max) {
        byte_range_request.promise.SetResult(kvstore::ReadResult{
            kvstore::ReadResult::kValue, absl::Cord(), stamp});
        return;
      }
      kvstore::ReadOptions kvs_read_options;
      kvs_read_options.generation_conditions.if_equal = stamp.generation;
      kvs_read_options.staleness_bound = self->request_batch.staleness_bound;
      kvs_read_options.batch = successor_batch;
      kvs_read_options.byte_range =
          ByteRange{static_cast<int64_t>(index_entry.offset +
                                         validated_byte_range.inclusive_min),
                    static_cast<int64_t>(index_entry.offset +
                                         validated_byte_range.exclusive_max)};
      self->driver()
          .base_kvstore_driver()
          ->Read(std::string(self->driver().base_kvstore_path()),
                 std::move(kvs_read_options))
          .ExecuteWhenReady([self, &request](ReadyFuture<kvstore::ReadResult>
                                                 future) mutable {
            const auto& status = future.status();
            if (!status.ok()) {
              std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
                  .promise.SetResult(status);
              return;
            }
            const auto& executor = self->driver().executor();
            executor([self = std::move(self), &request,
                      future = std::move(future)] {
              OnValueReady(std::move(self), request, std::move(future.value()));
            });
          });
    };
    for (auto& request : self->request_batch.requests) {
      process_request(request);
    }
  }
  static void OnValueReady(internal::IntrusivePtr<ReadOperationState> self,
                           Request& request, kvstore::ReadResult&& value) {
    if (value.aborted()) {
      MakeRequest<ReadOperationState>(self->driver(), self->successor_batch_,
                                      value.stamp.time, std::move(request));
      return;
    }
    std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
        .promise.SetResult(std::move(value));
  }
};
Future<kvstore::ReadResult> ShardedKeyValueStore::Read(Key key,
                                                       ReadOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      EntryId entry_id,
      KeyToEntryIdOrError(key, shard_index_params().grid_shape()));
  auto [promise, future] = PromiseFuturePair<kvstore::ReadResult>::Make();
  ReadOperationState::MakeRequest<ReadOperationState>(
      *this, options.batch, options.staleness_bound,
      ReadOperationState::Request{{std::move(promise), options.byte_range},
                                  {entry_id},
                                  std::move(options.generation_conditions)});
  return std::move(future);
}
struct ListOperationState
    : public internal::FlowSenderOperationState<kvstore::ListEntry> {
  using Base = internal::FlowSenderOperationState<kvstore::ListEntry>;
  using Base::Base;
  internal::PinnedCacheEntry<ShardIndexCache> shard_index_cache_entry_;
  kvstore::ListOptions options_;
  static void Start(ShardedKeyValueStore& store, kvstore::ListOptions&& options,
                    ListReceiver&& receiver) {
    options.range = KeyRangeToInternalKeyRange(
        options.range, store.shard_index_params().grid_shape());
    auto self =
        internal::MakeIntrusivePtr<ListOperationState>(std::move(receiver));
    self->options_ = std::move(options);
    self->shard_index_cache_entry_ =
        GetCacheEntry(store.shard_index_cache(), std::string_view{});
    auto shard_index_read_future =
        self->shard_index_cache_entry_->Read({self->options_.staleness_bound});
    auto* self_ptr = self.get();
    LinkValue(
        WithExecutor(store.executor(),
                     [self = std::move(self)](Promise<void> promise,
                                              ReadyFuture<const void> future) {
                       if (self->cancelled()) return;
                       self->OnShardIndexReady();
                     }),
        self_ptr->promise, std::move(shard_index_read_future));
  }
  void OnShardIndexReady() {
    auto shard_index =
        internal::AsyncCache::ReadLock<ShardIndex>(*shard_index_cache_entry_)
            .shared_data();
    if (!shard_index) {
      return;
    }
    const auto& shard_index_params =
        GetOwningCache(*shard_index_cache_entry_).shard_index_params();
    span<const Index> grid_shape = shard_index_params.grid_shape();
    auto start_index = InternalKeyToEntryId(options_.range.inclusive_min);
    auto end_index = InternalKeyToEntryId(options_.range.exclusive_max);
    auto& receiver = shared_receiver->receiver;
    for (EntryId i = start_index; i < end_index; ++i) {
      auto index_entry = (*shard_index)[i];
      if (index_entry.IsMissing()) continue;
      auto key = EntryIdToKey(i, grid_shape);
      key.erase(0, options_.strip_prefix_length);
      execution::set_value(receiver,
                           ListEntry{
                               std::move(key),
                               ListEntry::checked_size(index_entry.length),
                           });
    }
  }
};
void ShardedKeyValueStore::ListImpl(ListOptions options,
                                    ListReceiver receiver) {
  ListOperationState::Start(*this, std::move(options), std::move(receiver));
}
Future<TimestampedStorageGeneration> ShardedKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  return internal_kvstore::WriteViaTransaction(
      this, std::move(key), std::move(value), std::move(options));
}
absl::Status ShardedKeyValueStore::ReadModifyWrite(
    internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
    ReadModifyWriteSource& source) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      EntryId entry_id,
      KeyToEntryIdOrError(key, shard_index_params().grid_shape()));
  key = EntryIdToInternalKey(entry_id);
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->ReadModifyWrite(phase, std::move(key), source);
  if (!transaction) {
    transaction.reset(node.unlock()->transaction());
  }
  return absl::OkStatus();
}
absl::Status ShardedKeyValueStore::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  range = KeyRangeToInternalKeyRange(range, shard_index_params().grid_shape());
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->DeleteRange(std::move(range));
  return absl::OkStatus();
}
Future<const void> ShardedKeyValueStore::DeleteRange(KeyRange range) {
  range = KeyRangeToInternalKeyRange(range, shard_index_params().grid_shape());
  internal::OpenTransactionPtr transaction;
  auto entry = GetCacheEntry(write_cache_, std::string_view{});
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*entry, transaction));
  node->DeleteRange(std::move(range));
  return node->transaction()->future();
}
std::string ShardedKeyValueStore::DescribeKey(std::string_view key) {
  return tensorstore::StrCat(
      zarr3_sharding_indexed::DescribeKey(key,
                                          shard_index_params().grid_shape()),
      " in ", base_kvstore_driver()->DescribeKey(base_kvstore_path()));
}
kvstore::SupportedFeatures ShardedKeyValueStore::GetSupportedFeatures(
    const KeyRange& key_range) const {
  return base_kvstore_driver()->GetSupportedFeatures(
      KeyRange::Singleton(base_kvstore_path()));
}
Result<KvStore> ShardedKeyValueStore::GetBase(
    std::string_view path, const Transaction& transaction) const {
  return KvStore(kvstore::DriverPtr(base_kvstore_driver()), base_kvstore_path(),
                 transaction);
}
}  
}  
namespace garbage_collection {
template <>
struct GarbageCollection<zarr3_sharding_indexed::ShardedKeyValueStore> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const zarr3_sharding_indexed::ShardedKeyValueStore& value) {
    garbage_collection::GarbageCollectionVisit(visitor,
                                               *value.base_kvstore_driver());
  }
};
}  
namespace zarr3_sharding_indexed {
absl::Status ShardedKeyValueStore::GetBoundSpecData(
    ShardedKeyValueStoreSpecData& spec) const {
  if (!data_for_spec_) {
    return absl::UnimplementedError("");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(spec.base.driver,
                               base_kvstore_driver()->GetBoundSpec());
  spec.base.path = base_kvstore_path();
  spec.data_copy_concurrency = data_for_spec_->data_copy_concurrency_resource;
  spec.cache_pool = data_for_spec_->cache_pool_resource;
  spec.index_codecs = data_for_spec_->index_codecs;
  const auto& shard_index_params = this->shard_index_params();
  spec.index_location = shard_index_params.index_location;
  spec.grid_shape.assign(shard_index_params.index_shape.begin(),
                         shard_index_params.index_shape.end() - 1);
  return absl::OkStatus();
}
Future<kvstore::DriverPtr> ShardedKeyValueStoreSpec::DoOpen() const {
  ShardIndexParameters index_params;
  index_params.index_location = data_.index_location;
  TENSORSTORE_RETURN_IF_ERROR(
      index_params.Initialize(data_.index_codecs, data_.grid_shape));
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const ShardedKeyValueStoreSpec>(this),
       index_params =
           std::move(index_params)](kvstore::KvStore& base_kvstore) mutable
      -> Result<kvstore::DriverPtr> {
        std::string cache_key;
        internal::EncodeCacheKey(
            &cache_key, base_kvstore.driver, base_kvstore.path,
            spec->data_.data_copy_concurrency, spec->data_.grid_shape,
            spec->data_.index_codecs);
        ShardedKeyValueStoreParameters params;
        params.base_kvstore = std::move(base_kvstore.driver);
        params.base_kvstore_path = std::move(base_kvstore.path);
        params.executor = spec->data_.data_copy_concurrency->executor;
        params.cache_pool = *spec->data_.cache_pool;
        params.index_params = std::move(index_params);
        auto driver = internal::MakeIntrusivePtr<ShardedKeyValueStore>(
            std::move(params), cache_key);
        driver->data_for_spec_.reset(new ShardedKeyValueStore::DataForSpec{
            spec->data_.cache_pool,
            spec->data_.data_copy_concurrency,
            spec->data_.index_codecs,
        });
        return driver;
      },
      kvstore::Open(data_.base));
}
kvstore::DriverPtr GetShardedKeyValueStore(
    ShardedKeyValueStoreParameters&& parameters) {
  return kvstore::DriverPtr(new ShardedKeyValueStore(std::move(parameters)));
}
}  
}  
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::zarr3_sharding_indexed::ShardedKeyValueStoreSpec>
    registration;
}  