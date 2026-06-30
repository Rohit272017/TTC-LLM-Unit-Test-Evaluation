#include "tensorstore/kvstore/transaction.h"
#include <stddef.h>
#include <stdint.h>
#include <cassert>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/compare.h"
#include "tensorstore/internal/compare.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/future_sender.h"  
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_kvstore {
namespace {
auto& kvstore_transaction_retries = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/transaction_retries",
    internal_metrics::MetricMetadata("Count of kvstore transaction retries"));
template <typename Controller>
void ReportWritebackError(Controller controller, std::string_view action,
                          const absl::Status& error,
                          SourceLocation loc = SourceLocation::current()) {
  controller.Error(kvstore::Driver::AnnotateErrorWithKeyDescription(
      controller.DescribeKey(controller.GetKey()), action, error, loc));
}
template <typename Controller>
void PerformWriteback(Driver* driver, Controller controller,
                      ReadResult read_result) {
  if (!StorageGeneration::IsDirty(read_result.stamp.generation)) {
    if (!StorageGeneration::IsConditional(read_result.stamp.generation) ||
        read_result.stamp.time > controller.GetTransactionNode()
                                     .transaction()
                                     ->commit_start_time()) {
      controller.Success(std::move(read_result.stamp));
      return;
    }
    ReadOptions read_options;
    auto if_not_equal =
        StorageGeneration::Clean(std::move(read_result.stamp.generation));
    read_options.generation_conditions.if_not_equal = if_not_equal;
    read_options.byte_range = OptionalByteRangeRequest{0, 0};
    auto future = driver->Read(controller.GetKey(), std::move(read_options));
    future.Force();
    std::move(future).ExecuteWhenReady(
        [controller, if_not_equal = std::move(if_not_equal)](
            ReadyFuture<ReadResult> future) mutable {
          auto& r = future.result();
          if (!r.ok()) {
            ReportWritebackError(controller, "reading", r.status());
          } else if (r->aborted() || r->stamp.generation == if_not_equal) {
            controller.Success(std::move(r->stamp));
          } else {
            controller.Retry(r->stamp.time);
          }
        });
    return;
  }
  WriteOptions write_options;
  assert(!read_result.aborted());
  write_options.generation_conditions.if_equal =
      StorageGeneration::Clean(std::move(read_result.stamp.generation));
  auto future = driver->Write(controller.GetKey(),
                              std::move(read_result).optional_value(),
                              std::move(write_options));
  future.Force();
  std::move(future).ExecuteWhenReady(
      [controller](ReadyFuture<TimestampedStorageGeneration> future) mutable {
        auto& r = future.result();
        if (!r.ok()) {
          ReportWritebackError(controller, "writing", r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          controller.Retry(r->time);
        } else {
          controller.Success(std::move(*r));
        }
      });
}
void StartWriteback(ReadModifyWriteEntry& entry,
                    absl::Time staleness_bound = absl::InfinitePast());
void DeletedEntryDone(DeleteRangeEntry& dr_entry, bool error, size_t count = 1);
void EntryDone(SinglePhaseMutation& single_phase_mutation, bool error,
               size_t count = 1);
[[maybe_unused]] void CheckInvariants(ReadModifyWriteEntry* entry) {
  do {
    assert(!(entry->flags_ & ReadModifyWriteEntry::kDeleted));
    if (entry->prev_) {
      assert(entry->prev_->single_phase_mutation().phase_number_ <=
             entry->single_phase_mutation().phase_number_);
      assert(entry->prev_->key_ == entry->key_);
    }
    entry = entry->prev_;
  } while (entry);
}
[[maybe_unused]] void CheckInvariants(MultiPhaseMutation& multi_phase,
                                      bool commit_started) {
  absl::btree_map<size_t, size_t> phase_entry_count;
  for (auto* single_phase_mutation = &multi_phase.phases_;;) {
    if (single_phase_mutation != &multi_phase.phases_) {
      assert(single_phase_mutation->phase_number_ >
             single_phase_mutation->prev_->phase_number_);
    }
    for (MutationEntry *
             tree_entry = single_phase_mutation->entries_.begin().to_pointer(),
            *tree_next;
         tree_entry; tree_entry = tree_next) {
      ++phase_entry_count[tree_entry->single_phase_mutation().phase_number_];
      if (commit_started) {
        assert(&tree_entry->single_phase_mutation() == single_phase_mutation);
      } else {
        assert(&tree_entry->single_phase_mutation() == single_phase_mutation ||
               single_phase_mutation == multi_phase.phases_.prev_);
      }
      tree_next =
          MutationEntryTree::Traverse(*tree_entry, MutationEntryTree::kRight);
      if (tree_next) {
        assert(tree_next->key_ > tree_entry->key_);
        if (tree_entry->entry_type() != kReadModifyWrite) {
          [[maybe_unused]] auto* dr_entry =
              static_cast<DeleteRangeEntry*>(tree_entry);
          assert(KeyRange::CompareExclusiveMaxAndKey(dr_entry->exclusive_max_,
                                                     tree_next->key_) <= 0);
        }
      }
      if (tree_entry->entry_type() == kReadModifyWrite) {
        auto* rmw_entry = static_cast<ReadModifyWriteEntry*>(tree_entry);
        CheckInvariants(rmw_entry);
      } else {
        auto* dr_entry = static_cast<DeleteRangeEntry*>(tree_entry);
        if (dr_entry->entry_type() == kDeleteRangePlaceholder) {
          --phase_entry_count[tree_entry->single_phase_mutation()
                                  .phase_number_];
          assert(dr_entry->superseded_.empty());
        }
        assert(KeyRange::CompareKeyAndExclusiveMax(
                   dr_entry->key_, dr_entry->exclusive_max_) < 0);
        for (ReadModifyWriteEntryTree::iterator
                 entry = dr_entry->superseded_.begin(),
                 next;
             entry != dr_entry->superseded_.end(); entry = next) {
          next = std::next(entry);
          if (next) {
            assert(next->key_ > entry->key_);
          }
          assert(entry->entry_type() == kReadModifyWrite);
          assert(&entry->single_phase_mutation() ==
                 &dr_entry->single_phase_mutation());
          assert(entry->key_ >= dr_entry->key_);
          assert(KeyRange::CompareKeyAndExclusiveMax(
                     entry->key_, dr_entry->exclusive_max_) < 0);
          assert(entry->flags_ & ReadModifyWriteEntry::kDeleted);
          if (entry->prev_) {
            CheckInvariants(entry->prev_);
          }
        }
      }
    }
    single_phase_mutation = single_phase_mutation->next_;
    if (single_phase_mutation == &multi_phase.phases_) break;
  }
  for (auto* single_phase_mutation = &multi_phase.phases_;
       single_phase_mutation->next_ != &multi_phase.phases_;
       single_phase_mutation = single_phase_mutation->next_) {
    if (single_phase_mutation->phase_number_ <
        multi_phase.GetTransactionNode().phase()) {
      assert(single_phase_mutation->entries_.empty());
    }
  }
}
#ifdef TENSORSTORE_INTERNAL_KVSTORETORE_TRANSACTION_DEBUG
inline void DebugCheckInvariants(MultiPhaseMutation& multi_phase,
                                 bool commit_started) {
  CheckInvariants(multi_phase, commit_started);
}
class DebugCheckInvariantsInDestructor {
 public:
  explicit DebugCheckInvariantsInDestructor(MultiPhaseMutation& multi_phase,
                                            bool commit_started)
      : multi_phase_(multi_phase), commit_started_(commit_started) {}
  ~DebugCheckInvariantsInDestructor() {
    CheckInvariants(multi_phase_, commit_started_);
  }
 private:
  MultiPhaseMutation& multi_phase_;
  bool commit_started_;
};
#else
inline void DebugCheckInvariants(MultiPhaseMutation& multi_phase,
                                 bool commit_started) {}
class DebugCheckInvariantsInDestructor {
 public:
  explicit DebugCheckInvariantsInDestructor(MultiPhaseMutation& multi_phase,
                                            bool commit_started) {}
};
#endif
void DestroyReadModifyWriteSequence(ReadModifyWriteEntry* entry) {
  if (auto* next_rmw = entry->next_read_modify_write()) {
    next_rmw->prev_ = nullptr;
  }
  auto& multi_phase = entry->multi_phase();
  while (true) {
    auto* prev = entry->prev_;
    multi_phase.FreeReadModifyWriteEntry(entry);
    if (!prev) break;
    entry = prev;
  }
}
auto CompareToEntry(MutationEntry& e) {
  return [&e](MutationEntry& other) {
    return internal::CompareResultAsWeakOrdering(e.key_.compare(other.key_));
  };
}
void InsertIntoPriorPhase(MutationEntry* entry) {
  if (entry->entry_type() == kDeleteRangePlaceholder) {
    delete static_cast<DeleteRangeEntry*>(entry);
    return;
  }
  entry->single_phase_mutation().entries_.FindOrInsert(
      CompareToEntry(*entry), [entry] { return entry; });
}
DeleteRangeEntry* MakeDeleteRangeEntry(
    MutationEntryType entry_type,
    SinglePhaseMutation& assigned_single_phase_mutation, KeyRange&& range) {
  auto* entry = new DeleteRangeEntry;
  entry->key_ = std::move(range.inclusive_min);
  entry->exclusive_max_ = std::move(range.exclusive_max);
  entry->single_phase_mutation_ = {&assigned_single_phase_mutation,
                                   static_cast<uintptr_t>(entry_type)};
  return entry;
}
DeleteRangeEntry* InsertDeleteRangeEntry(
    MutationEntryType entry_type,
    SinglePhaseMutation& insert_single_phase_mutation,
    SinglePhaseMutation& assigned_single_phase_mutation, KeyRange&& range,
    MutationEntryTree::InsertPosition position) {
  assert(entry_type == kDeleteRange || entry_type == kDeleteRangePlaceholder);
  auto* entry = MakeDeleteRangeEntry(entry_type, assigned_single_phase_mutation,
                                     std::move(range));
  insert_single_phase_mutation.entries_.Insert(position, *entry);
  return entry;
}
ReadModifyWriteEntry* MakeReadModifyWriteEntry(
    SinglePhaseMutation& assigned_single_phase_mutation, std::string&& key) {
  auto* entry = assigned_single_phase_mutation.multi_phase_
                    ->AllocateReadModifyWriteEntry();
  entry->key_ = std::move(key);
  entry->single_phase_mutation_ = {&assigned_single_phase_mutation, 0};
  return entry;
}
SinglePhaseMutation& GetCurrentSinglePhaseMutation(
    MultiPhaseMutation& multi_phase) {
  size_t phase = multi_phase.GetTransactionNode().transaction()->phase();
  SinglePhaseMutation* single_phase_mutation;
  if (multi_phase.phases_.phase_number_ ==
      internal::TransactionState::kInvalidPhase) {
    single_phase_mutation = &multi_phase.phases_;
    single_phase_mutation->phase_number_ = phase;
  } else {
    single_phase_mutation = multi_phase.phases_.prev_;
    assert(single_phase_mutation->phase_number_ <= phase);
    if (single_phase_mutation->phase_number_ != phase) {
      auto* new_single_phase_mutation = new SinglePhaseMutation;
      std::swap(new_single_phase_mutation->entries_,
                single_phase_mutation->entries_);
      new_single_phase_mutation->next_ = &multi_phase.phases_;
      new_single_phase_mutation->prev_ = single_phase_mutation;
      new_single_phase_mutation->phase_number_ = phase;
      new_single_phase_mutation->prev_->next_ = new_single_phase_mutation;
      new_single_phase_mutation->next_->prev_ = new_single_phase_mutation;
      new_single_phase_mutation->multi_phase_ = &multi_phase;
      single_phase_mutation = new_single_phase_mutation;
    }
  }
  return *single_phase_mutation;
}
struct Controller {
  ReadModifyWriteEntry* entry_;
  internal::TransactionState::Node& GetTransactionNode() {
    return entry_->multi_phase().GetTransactionNode();
  }
  std::string DescribeKey(std::string_view key) {
    return entry_->multi_phase().DescribeKey(key);
  }
  const Key& GetKey() { return entry_->key_; }
  void Success(TimestampedStorageGeneration new_stamp) {
    if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry_->next_)) {
      DeletedEntryDone(*dr_entry, false);
      return;
    }
    WritebackSuccess(*entry_, std::move(new_stamp));
    EntryDone(entry_->single_phase_mutation(), false);
  }
  void Error(absl::Status error) {
    auto* dr_entry = static_cast<DeleteRangeEntry*>(entry_->next_);
    auto& single_phase_mutation = entry_->single_phase_mutation();
    entry_->multi_phase().RecordEntryWritebackError(*entry_, std::move(error));
    if (dr_entry) {
      DeletedEntryDone(*dr_entry, true);
    } else {
      EntryDone(single_phase_mutation, true);
    }
  }
  void Retry(absl::Time time) {
    kvstore_transaction_retries.Increment();
    StartWriteback(*entry_, time);
  }
};
void ReceiveWritebackCommon(ReadModifyWriteEntry& entry,
                            ReadResult& read_result) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "ReceiveWritebackCommon: state=", read_result.state,
      ", stamp=", read_result.stamp);
  auto flags =
      (entry.flags_ & ~(ReadModifyWriteEntry::kTransitivelyUnconditional |
                        ReadModifyWriteEntry::kDirty |
                        ReadModifyWriteEntry::kTransitivelyDirty)) |
      ReadModifyWriteEntry::kWritebackProvided;
  if (!StorageGeneration::IsConditional(read_result.stamp.generation)) {
    flags |= ReadModifyWriteEntry::kTransitivelyUnconditional;
  }
  if (read_result.stamp.generation.ClearNewlyDirty()) {
    flags |= ReadModifyWriteEntry::kDirty;
  }
  if (read_result.state != ReadResult::kUnspecified) {
    flags |= ReadModifyWriteEntry::kTransitivelyDirty;
  }
  entry.flags_ = flags;
}
void StartWriteback(ReadModifyWriteEntry& entry, absl::Time staleness_bound) {
  TENSORSTORE_KVSTORE_DEBUG_LOG(
      entry, "StartWriteback: staleness_bound=", staleness_bound);
  for (auto* e = &entry;;) {
    e->flags_ &= ~(ReadModifyWriteEntry::kWritebackProvided |
                   ReadModifyWriteEntry::kTransitivelyDirty);
    e = e->prev_;
    if (!e) break;
  }
  ReadModifyWriteSource::WritebackOptions writeback_options;
  writeback_options.staleness_bound = staleness_bound;
  writeback_options.writeback_mode =
      (entry.flags_ & ReadModifyWriteEntry::kDeleted)
          ? ReadModifyWriteSource::kValidateOnly
          : ReadModifyWriteSource::kNormalWriteback;
  if (!entry.prev_ && !(entry.flags_ & ReadModifyWriteEntry::kDeleted)) {
    struct WritebackReceiverImpl {
      ReadModifyWriteEntry* entry_;
      void set_error(absl::Status error) {
        ReportWritebackError(Controller{entry_}, "writing", error);
      }
      void set_cancel() { ABSL_UNREACHABLE(); }  
      void set_value(ReadResult read_result) {
        ReceiveWritebackCommon(*entry_, read_result);
        entry_->multi_phase().Writeback(*entry_, *entry_,
                                        std::move(read_result));
      }
    };
    entry.source_->KvsWriteback(std::move(writeback_options),
                                WritebackReceiverImpl{&entry});
    return;
  }
  struct SequenceWritebackReceiverImpl {
    struct State {
      ReadModifyWriteEntry* entry;
      absl::Time staleness_bound;
      ReadResult read_result;
      ReadModifyWriteEntry* source_entry = nullptr;
      ReadModifyWriteEntry* GetLastReadModifyWriteEntry() {
        auto* e = entry;
        while (auto* next = e->next_read_modify_write()) e = next;
        return e;
      }
    };
    std::unique_ptr<State> state_;
    void set_error(absl::Status error) {
      ReportWritebackError(Controller{state_->GetLastReadModifyWriteEntry()},
                           "writing", error);
    }
    void set_cancel() { ABSL_UNREACHABLE(); }  
    void set_value(ReadResult read_result) {
      auto& entry = *state_->entry;
      ReceiveWritebackCommon(entry, read_result);
      if (!state_->entry->next_ &&
          !(state_->entry->flags_ & ReadModifyWriteEntry::kDeleted)) {
        state_->read_result = std::move(read_result);
        state_->source_entry = &entry;
      } else {
        assert(!StorageGeneration::IsConditional(
            state_->read_result.stamp.generation));
        if (state_->read_result.state == ReadResult::kUnspecified) {
          TENSORSTORE_KVSTORE_DEBUG_LOG(
              entry,
              "Replacing: existing_result state=", state_->read_result.state,
              ", stamp=", state_->read_result.stamp,
              ", new_result state=", read_result.state,
              ", stamp=", read_result.stamp);
          state_->read_result = std::move(read_result);
          state_->source_entry = &entry;
        } else {
          state_->read_result.stamp.time = read_result.stamp.time;
          TENSORSTORE_KVSTORE_DEBUG_LOG(entry, "Conditioning: existing_stamp=",
                                        state_->read_result.stamp.generation,
                                        ", new_stamp=", read_result.stamp);
          state_->read_result.stamp.generation = StorageGeneration::Condition(
              state_->read_result.stamp.generation,
              std::move(read_result.stamp.generation));
        }
      }
      if (entry.flags_ & ReadModifyWriteEntry::kTransitivelyUnconditional) {
        const bool unmodified =
            state_->read_result.state == ReadResult::kUnspecified;
        auto GetPrevSupersededEntryToWriteback =
            [&](ReadModifyWriteEntry* entry) -> ReadModifyWriteEntry* {
          while (true) {
            entry = entry->prev_;
            if (!entry) return nullptr;
            if (unmodified) {
              if (!(entry->flags_ & ReadModifyWriteEntry::kWritebackProvided) ||
                  (entry->flags_ & ReadModifyWriteEntry::kTransitivelyDirty)) {
                return entry;
              }
            } else {
              if (!(entry->flags_ &
                    (ReadModifyWriteEntry::kWritebackProvided |
                     ReadModifyWriteEntry::kTransitivelyUnconditional))) {
                return entry;
              }
            }
          }
        };
        if (auto* prev = GetPrevSupersededEntryToWriteback(&entry)) {
          state_->entry = prev;
          TENSORSTORE_KVSTORE_DEBUG_LOG(*prev,
                                        "Continuing writeback validate only");
          ReadModifyWriteSource::WritebackOptions writeback_options;
          writeback_options.staleness_bound = state_->staleness_bound;
          writeback_options.writeback_mode =
              unmodified ? ReadModifyWriteSource::kNormalWriteback
                         : ReadModifyWriteSource::kValidateOnly;
          prev->source_->KvsWriteback(std::move(writeback_options),
                                      std::move(*this));
          return;
        }
      }
      auto* last_entry = state_->GetLastReadModifyWriteEntry();
      if (last_entry->next_) {
        state_->read_result.state = ReadResult::kUnspecified;
      }
      TENSORSTORE_KVSTORE_DEBUG_LOG(*last_entry,
                                    "No remaining skipped entries, forwarding "
                                    "to MultiPhaseMutation::Writeback: ",
                                    state_->read_result.stamp);
      last_entry->multi_phase().Writeback(
          *last_entry,
          state_->source_entry ? *state_->source_entry : *last_entry,
          std::move(state_->read_result));
    }
  };
  auto state = std::unique_ptr<SequenceWritebackReceiverImpl::State>(
      new SequenceWritebackReceiverImpl::State{&entry, staleness_bound});
  if (entry.flags_ & ReadModifyWriteEntry::kDeleted) {
    state->read_result.state = ReadResult::kMissing;
  }
  entry.source_->KvsWriteback(std::move(writeback_options),
                              SequenceWritebackReceiverImpl{std::move(state)});
}
void HandleDeleteRangeDone(DeleteRangeEntry& dr_entry) {
  const bool error = dr_entry.remaining_entries_.HasError();
  if (error) {
    WritebackError(dr_entry);
  } else {
    WritebackSuccess(dr_entry);
  }
  EntryDone(dr_entry.single_phase_mutation(), error);
}
void DeletedEntryDone(DeleteRangeEntry& dr_entry, bool error, size_t count) {
  if (error) dr_entry.remaining_entries_.SetError();
  if (!dr_entry.remaining_entries_.DecrementCount(count)) return;
  if (dr_entry.remaining_entries_.HasError()) {
    HandleDeleteRangeDone(dr_entry);
    return;
  }
  dr_entry.multi_phase().WritebackDelete(dr_entry);
}
std::string DescribeEntry(MutationEntry& entry) {
  return tensorstore::StrCat(
      entry.entry_type() == kReadModifyWrite ? "read/write " : "delete ",
      entry.multi_phase().DescribeKey(entry.key_));
}
void EntryDone(SinglePhaseMutation& single_phase_mutation, bool error,
               size_t count) {
  auto& multi_phase = *single_phase_mutation.multi_phase_;
  if (error) single_phase_mutation.remaining_entries_.SetError();
  if (!single_phase_mutation.remaining_entries_.DecrementCount(count)) {
    return;
  }
  multi_phase.AllEntriesDone(single_phase_mutation);
}
}  
void ReadModifyWriteEntry::KvsRead(
    ReadModifyWriteTarget::TransactionalReadOptions options,
    ReadModifyWriteTarget::ReadReceiver receiver) {
  struct ReadReceiverImpl {
    ReadModifyWriteEntry* entry_;
    ReadModifyWriteTarget::ReadReceiver receiver_;
    void set_cancel() { execution::set_cancel(receiver_); }
    void set_value(ReadResult read_result) {
      {
        assert(!StorageGeneration::IsUnknown(read_result.stamp.generation));
        absl::MutexLock lock(&entry_->mutex());
        ReceiveWritebackCommon(*entry_->prev_, read_result);
        entry_->flags_ |= (entry_->prev_->flags_ &
                           ReadModifyWriteEntry::kTransitivelyUnconditional);
      }
      execution::set_value(receiver_, std::move(read_result));
    }
    void set_error(absl::Status error) {
      execution::set_error(receiver_, std::move(error));
    }
  };
  if (flags_ & ReadModifyWriteEntry::kPrevDeleted) {
    execution::set_value(
        receiver, ReadResult::Missing(
                      {StorageGeneration::Dirty(StorageGeneration::Unknown()),
                       absl::InfiniteFuture()}));
  } else if (prev_) {
    TENSORSTORE_KVSTORE_DEBUG_LOG(*prev_, "Requesting writeback for read");
    ReadModifyWriteSource::WritebackOptions writeback_options;
    writeback_options.generation_conditions.if_not_equal =
        std::move(options.generation_conditions.if_not_equal);
    writeback_options.staleness_bound = options.staleness_bound;
    writeback_options.writeback_mode =
        ReadModifyWriteSource::kSpecifyUnchangedWriteback;
    this->prev_->source_->KvsWriteback(
        std::move(writeback_options),
        ReadReceiverImpl{this, std::move(receiver)});
  } else {
    multi_phase().Read(*this, std::move(options), std::move(receiver));
  }
}
bool ReadModifyWriteEntry::KvsReadsCommitted() {
  return prev_ == nullptr && !(flags_ & ReadModifyWriteEntry::kPrevDeleted) &&
         multi_phase().MultiPhaseReadsCommitted();
}
void DestroyPhaseEntries(SinglePhaseMutation& single_phase_mutation) {
  auto& multi_phase = *single_phase_mutation.multi_phase_;
  for (MutationEntryTree::iterator
           tree_entry = single_phase_mutation.entries_.begin(),
           tree_next;
       tree_entry != single_phase_mutation.entries_.end();
       tree_entry = tree_next) {
    tree_next = std::next(tree_entry);
    single_phase_mutation.entries_.Remove(*tree_entry);
    if (tree_entry->entry_type() == kReadModifyWrite) {
      DestroyReadModifyWriteSequence(
          static_cast<ReadModifyWriteEntry*>(&*tree_entry));
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(*tree_entry);
      for (ReadModifyWriteEntryTree::iterator
               entry = dr_entry.superseded_.begin(),
               next;
           entry != dr_entry.superseded_.end(); entry = next) {
        next = std::next(entry);
        dr_entry.superseded_.Remove(*entry);
        DestroyReadModifyWriteSequence(entry.to_pointer());
      }
      delete &dr_entry;
    }
  }
  if (&single_phase_mutation != &multi_phase.phases_) {
    single_phase_mutation.prev_->next_ = single_phase_mutation.next_;
    single_phase_mutation.next_->prev_ = single_phase_mutation.prev_;
    delete &single_phase_mutation;
  }
}
namespace {
void InvalidateReadStateGoingBackward(ReadModifyWriteEntry* entry) {
  do {
    entry->source_->KvsInvalidateReadState();
    entry = entry->prev_;
  } while (entry);
}
}  
void InvalidateReadState(SinglePhaseMutation& single_phase_mutation) {
  for (auto& entry : single_phase_mutation.entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      InvalidateReadStateGoingBackward(
          static_cast<ReadModifyWriteEntry*>(&entry));
    } else {
      for (auto& deleted_entry :
           static_cast<DeleteRangeEntry&>(entry).superseded_) {
        InvalidateReadStateGoingBackward(&deleted_entry);
      }
    }
  }
}
void WritebackSuccess(ReadModifyWriteEntry& entry,
                      TimestampedStorageGeneration new_stamp) {
  assert(!entry.next_read_modify_write());
  for (ReadModifyWriteEntry* e = &entry;;) {
    e->source_->KvsWritebackSuccess(new_stamp);
    bool dirty = static_cast<bool>(e->flags_ & ReadModifyWriteEntry::kDirty);
    e = e->prev_;
    if (!e) break;
    if (dirty || !(e->flags_ & ReadModifyWriteEntry::kWritebackProvided)) {
      new_stamp.generation = StorageGeneration::Unknown();
      new_stamp.time = absl::InfiniteFuture();
    }
  }
}
void WritebackError(ReadModifyWriteEntry& entry) {
  assert(!entry.next_read_modify_write());
  if (entry.flags_ & ReadModifyWriteEntry::kError) return;
  entry.flags_ |= ReadModifyWriteEntry::kError;
  for (ReadModifyWriteEntry* e = &entry;;) {
    e->source_->KvsWritebackError();
    e = e->prev_;
    if (!e) break;
  }
}
void WritebackError(DeleteRangeEntry& entry) {
  for (auto& e : entry.superseded_) {
    WritebackError(e);
  }
}
void WritebackSuccess(DeleteRangeEntry& entry) {
  for (auto& e : entry.superseded_) {
    WritebackSuccess(e,
                     TimestampedStorageGeneration{StorageGeneration::Unknown(),
                                                  absl::InfiniteFuture()});
  }
}
void WritebackError(MutationEntry& entry) {
  if (entry.entry_type() == kReadModifyWrite) {
    WritebackError(static_cast<ReadModifyWriteEntry&>(entry));
  } else {
    WritebackError(static_cast<DeleteRangeEntry&>(entry));
  }
}
void WritebackError(SinglePhaseMutation& single_phase_mutation) {
  for (auto& entry : single_phase_mutation.entries_) {
    WritebackError(entry);
  }
}
MultiPhaseMutation::MultiPhaseMutation() {
  phases_.next_ = phases_.prev_ = &phases_;
  phases_.phase_number_ = internal::TransactionState::kInvalidPhase;
  phases_.multi_phase_ = this;
}
SinglePhaseMutation& MultiPhaseMutation::GetCommittingPhase() {
  auto* single_phase_mutation = &phases_;
  auto initial_phase_number = single_phase_mutation->phase_number_;
  if (initial_phase_number != this->GetTransactionNode().phase() &&
      initial_phase_number != internal::TransactionState::kInvalidPhase) {
    single_phase_mutation = single_phase_mutation->next_;
    assert(single_phase_mutation->phase_number_ ==
           this->GetTransactionNode().phase());
  }
  return *single_phase_mutation;
}
void MultiPhaseMutation::AllEntriesDone(
    SinglePhaseMutation& single_phase_mutation) {
  size_t next_phase = 0;
  if (single_phase_mutation.next_ != &this->phases_) {
    next_phase = single_phase_mutation.next_->phase_number_;
  }
  DestroyPhaseEntries(single_phase_mutation);
  this->PhaseCommitDone(next_phase);
}
namespace {
void InvalidateReadStateGoingForward(ReadModifyWriteEntry* entry) {
  auto& single_phase_mutation = entry->single_phase_mutation();
  do {
    entry->source_->KvsInvalidateReadState();
    entry->flags_ &= ~ReadModifyWriteEntry::kTransitivelyUnconditional;
    entry = entry->next_read_modify_write();
  } while (entry &&
           (&entry->single_phase_mutation() == &single_phase_mutation));
}
void WritebackPhase(
    SinglePhaseMutation& single_phase_mutation, absl::Time staleness_bound,
    absl::FunctionRef<bool(ReadModifyWriteEntry& entry)> predicate) {
  assert(single_phase_mutation.remaining_entries_.IsDone());
  size_t entry_count = 0;
  for (auto& entry : single_phase_mutation.entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(entry);
      if (auto* next = static_cast<ReadModifyWriteEntry*>(rmw_entry.next_)) {
        assert(next->entry_type() == kReadModifyWrite);
        assert(&next->single_phase_mutation() != &single_phase_mutation);
        next->prev_ = nullptr;
        InvalidateReadStateGoingForward(next);
        rmw_entry.next_ = nullptr;
      }
      if (predicate(rmw_entry)) {
        ++entry_count;
        StartWriteback(rmw_entry, staleness_bound);
      }
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      assert(dr_entry.remaining_entries_.IsDone());
      ++entry_count;
      size_t deleted_entry_count = 0;
      for (auto& deleted_entry : dr_entry.superseded_) {
        auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(deleted_entry);
        rmw_entry.next_ = &dr_entry;
        if (predicate(rmw_entry)) {
          ++deleted_entry_count;
          StartWriteback(static_cast<ReadModifyWriteEntry&>(deleted_entry),
                         staleness_bound);
        }
      }
      DeletedEntryDone(dr_entry, false, -deleted_entry_count);
    }
  }
  EntryDone(single_phase_mutation, false, -entry_count);
}
}  
void MultiPhaseMutation::CommitNextPhase() {
  size_t cur_phase_number = GetTransactionNode().phase();
  DebugCheckInvariants(*this, false);
  {
    DebugCheckInvariantsInDestructor debug_check(*this, true);
    if (cur_phase_number == 0) {
      if (phases_.next_ != &phases_) {
        auto* last_phase = phases_.prev_;
        for (MutationEntryTree::iterator entry = last_phase->entries_.begin(),
                                         next;
             entry != last_phase->entries_.end(); entry = next) {
          next = std::next(entry);
          if (&entry->single_phase_mutation() != last_phase) {
            last_phase->entries_.Remove(*entry);
            InsertIntoPriorPhase(entry.to_pointer());
          }
        }
      }
      if (cur_phase_number != phases_.phase_number_) {
        this->PhaseCommitDone(phases_.phase_number_);
        return;
      }
    }
  }
  auto& single_phase_mutation = GetCommittingPhase();
  WritebackPhase(single_phase_mutation, absl::InfinitePast(),
                 [](ReadModifyWriteEntry& entry) { return true; });
}
void MultiPhaseMutation::AbortRemainingPhases() {
  for (auto* single_phase_mutation = &phases_;;) {
    auto* next = single_phase_mutation->next_;
    DestroyPhaseEntries(*single_phase_mutation);
    if (next == &phases_) break;
    single_phase_mutation = next;
  }
}
MultiPhaseMutation::ReadModifyWriteStatus MultiPhaseMutation::ReadModifyWrite(
    size_t& phase, Key key, ReadModifyWriteSource& source) {
  DebugCheckInvariantsInDestructor debug_check(*this, false);
#ifndef NDEBUG
  mutex().AssertHeld();
#endif
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);
  phase = single_phase_mutation.phase_number_;
  auto* entry = MakeReadModifyWriteEntry(single_phase_mutation, std::move(key));
  entry->source_ = &source;
  entry->source_->KvsSetTarget(*entry);
  auto find_result = single_phase_mutation.entries_.Find(
      [key = std::string_view(entry->key_)](MutationEntry& existing_entry) {
        auto c = key.compare(existing_entry.key_);
        if (c <= 0) return internal::CompareResultAsWeakOrdering(c);
        if (existing_entry.entry_type() == kReadModifyWrite) {
          return absl::weak_ordering::greater;
        }
        return KeyRange::CompareKeyAndExclusiveMax(
                   key, static_cast<DeleteRangeEntry&>(existing_entry)
                            .exclusive_max_) < 0
                   ? absl::weak_ordering::equivalent
                   : absl::weak_ordering::greater;
      });
  if (!find_result.found) {
    const bool was_empty = single_phase_mutation.entries_.empty();
    single_phase_mutation.entries_.Insert(find_result.insert_position(),
                                          *entry);
    return was_empty ? ReadModifyWriteStatus::kAddedFirst
                     : ReadModifyWriteStatus::kAddedSubsequent;
  }
  single_phase_mutation.entries_.Replace(*find_result.node, *entry);
  if (find_result.node->entry_type() == kReadModifyWrite) {
    auto* existing_entry = static_cast<ReadModifyWriteEntry*>(find_result.node);
    assert(existing_entry->key_ == entry->key_);
    if (&existing_entry->single_phase_mutation() != &single_phase_mutation) {
      InsertIntoPriorPhase(existing_entry);
    }
    existing_entry->source_->KvsRevoke();
    assert(existing_entry->next_ == nullptr);
    entry->prev_ = existing_entry;
    existing_entry->next_ = entry;
    return ReadModifyWriteStatus::kExisting;
  }
  auto* existing_entry = static_cast<DeleteRangeEntry*>(find_result.node);
  assert(existing_entry->key_ <= entry->key_);
  assert(KeyRange::CompareKeyAndExclusiveMax(
             entry->key_, existing_entry->exclusive_max_) < 0);
  entry->flags_ |= (ReadModifyWriteEntry::kPrevDeleted |
                    ReadModifyWriteEntry::kTransitivelyUnconditional);
  if (&existing_entry->single_phase_mutation() != &single_phase_mutation) {
    if (existing_entry->key_ != entry->key_) {
      InsertDeleteRangeEntry(kDeleteRangePlaceholder, single_phase_mutation,
                             existing_entry->single_phase_mutation(),
                             KeyRange{existing_entry->key_, entry->key_},
                             {entry, MutationEntryTree::kLeft});
    }
    if (auto successor = KeyRange::Successor(entry->key_);
        successor != existing_entry->exclusive_max_) {
      InsertDeleteRangeEntry(
          kDeleteRangePlaceholder, single_phase_mutation,
          existing_entry->single_phase_mutation(),
          KeyRange{std::move(successor), existing_entry->exclusive_max_},
          {entry, MutationEntryTree::kRight});
    }
    InsertIntoPriorPhase(existing_entry);
    return ReadModifyWriteStatus::kExisting;
  }
  auto split_result = existing_entry->superseded_.FindSplit(
      [key = std::string_view(entry->key_)](MutationEntry& e) {
        return internal::CompareResultAsWeakOrdering(key.compare(e.key_));
      });
  if (split_result.center) {
    split_result.center->flags_ &= ~ReadModifyWriteEntry::kDeleted;
    entry->prev_ = split_result.center;
    split_result.center->next_ = entry;
  }
  if (existing_entry->key_ != entry->key_) {
    auto* dr_entry = InsertDeleteRangeEntry(
        kDeleteRange, single_phase_mutation,
        existing_entry->single_phase_mutation(),
        KeyRange{std::move(existing_entry->key_), entry->key_},
        {entry, MutationEntryTree::kLeft});
    dr_entry->superseded_ = std::move(split_result.trees[0]);
  } else {
    assert(split_result.trees[0].empty());
  }
  existing_entry->key_ = KeyRange::Successor(entry->key_);
  if (existing_entry->key_ != existing_entry->exclusive_max_) {
    single_phase_mutation.entries_.Insert({entry, MutationEntryTree::kRight},
                                          *existing_entry);
    existing_entry->superseded_ = std::move(split_result.trees[1]);
  } else {
    assert(split_result.trees[1].empty());
    delete existing_entry;
  }
  return ReadModifyWriteStatus::kExisting;
}
void MultiPhaseMutation::DeleteRange(KeyRange range) {
#ifndef NDEBUG
  mutex().AssertHeld();
#endif
  if (range.empty()) return;
  DebugCheckInvariantsInDestructor debug_check(*this, false);
  auto& single_phase_mutation = GetCurrentSinglePhaseMutation(*this);
  auto find_result =
      single_phase_mutation.entries_.FindBound<MutationEntryTree::kLeft>(
          [&](MutationEntry& existing_entry) {
            if (existing_entry.entry_type() == kReadModifyWrite) {
              return existing_entry.key_ < range.inclusive_min;
            } else {
              return KeyRange::CompareExclusiveMaxAndKey(
                         static_cast<DeleteRangeEntry&>(existing_entry)
                             .exclusive_max_,
                         range.inclusive_min) <= 0;
            }
          });
  DeleteRangeEntry* new_entry = nullptr;
  ReadModifyWriteEntryTree superseded;
  DeleteRangeEntry insert_placeholder;
  single_phase_mutation.entries_.Insert(find_result.insert_position(),
                                        insert_placeholder);
  for (MutationEntry *existing_entry = find_result.found_node(), *next;
       existing_entry; existing_entry = next) {
    if (KeyRange::CompareKeyAndExclusiveMax(existing_entry->key_,
                                            range.exclusive_max) >= 0) {
      break;
    }
    next =
        MutationEntryTree::Traverse(*existing_entry, MutationEntryTree::kRight);
    single_phase_mutation.entries_.Remove(*existing_entry);
    if (existing_entry->entry_type() == kReadModifyWrite) {
      auto* existing_rmw_entry =
          static_cast<ReadModifyWriteEntry*>(existing_entry);
      existing_rmw_entry->source_->KvsRevoke();
      if (&existing_rmw_entry->single_phase_mutation() !=
          &single_phase_mutation) {
        InsertIntoPriorPhase(existing_entry);
      } else {
        existing_rmw_entry->flags_ |= ReadModifyWriteEntry::kDeleted;
        [[maybe_unused]] bool inserted =
            superseded
                .FindOrInsert(CompareToEntry(*existing_rmw_entry),
                              [=] { return existing_rmw_entry; })
                .second;
        assert(inserted);
      }
    } else {
      auto* existing_dr_entry = static_cast<DeleteRangeEntry*>(existing_entry);
      if (&existing_dr_entry->single_phase_mutation() !=
          &single_phase_mutation) {
        if (KeyRange::CompareExclusiveMax(
                range.exclusive_max, existing_dr_entry->exclusive_max_) < 0) {
          InsertDeleteRangeEntry(
              kDeleteRangePlaceholder, single_phase_mutation,
              existing_dr_entry->single_phase_mutation(),
              KeyRange{range.exclusive_max, existing_dr_entry->exclusive_max_},
              {&insert_placeholder, MutationEntryTree::kRight});
        }
        if (existing_dr_entry->key_ < range.inclusive_min) {
          InsertDeleteRangeEntry(
              kDeleteRangePlaceholder, single_phase_mutation,
              existing_dr_entry->single_phase_mutation(),
              KeyRange{existing_dr_entry->key_, range.inclusive_min},
              {&insert_placeholder, MutationEntryTree::kLeft});
        }
        InsertIntoPriorPhase(existing_dr_entry);
      } else {
        superseded = ReadModifyWriteEntryTree::Join(
            superseded, existing_dr_entry->superseded_);
        if (!new_entry) {
          new_entry = existing_dr_entry;
        } else {
          new_entry->exclusive_max_ =
              std::move(existing_dr_entry->exclusive_max_);
          delete existing_dr_entry;
        }
      }
    }
  }
  if (new_entry) {
    if (range.inclusive_min < new_entry->key_) {
      new_entry->key_ = std::move(range.inclusive_min);
    }
    if (KeyRange::CompareExclusiveMax(range.exclusive_max,
                                      new_entry->exclusive_max_) > 0) {
      new_entry->exclusive_max_ = std::move(range.exclusive_max);
    }
  } else {
    new_entry = MakeDeleteRangeEntry(kDeleteRange, single_phase_mutation,
                                     std::move(range));
  }
  new_entry->superseded_ = std::move(superseded);
  single_phase_mutation.entries_.Replace(insert_placeholder, *new_entry);
}
std::string MultiPhaseMutation::DescribeFirstEntry() {
  assert(!phases_.prev_->entries_.empty());
  return DescribeEntry(*phases_.prev_->entries_.begin());
}
ReadModifyWriteEntry* MultiPhaseMutation::AllocateReadModifyWriteEntry() {
  return new ReadModifyWriteEntry;
}
void MultiPhaseMutation::FreeReadModifyWriteEntry(ReadModifyWriteEntry* entry) {
  delete entry;
}
void ReadDirectly(Driver* driver, ReadModifyWriteEntry& entry,
                  ReadModifyWriteTarget::TransactionalReadOptions&& options,
                  ReadModifyWriteTarget::ReadReceiver&& receiver) {
  ReadOptions kvstore_options;
  kvstore_options.staleness_bound = options.staleness_bound;
  kvstore_options.generation_conditions.if_not_equal =
      std::move(options.generation_conditions.if_not_equal);
  kvstore_options.batch = std::move(options.batch);
  execution::submit(driver->Read(entry.key_, std::move(kvstore_options)),
                    std::move(receiver));
}
void WritebackDirectly(Driver* driver, ReadModifyWriteEntry& entry,
                       ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  PerformWriteback(driver, Controller{&entry}, std::move(read_result));
}
void WritebackDirectly(Driver* driver, DeleteRangeEntry& entry) {
  auto future = driver->DeleteRange(KeyRange{entry.key_, entry.exclusive_max_});
  future.Force();
  std::move(future).ExecuteWhenReady([&entry](ReadyFuture<const void> future) {
    auto& r = future.result();
    if (!r.ok()) {
      entry.multi_phase().GetTransactionNode().SetError(r.status());
      entry.remaining_entries_.SetError();
    }
    HandleDeleteRangeDone(entry);
  });
}
void MultiPhaseMutation::RecordEntryWritebackError(ReadModifyWriteEntry& entry,
                                                   absl::Status error) {
  this->GetTransactionNode().SetError(std::move(error));
  WritebackError(entry);
}
void AtomicMultiPhaseMutationBase::RetryAtomicWriteback(
    absl::Time staleness_bound) {
  auto& single_phase_mutation = GetCommittingPhase();
  WritebackPhase(
      single_phase_mutation, staleness_bound, [&](ReadModifyWriteEntry& entry) {
        return static_cast<ReadModifyWriteEntryWithStamp&>(entry).IsOutOfDate(
            staleness_bound);
      });
}
ReadModifyWriteEntry* AtomicMultiPhaseMutation::AllocateReadModifyWriteEntry() {
  return new BufferedReadModifyWriteEntry;
}
void AtomicMultiPhaseMutation::FreeReadModifyWriteEntry(
    ReadModifyWriteEntry* entry) {
  delete static_cast<BufferedReadModifyWriteEntry*>(entry);
}
void AtomicMultiPhaseMutationBase::AtomicWritebackReady(
    ReadModifyWriteEntry& entry) {
  if (auto* dr_entry = static_cast<DeleteRangeEntry*>(entry.next_)) {
    DeletedEntryDone(*dr_entry, false);
  } else {
    EntryDone(entry.single_phase_mutation(), false);
  }
}
void AtomicMultiPhaseMutation::Writeback(ReadModifyWriteEntry& entry,
                                         ReadModifyWriteEntry& source_entry,
                                         ReadResult&& read_result) {
  assert(read_result.stamp.time != absl::InfinitePast());
  auto& buffered = static_cast<BufferedReadModifyWriteEntry&>(entry);
  buffered.stamp() = std::move(read_result.stamp);
  buffered.value_state_ = read_result.state;
  buffered.value_ = std::move(read_result.value);
  AtomicWritebackReady(entry);
}
void AtomicMultiPhaseMutationBase::WritebackDelete(DeleteRangeEntry& entry) {
  EntryDone(entry.single_phase_mutation(), false);
}
void AtomicMultiPhaseMutationBase::AtomicCommitWritebackSuccess() {
  for (auto& entry : GetCommittingPhase().entries_) {
    if (entry.entry_type() == kReadModifyWrite) {
      auto& rmw_entry = static_cast<ReadModifyWriteEntryWithStamp&>(entry);
      internal_kvstore::WritebackSuccess(rmw_entry,
                                         std::move(rmw_entry.stamp_));
    } else {
      auto& dr_entry = static_cast<DeleteRangeEntry&>(entry);
      internal_kvstore::WritebackSuccess(dr_entry);
    }
  }
}
void AtomicMultiPhaseMutationBase::RevokeAllEntries() {
  assert(phases_.next_ == &phases_);
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != kReadModifyWrite) continue;
    auto& rmw_entry = static_cast<ReadModifyWriteEntry&>(entry);
    rmw_entry.source_->KvsRevoke();
  }
}
namespace {
absl::Status GetNonAtomicReadModifyWriteError(
    NonAtomicTransactionNode& node,
    MultiPhaseMutation::ReadModifyWriteStatus modify_status) {
  if (!node.transaction()->atomic()) {
    return absl::OkStatus();
  }
  using ReadModifyWriteStatus = MultiPhaseMutation::ReadModifyWriteStatus;
  if (modify_status == ReadModifyWriteStatus::kAddedFirst) {
    return node.MarkAsTerminal();
  }
  if (modify_status == ReadModifyWriteStatus::kAddedSubsequent) {
    absl::MutexLock lock(&node.mutex_);
    auto& single_phase_mutation = *node.phases_.prev_;
    MutationEntry* e0 = single_phase_mutation.entries_.begin().to_pointer();
    assert(e0);
    MutationEntry* e1 =
        MutationEntryTree::Traverse(*e0, MutationEntryTree::kRight);
    assert(e1);
    auto error = internal::TransactionState::Node::GetAtomicError(
        DescribeEntry(*e0), DescribeEntry(*e1));
    node.transaction()->RequestAbort(error);
    return error;
  }
  return absl::OkStatus();
}
class ReadViaExistingTransactionNode : public internal::TransactionState::Node,
                                       public ReadModifyWriteSource {
 public:
  ReadViaExistingTransactionNode()
      :  
        internal::TransactionState::Node(nullptr) {}
  void PrepareForCommit() override {
    intrusive_ptr_increment(this);
    this->PrepareDone();
    this->ReadyForCommit();
  }
  void Commit() override { intrusive_ptr_decrement(this); }
  void Abort() override { AbortDone(); }
  void KvsSetTarget(ReadModifyWriteTarget& target) override {
    target_ = &target;
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    ReadModifyWriteTarget::TransactionalReadOptions read_options = options;
    if (options.writeback_mode !=
        ReadModifyWriteSource::kSpecifyUnchangedWriteback) {
      TimestampedStorageGeneration expected_stamp;
      {
        absl::MutexLock lock(&mutex_);
        expected_stamp = expected_stamp_;
      }
      if (StorageGeneration::IsUnknown(expected_stamp.generation)) {
        execution::set_value(
            receiver, ReadResult::Unspecified(
                          TimestampedStorageGeneration::Unconditional()));
        return;
      }
      if (StorageGeneration::IsClean(expected_stamp.generation) &&
          expected_stamp.time >= read_options.staleness_bound) {
        execution::set_value(
            receiver, ReadResult::Unspecified(std::move(expected_stamp)));
        return;
      }
    }
    struct ReadReceiverImpl {
      ReadViaExistingTransactionNode& node_;
      ReadModifyWriteSource::WritebackReceiver receiver_;
      void set_value(ReadResult read_result) {
        bool mismatch;
        {
          absl::MutexLock lock(&node_.mutex_);
          mismatch = !StorageGeneration::EqualOrUnspecified(
              read_result.stamp.generation, node_.expected_stamp_.generation);
        }
        if (mismatch) {
          execution::set_error(receiver_,
                               absl::AbortedError("Generation mismatch"));
          return;
        }
        execution::set_value(receiver_, std::move(read_result));
      }
      void set_cancel() { execution::set_cancel(receiver_); }
      void set_error(absl::Status error) {
        execution::set_error(receiver_, std::move(error));
      }
    };
    target_->KvsRead(std::move(read_options),
                     ReadReceiverImpl{*this, std::move(receiver)});
  }
  void KvsWritebackError() override { this->CommitDone(); }
  void KvsRevoke() override {}
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
    this->CommitDone();
  }
  absl::Mutex mutex_;
  TimestampedStorageGeneration expected_stamp_;
  ReadModifyWriteTarget* target_;
};
}  
Future<ReadResult> ReadViaExistingTransaction(
    Driver* driver, internal::OpenTransactionPtr& transaction, size_t& phase,
    Key key, kvstore::TransactionalReadOptions options) {
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  using Node = ReadViaExistingTransactionNode;
  internal::WeakTransactionNodePtr<Node> node;
  node.reset(new Node);
  TENSORSTORE_RETURN_IF_ERROR(
      driver->ReadModifyWrite(transaction, phase, std::move(key), *node));
  node->SetTransaction(*transaction);
  node->SetPhase(phase);
  TENSORSTORE_RETURN_IF_ERROR(node->Register());
  struct InitialReadReceiverImpl {
    internal::OpenTransactionNodePtr<Node> node_;
    Promise<ReadResult> promise_;
    void set_value(ReadResult read_result) {
      if (node_->transaction()->mode() & repeatable_read) {
        absl::MutexLock lock(&node_->mutex_);
        node_->expected_stamp_ = read_result.stamp;
      }
      promise_.SetResult(std::move(read_result));
    }
    void set_cancel() {}
    void set_error(absl::Status error) { promise_.SetResult(std::move(error)); }
  };
  node->target_->KvsRead(std::move(options),
                         InitialReadReceiverImpl{
                             internal::OpenTransactionNodePtr<Node>(node.get()),
                             std::move(promise)});
  return std::move(future);
}
namespace {
class WriteViaExistingTransactionNode : public internal::TransactionState::Node,
                                        public ReadModifyWriteSource {
 public:
  WriteViaExistingTransactionNode()
      :  
        internal::TransactionState::Node(nullptr) {}
  void PrepareForCommit() override {
    intrusive_ptr_increment(this);
    this->PrepareDone();
    this->ReadyForCommit();
  }
  void Commit() override { intrusive_ptr_decrement(this); }
  void Abort() override { AbortDone(); }
  void KvsSetTarget(ReadModifyWriteTarget& target) override {
    target_ = &target;
  }
  void KvsInvalidateReadState() override {}
  void KvsWriteback(
      ReadModifyWriteSource::WritebackOptions options,
      ReadModifyWriteSource::WritebackReceiver receiver) override {
    if (!StorageGeneration::IsConditional(read_result_.stamp.generation)) {
      execution::set_value(receiver, read_result_);
      return;
    }
    ReadModifyWriteTarget::TransactionalReadOptions read_options;
    read_options.generation_conditions.if_not_equal =
        StorageGeneration::Clean(read_result_.stamp.generation);
    read_options.staleness_bound = options.staleness_bound;
    struct ReadReceiverImpl {
      WriteViaExistingTransactionNode& source_;
      ReadModifyWriteSource::WritebackReceiver receiver_;
      void set_value(ReadResult read_result) {
        auto& existing_generation = source_.read_result_.stamp.generation;
        auto clean_generation = StorageGeneration::Clean(existing_generation);
        if (read_result.stamp.generation == clean_generation ||
            (source_.if_equal_no_value_ &&
             read_result.state == ReadResult::kMissing)) {
          source_.read_result_.stamp = std::move(read_result.stamp);
          source_.read_result_.stamp.generation.MarkDirty();
        } else {
          assert(
              !StorageGeneration::IsNewlyDirty(read_result.stamp.generation));
          source_.read_result_ = std::move(read_result);
          source_.if_equal_no_value_ = false;
        }
        execution::set_value(receiver_, source_.read_result_);
      }
      void set_cancel() { execution::set_cancel(receiver_); }
      void set_error(absl::Status error) {
        execution::set_error(receiver_, std::move(error));
      }
    };
    target_->KvsRead(std::move(read_options),
                     ReadReceiverImpl{*this, std::move(receiver)});
  }
  void KvsWritebackError() override { this->CommitDone(); }
  void KvsRevoke() override {}
  void KvsWritebackSuccess(TimestampedStorageGeneration new_stamp) override {
    if (!StorageGeneration::IsNewlyDirty(read_result_.stamp.generation)) {
      new_stamp = TimestampedStorageGeneration{};
    } else if (new_stamp.time == absl::InfiniteFuture()) {
      new_stamp.generation = StorageGeneration::Invalid();
    }
    promise_.SetResult(std::move(new_stamp));
    this->CommitDone();
  }
  Promise<TimestampedStorageGeneration> promise_;
  ReadResult read_result_;
  bool if_equal_no_value_;
  ReadModifyWriteTarget* target_;
};
}  
Future<TimestampedStorageGeneration> WriteViaExistingTransaction(
    Driver* driver, internal::OpenTransactionPtr& transaction, size_t& phase,
    Key key, std::optional<Value> value, WriteOptions options) {
  TimestampedStorageGeneration stamp;
  if (StorageGeneration::IsUnknown(options.generation_conditions.if_equal)) {
    stamp.time = absl::InfiniteFuture();
  } else {
    assert(StorageGeneration::IsClean(options.generation_conditions.if_equal));
    stamp.time = absl::Time();
  }
  bool if_equal_no_value =
      StorageGeneration::IsNoValue(options.generation_conditions.if_equal);
  stamp.generation = std::move(options.generation_conditions.if_equal);
  stamp.generation.MarkDirty();
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();
  using Node = WriteViaExistingTransactionNode;
  internal::WeakTransactionNodePtr<Node> node;
  node.reset(new Node);
  node->promise_ = promise;
  node->read_result_ =
      value ? ReadResult::Value(*std::move(value), std::move(stamp))
            : ReadResult::Missing(std::move(stamp));
  node->if_equal_no_value_ = if_equal_no_value;
  TENSORSTORE_RETURN_IF_ERROR(
      driver->ReadModifyWrite(transaction, phase, std::move(key), *node));
  node->SetTransaction(*transaction);
  node->SetPhase(phase);
  TENSORSTORE_RETURN_IF_ERROR(node->Register());
  LinkError(std::move(promise), transaction->future());
  return std::move(future);
}
Future<TimestampedStorageGeneration> WriteViaTransaction(
    Driver* driver, Key key, std::optional<Value> value, WriteOptions options) {
  internal::OpenTransactionPtr transaction;
  size_t phase;
  return WriteViaExistingTransaction(driver, transaction, phase, std::move(key),
                                     std::move(value), std::move(options));
}
}  
namespace kvstore {
absl::Status Driver::ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                                     size_t& phase, Key key,
                                     ReadModifyWriteSource& source) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node,
      internal_kvstore::GetTransactionNode<
          internal_kvstore::NonAtomicTransactionNode>(this, transaction));
  internal_kvstore::MultiPhaseMutation::ReadModifyWriteStatus rmw_status;
  {
    absl::MutexLock lock(&node->mutex_);
    rmw_status = node->ReadModifyWrite(phase, std::move(key), source);
  }
  return internal_kvstore::GetNonAtomicReadModifyWriteError(*node, rmw_status);
}
absl::Status Driver::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  if (range.empty()) return absl::OkStatus();
  if (transaction && transaction->atomic()) {
    auto error = absl::InvalidArgumentError(
        tensorstore::StrCat("Cannot delete range starting at ",
                            this->DescribeKey(range.inclusive_min),
                            " as single atomic transaction"));
    transaction->RequestAbort(error);
    return error;
  }
  return internal_kvstore::AddDeleteRange<
      internal_kvstore::NonAtomicTransactionNode>(this, transaction,
                                                  std::move(range));
}
}  
}  