#ifndef TENSORFLOW_CORE_DATA_SERVICE_CROSS_TRAINER_CACHE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CROSS_TRAINER_CACHE_H_
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
namespace tensorflow {
namespace data {
template <class ElementType>
class CachableSequence {
 public:
  virtual ~CachableSequence() = default;
  virtual StatusOr<ElementType> GetNext() = 0;
  virtual size_t GetElementSizeBytes(const ElementType&) const = 0;
};
template <class ElementType>
class CrossTrainerCache {
 public:
  explicit CrossTrainerCache(
      size_t max_cache_size_bytes,
      std::unique_ptr<CachableSequence<ElementType>> cachable_sequence);
  virtual ~CrossTrainerCache() = default;
  CrossTrainerCache(const CrossTrainerCache&) = delete;
  CrossTrainerCache& operator=(const CrossTrainerCache&) = delete;
  StatusOr<std::shared_ptr<const ElementType>> Get(
      const std::string& trainer_id);
  void Cancel(Status status);
  bool IsCancelled() const;
 private:
  struct CacheQueryResult {
    std::shared_ptr<const ElementType> element;
    bool cache_hit;
  };
  StatusOr<CacheQueryResult> GetCacheQueryResult(const std::string& trainer_id);
  bool IsElementReady(const std::string& trainer_id);
  size_t GetElementIndex(const std::string& trainer_id);
  StatusOr<std::shared_ptr<const ElementType>> GetElement(
      const std::string& trainer_id);
  Status ExtendCache();
  void FreeSpace(size_t new_element_size_bytes);
  void RecordMetrics(const CacheQueryResult& result);
  const size_t max_cache_size_bytes_;
  std::unique_ptr<CachableSequence<ElementType>> cachable_sequence_;
  mutable mutex mu_;
  mutable condition_variable cv_;
  Status status_ TF_GUARDED_BY(mu_) = absl::OkStatus();
  std::deque<std::shared_ptr<const ElementType>> cache_ TF_GUARDED_BY(mu_);
  size_t cache_size_bytes_ TF_GUARDED_BY(mu_) = 0;
  size_t cache_start_index_ TF_GUARDED_BY(mu_) = 0;
  bool extending_cache_ TF_GUARDED_BY(mu_) = false;
  absl::flat_hash_map<std::string, size_t> trainer_to_element_index_map_
      TF_GUARDED_BY(mu_);
};
template <class ElementType>
CrossTrainerCache<ElementType>::CrossTrainerCache(
    size_t max_cache_size_bytes,
    std::unique_ptr<CachableSequence<ElementType>> cachable_sequence)
    : max_cache_size_bytes_(max_cache_size_bytes),
      cachable_sequence_(std::move(cachable_sequence)) {
  DCHECK_GT(max_cache_size_bytes, 0)
      << "CrossTrainerCache size must be greater than 0.";
  VLOG(2) << "Initialized tf.data service cross-trainer cache with "
          << ByteSize::Bytes(max_cache_size_bytes) << " of memory.";
}
template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
CrossTrainerCache<ElementType>::Get(const std::string& trainer_id)
    TF_LOCKS_EXCLUDED(mu_) {
  if (trainer_id.empty()) {
    return errors::InvalidArgument(
        "tf.data service cross-trainer cache requires a non-empty trainer ID.");
  }
  TF_ASSIGN_OR_RETURN(CacheQueryResult result, GetCacheQueryResult(trainer_id));
  RecordMetrics(result);
  return result.element;
}
template <class ElementType>
StatusOr<typename CrossTrainerCache<ElementType>::CacheQueryResult>
CrossTrainerCache<ElementType>::GetCacheQueryResult(
    const std::string& trainer_id) {
  bool should_extend_cache = false;
  while (true) {
    {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(status_);
      if (IsElementReady(trainer_id)) {
        TF_ASSIGN_OR_RETURN(std::shared_ptr<const ElementType> element,
                            GetElement(trainer_id));
        return CacheQueryResult{element,
                                !should_extend_cache};
      }
      if (extending_cache_) {
        should_extend_cache = false;
        cv_.wait(l);
      } else {
        should_extend_cache = true;
        extending_cache_ = true;
      }
    }
    if (should_extend_cache) {
      Status s = ExtendCache();
      mutex_lock l(mu_);
      extending_cache_ = false;
      cv_.notify_all();
      TF_RETURN_IF_ERROR(s);
    }
  }
}
template <class ElementType>
bool CrossTrainerCache<ElementType>::IsElementReady(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  return GetElementIndex(trainer_id) < cache_start_index_ + cache_.size();
}
template <class ElementType>
StatusOr<std::shared_ptr<const ElementType>>
CrossTrainerCache<ElementType>::GetElement(const std::string& trainer_id)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t element_index = GetElementIndex(trainer_id);
  if (element_index >= std::numeric_limits<size_t>::max()) {
    return errors::Internal(
        "tf.data service caching element index exceeds integer limit. Got ",
        element_index);
  }
  std::shared_ptr<const ElementType> result =
      cache_[element_index - cache_start_index_];
  trainer_to_element_index_map_[trainer_id] = element_index + 1;
  return result;
}
template <class ElementType>
size_t CrossTrainerCache<ElementType>::GetElementIndex(
    const std::string& trainer_id) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t element_index = trainer_to_element_index_map_[trainer_id];
  if (element_index < cache_start_index_) {
    element_index = cache_start_index_;
  }
  return element_index;
}
template <class ElementType>
Status CrossTrainerCache<ElementType>::ExtendCache() TF_LOCKS_EXCLUDED(mu_) {
  TF_ASSIGN_OR_RETURN(ElementType element, cachable_sequence_->GetNext());
  size_t new_element_size_bytes =
      cachable_sequence_->GetElementSizeBytes(element);
  if (new_element_size_bytes > max_cache_size_bytes_) {
    return errors::InvalidArgument(
        "tf.data service element size is larger than cache size in bytes. Got ",
        "element size: ", new_element_size_bytes,
        " and cache size: ", max_cache_size_bytes_);
  }
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(status_);
  FreeSpace(new_element_size_bytes);
  cache_.push_back(std::make_shared<ElementType>(std::move(element)));
  cache_size_bytes_ += new_element_size_bytes;
  return absl::OkStatus();
}
template <class ElementType>
void CrossTrainerCache<ElementType>::FreeSpace(size_t new_element_size_bytes)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  size_t num_elements_discarded = 0;
  while (!cache_.empty() &&
         cache_size_bytes_ + new_element_size_bytes > max_cache_size_bytes_) {
    size_t free_bytes =
        cachable_sequence_->GetElementSizeBytes(*cache_.front());
    cache_.pop_front();
    cache_size_bytes_ -= free_bytes;
    ++cache_start_index_;
    ++num_elements_discarded;
  }
  VLOG(3) << "Freed " << num_elements_discarded << " element(s) from "
          << "tf.data service cross-trainer cache. Memory usage: "
          << ByteSize::Bytes(cache_size_bytes_) << ".";
}
template <class ElementType>
void CrossTrainerCache<ElementType>::Cancel(Status status)
    TF_LOCKS_EXCLUDED(mu_) {
  DCHECK(!status.ok())
      << "Cancelling CrossTrainerCache requires a non-OK status. Got "
      << status;
  VLOG(2) << "Cancel tf.data service cross-trainer cache with status "
          << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  cv_.notify_all();
}
template <class ElementType>
bool CrossTrainerCache<ElementType>::IsCancelled() const
    TF_LOCKS_EXCLUDED(mu_) {
  mutex_lock l(mu_);
  return !status_.ok();
}
template <class ElementType>
void CrossTrainerCache<ElementType>::RecordMetrics(
    const CacheQueryResult& result) {
  metrics::RecordTFDataServiceCrossTrainerCacheQuery(result.cache_hit);
  size_t cache_size_bytes = 0;
  {
    mutex_lock l(mu_);
    cache_size_bytes = cache_size_bytes_;
  }
  metrics::RecordTFDataServiceCrossTrainerCacheSizeBytes(cache_size_bytes);
}
}  
}  
#endif  