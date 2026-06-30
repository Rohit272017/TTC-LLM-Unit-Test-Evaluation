#include "tensorflow/c/experimental/filesystem/plugins/gcs/ram_file_block_cache.h"
#include <cstring>
#include <memory>
#include <sstream>
#include <utility>
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/experimental/filesystem/plugins/gcs/cleanup.h"
namespace tf_gcs_filesystem {
bool RamFileBlockCache::BlockNotStale(const std::shared_ptr<Block>& block) {
  absl::MutexLock l(&block->mu);
  if (block->state != FetchState::FINISHED) {
    return true;  
  }
  if (max_staleness_ == 0) return true;  
  return timer_seconds_() - block->timestamp <= max_staleness_;
}
std::shared_ptr<RamFileBlockCache::Block> RamFileBlockCache::Lookup(
    const Key& key) {
  absl::MutexLock lock(&mu_);
  auto entry = block_map_.find(key);
  if (entry != block_map_.end()) {
    if (BlockNotStale(entry->second)) {
      return entry->second;
    } else {
      RemoveFile_Locked(key.first);
    }
  }
  auto new_entry = std::make_shared<Block>();
  lru_list_.push_front(key);
  lra_list_.push_front(key);
  new_entry->lru_iterator = lru_list_.begin();
  new_entry->lra_iterator = lra_list_.begin();
  new_entry->timestamp = timer_seconds_();
  block_map_.emplace(std::make_pair(key, new_entry));
  return new_entry;
}
void RamFileBlockCache::Trim() {
  while (!lru_list_.empty() && cache_size_ > max_bytes_) {
    RemoveBlock(block_map_.find(lru_list_.back()));
  }
}
void RamFileBlockCache::UpdateLRU(const Key& key,
                                  const std::shared_ptr<Block>& block,
                                  TF_Status* status) {
  absl::MutexLock lock(&mu_);
  if (block->timestamp == 0) {
    return TF_SetStatus(status, TF_OK, "");
  }
  if (block->lru_iterator != lru_list_.begin()) {
    lru_list_.erase(block->lru_iterator);
    lru_list_.push_front(key);
    block->lru_iterator = lru_list_.begin();
  }
  if (block->data.size() < block_size_) {
    Key fmax = std::make_pair(key.first, std::numeric_limits<size_t>::max());
    auto fcmp = block_map_.upper_bound(fmax);
    if (fcmp != block_map_.begin() && key < (--fcmp)->first) {
      return TF_SetStatus(status, TF_INTERNAL,
                          "Block cache contents are inconsistent.");
    }
  }
  Trim();
  return TF_SetStatus(status, TF_OK, "");
}
void RamFileBlockCache::MaybeFetch(const Key& key,
                                   const std::shared_ptr<Block>& block,
                                   TF_Status* status) {
  bool downloaded_block = false;
  auto reconcile_state = MakeCleanup([this, &downloaded_block, &key, &block] {
    if (downloaded_block) {
      absl::MutexLock l(&mu_);
      if (block->timestamp != 0) {
        cache_size_ += block->data.capacity();
        lra_list_.erase(block->lra_iterator);
        lra_list_.push_front(key);
        block->lra_iterator = lra_list_.begin();
        block->timestamp = timer_seconds_();
      }
    }
  });
  absl::MutexLock l(&block->mu);
  TF_SetStatus(status, TF_OK, "");
  while (true) {
    switch (block->state) {
      case FetchState::ERROR:
      case FetchState::CREATED:
        block->state = FetchState::FETCHING;
        block->mu.Unlock();  
        block->data.clear();
        block->data.resize(block_size_, 0);
        int64_t bytes_transferred;
        bytes_transferred = block_fetcher_(key.first, key.second, block_size_,
                                           block->data.data(), status);
        block->mu.Lock();  
        if (TF_GetCode(status) == TF_OK) {
          block->data.resize(bytes_transferred, 0);
          std::vector<char>(block->data).swap(block->data);
          downloaded_block = true;
          block->state = FetchState::FINISHED;
        } else {
          block->state = FetchState::ERROR;
        }
        block->cond_var.SignalAll();
        return;
      case FetchState::FETCHING:
        block->cond_var.WaitWithTimeout(&block->mu, absl::Minutes(1));
        if (block->state == FetchState::FINISHED) {
          return TF_SetStatus(status, TF_OK, "");
        }
        break;
      case FetchState::FINISHED:
        return TF_SetStatus(status, TF_OK, "");
    }
  }
  return TF_SetStatus(
      status, TF_INTERNAL,
      "Control flow should never reach the end of RamFileBlockCache::Fetch.");
}
int64_t RamFileBlockCache::Read(const std::string& filename, size_t offset,
                                size_t n, char* buffer, TF_Status* status) {
  if (n == 0) {
    TF_SetStatus(status, TF_OK, "");
    return 0;
  }
  if (!IsCacheEnabled() || (n > max_bytes_)) {
    return block_fetcher_(filename, offset, n, buffer, status);
  }
  size_t start = block_size_ * (offset / block_size_);
  size_t finish = block_size_ * ((offset + n) / block_size_);
  if (finish < offset + n) {
    finish += block_size_;
  }
  size_t total_bytes_transferred = 0;
  for (size_t pos = start; pos < finish; pos += block_size_) {
    Key key = std::make_pair(filename, pos);
    std::shared_ptr<Block> block = Lookup(key);
    if (!block) {
      std::cerr << "No block for key " << key.first << "@" << key.second;
      abort();
    }
    MaybeFetch(key, block, status);
    if (TF_GetCode(status) != TF_OK) return -1;
    UpdateLRU(key, block, status);
    if (TF_GetCode(status) != TF_OK) return -1;
    const auto& data = block->data;
    if (offset >= pos + data.size()) {
      std::stringstream os;
      os << "EOF at offset " << offset << " in file " << filename
         << " at position " << pos << " with data size " << data.size();
      TF_SetStatus(status, TF_OUT_OF_RANGE, std::move(os).str().c_str());
      return total_bytes_transferred;
    }
    auto begin = data.begin();
    if (offset > pos) {
      begin += offset - pos;
    }
    auto end = data.end();
    if (pos + data.size() > offset + n) {
      end -= (pos + data.size()) - (offset + n);
    }
    if (begin < end) {
      size_t bytes_to_copy = end - begin;
      memcpy(&buffer[total_bytes_transferred], &*begin, bytes_to_copy);
      total_bytes_transferred += bytes_to_copy;
    }
    if (data.size() < block_size_) {
      break;
    }
  }
  TF_SetStatus(status, TF_OK, "");
  return total_bytes_transferred;
}
bool RamFileBlockCache::ValidateAndUpdateFileSignature(
    const std::string& filename, int64_t file_signature) {
  absl::MutexLock lock(&mu_);
  auto it = file_signature_map_.find(filename);
  if (it != file_signature_map_.end()) {
    if (it->second == file_signature) {
      return true;
    }
    RemoveFile_Locked(filename);
    it->second = file_signature;
    return false;
  }
  file_signature_map_[filename] = file_signature;
  return true;
}
size_t RamFileBlockCache::CacheSize() const {
  absl::MutexLock lock(&mu_);
  return cache_size_;
}
void RamFileBlockCache::Prune() {
  while (!stop_pruning_thread_.WaitForNotificationWithTimeout(
      absl::Microseconds(1000000))) {
    absl::MutexLock lock(&mu_);
    uint64_t now = timer_seconds_();
    while (!lra_list_.empty()) {
      auto it = block_map_.find(lra_list_.back());
      if (now - it->second->timestamp <= max_staleness_) {
        break;
      }
      RemoveFile_Locked(std::string(it->first.first));
    }
  }
}
void RamFileBlockCache::Flush() {
  absl::MutexLock lock(&mu_);
  block_map_.clear();
  lru_list_.clear();
  lra_list_.clear();
  cache_size_ = 0;
}
void RamFileBlockCache::RemoveFile(const std::string& filename) {
  absl::MutexLock lock(&mu_);
  RemoveFile_Locked(filename);
}
void RamFileBlockCache::RemoveFile_Locked(const std::string& filename) {
  Key begin = std::make_pair(filename, 0);
  auto it = block_map_.lower_bound(begin);
  while (it != block_map_.end() && it->first.first == filename) {
    auto next = std::next(it);
    RemoveBlock(it);
    it = next;
  }
}
void RamFileBlockCache::RemoveBlock(BlockMap::iterator entry) {
  entry->second->timestamp = 0;
  lru_list_.erase(entry->second->lru_iterator);
  lra_list_.erase(entry->second->lra_iterator);
  cache_size_ -= entry->second->data.capacity();
  block_map_.erase(entry);
}
}  