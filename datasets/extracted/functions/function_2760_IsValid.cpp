#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "tsl/platform/protobuf.h"
namespace xla {
namespace spmd {
namespace {
using PrimIdx = int64_t;  
using LiveIdx = int64_t;  
using GroupIdx = int64_t;  
using PrimPair = std::pair<PrimIdx, PrimIdx>;
using Interval = std::pair<LiveIdx, LiveIdx>;
using ActivePrim = std::pair<Interval, PrimIdx>;
bool IsValid(const Interval& interval) {
  return interval.first <= interval.second;
}
int64_t length(const Interval& interval) {
  return interval.second - interval.first + 1;  
}
}  
std::pair<int64_t, int64_t> MemoryTermReducer::Reduce(
    int64_t num_lives, int64_t num_primitives,
    const std::function<
        tsl::protobuf::RepeatedField<int64_t>(int64_t)>&  
        live,
    int64_t max_iterations) {
  LOG(INFO) << "Memory Term Reducer beginning to reduce number of terms ...";
  reduced_live_.clear();
  reduced_intervals_.clear();
  reduced_groups_.clear();
  int64_t num_terms = 0;
  reduced_intervals_.reserve(num_primitives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    reduced_intervals_.push_back({std::numeric_limits<LiveIdx>::max(), 0});
  }
  for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
    for (const PrimIdx prim_idx : live(live_idx)) {
      Interval& interval = reduced_intervals_[prim_idx];
      interval.first = std::min(interval.first, live_idx);
      interval.second = std::max(interval.second, live_idx);
      ++num_terms;
    }
  }
  Reduce(num_lives, num_primitives, max_iterations);
  int64_t num_reduced_terms = 0;
  reduced_live_.resize(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < reduced_intervals_.size(); ++prim_idx) {
    const Interval& interval = reduced_intervals_[prim_idx];
    for (LiveIdx live_idx = interval.first; live_idx <= interval.second;
         ++live_idx) {
      reduced_live_[live_idx].push_back(prim_idx);
      ++num_reduced_terms;
    }
  }
  for (const auto& group : reduced_groups_) num_reduced_terms += group.size();
  LOG(INFO) << "Memory Term Reducer finished reducing the number of terms.";
  return {num_terms, num_reduced_terms};
}
std::pair<int64_t, int64_t> MemoryTermReducer::Reduce(
    int64_t num_lives, int64_t num_primitives,
    const std::function<std::pair<int64_t, int64_t>(int64_t)>& intervals,
    int64_t max_iterations) {
  LOG(INFO) << "Memory Term Reducer beginning to reduce number of terms ...";
  reduced_live_.clear();
  reduced_intervals_.clear();
  reduced_groups_.clear();
  int64_t num_terms = 0;
  reduced_intervals_.reserve(num_primitives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    reduced_intervals_.push_back(intervals(prim_idx));
    const Interval& interval = reduced_intervals_.back();
    if (IsValid(interval)) num_terms += length(interval);
  }
  Reduce(num_lives, num_primitives, max_iterations);
  int64_t num_reduced_terms = 0;
  for (PrimIdx prim_idx = 0; prim_idx < reduced_intervals_.size(); ++prim_idx) {
    const Interval& interval = reduced_intervals_[prim_idx];
    if (IsValid(interval)) num_reduced_terms += length(interval);
  }
  for (const auto& group : reduced_groups_) num_reduced_terms += group.size();
  LOG(INFO) << "Memory Term Reducer finished reducing the number of terms.";
  return {num_terms, num_reduced_terms};
}
void MemoryTermReducer::Reduce(int64_t num_lives, int64_t num_primitives,
                               int64_t max_iterations) {
  std::vector<absl::btree_set<PrimIdx>> enter(num_lives), evict(num_lives);
  for (PrimIdx prim_idx = 0; prim_idx < num_primitives; ++prim_idx) {
    const Interval& interval = reduced_intervals_[prim_idx];
    if (!IsValid(interval)) continue;  
    enter[interval.first].insert(prim_idx);
    evict[interval.second].insert(prim_idx);
  }
  auto Splits = [this](PrimIdx large_idx, PrimIdx small_idx) -> bool {
    const Interval& large = reduced_intervals_[large_idx];
    const Interval& small = reduced_intervals_[small_idx];
    return large.first < small.first && large.second > small.second;
  };
  auto CalcOverlap = [this, Splits](
                         int64_t prim0_idx,
                         int64_t prim1_idx) -> std::optional<Interval> {
    if (prim0_idx == prim1_idx) return std::nullopt;  
    const Interval& interval0 = reduced_intervals_[prim0_idx];
    const Interval& interval1 = reduced_intervals_[prim1_idx];
    if (!IsValid(interval0) || !IsValid(interval1)) return std::nullopt;
    if (Splits(prim0_idx, prim1_idx)) return std::nullopt;
    if (Splits(prim1_idx, prim0_idx)) return std::nullopt;
    return Interval(std::max(interval0.first, interval1.first),
                    std::min(interval0.second, interval1.second));
  };
  auto MergeIntoGroup = [num_primitives, this](
                            PrimIdx prim_idx,
                            absl::btree_set<PrimIdx>& reduced_group) {
    if (prim_idx < num_primitives) {
      reduced_group.insert(prim_idx);
    } else {
      const auto& group = reduced_groups_[prim_idx - num_primitives];
      reduced_group.insert(group.begin(), group.end());
    }
  };
  auto CalcNumTerms = [num_primitives, this](
                          PrimIdx prim_idx,
                          std::optional<Interval> overlap = std::nullopt) {
    int64_t num_terms = length(reduced_intervals_[prim_idx]);
    if (overlap) num_terms -= length(*overlap);
    if (prim_idx >= num_primitives && num_terms > 0) {
      num_terms += reduced_groups_[prim_idx - num_primitives].size();
    }
    return num_terms;
  };
  auto UpdatePrimitive = [this, &enter, &evict](
                             PrimIdx prim_idx,
                             const Interval& overlap) mutable {
    Interval& interval = reduced_intervals_[prim_idx];
    enter[interval.first].erase(prim_idx);
    evict[interval.second].erase(prim_idx);
    if (auto& t = interval.first; t == overlap.first) t = overlap.second + 1;
    if (auto& t = interval.second; t == overlap.second) t = overlap.first - 1;
    if (!IsValid(interval)) return;  
    enter[interval.first].insert(prim_idx);
    evict[interval.second].insert(prim_idx);
  };
  auto SweepAndMerge = [&num_lives, &enter, &evict, &CalcOverlap, &CalcNumTerms,
                        &MergeIntoGroup, &UpdatePrimitive, this]() -> bool {
    absl::btree_set<ActivePrim> actives;  
    absl::btree_multimap<int64_t, PrimPair> overlaps;
    for (LiveIdx live_idx = 0; live_idx < num_lives; ++live_idx) {
      for (const PrimIdx prim_idx : enter[live_idx]) {
        actives.insert({reduced_intervals_[prim_idx], prim_idx});
      }
      for (const PrimIdx prim_idx : evict[live_idx]) {
        auto active = actives.find({reduced_intervals_[prim_idx], prim_idx});
        if (++active == actives.end()) continue;  
        std::optional<Interval> overlap = CalcOverlap(prim_idx, active->second);
        if (!overlap) continue;
        overlaps.insert({-length(*overlap), {prim_idx, active->second}});
      }
      for (const PrimIdx prim_idx : evict[live_idx]) {
        actives.erase({reduced_intervals_[prim_idx], prim_idx});
      }
    }
    bool changed = false;
    for (const auto& [_, prim_pair] : overlaps) {
      const PrimIdx prim0_idx = prim_pair.first, prim1_idx = prim_pair.second;
      const std::optional<Interval> overlap = CalcOverlap(prim0_idx, prim1_idx);
      if (!overlap) continue;
      absl::btree_set<PrimIdx> reduced_group;
      MergeIntoGroup(prim0_idx, reduced_group);
      MergeIntoGroup(prim1_idx, reduced_group);
      if (CalcNumTerms(prim0_idx) + CalcNumTerms(prim1_idx) <=
          CalcNumTerms(prim0_idx, overlap) + CalcNumTerms(prim1_idx, overlap) +
              length(*overlap) + reduced_group.size()) {
        continue;  
      }
      enter[overlap->first].insert(reduced_intervals_.size());
      evict[overlap->second].insert(reduced_intervals_.size());
      reduced_intervals_.push_back({overlap->first, overlap->second});
      reduced_groups_.push_back(reduced_group);
      UpdatePrimitive(prim0_idx, *overlap);
      UpdatePrimitive(prim1_idx, *overlap);
      changed = true;
    }
    return changed;
  };
  for (int64_t iteration = 0; iteration < max_iterations; ++iteration) {
    if (!SweepAndMerge()) break;
  }
  for (GroupIdx group_idx = reduced_groups_.size() - 1; group_idx >= 0;
       --group_idx) {
    if (IsValid(reduced_intervals_[num_primitives + group_idx])) continue;
    reduced_intervals_.erase(reduced_intervals_.begin() + num_primitives +
                             group_idx);
    reduced_groups_.erase(reduced_groups_.begin() + group_idx);
  }
}
const std::vector<std::vector<int64_t>>& MemoryTermReducer::GetReducedLive()
    const {
  return reduced_live_;
}
const std::vector<std::pair<int64_t, int64_t>>&
MemoryTermReducer::GetReducedIntervals() const {
  return reduced_intervals_;
}
const std::vector<absl::btree_set<int64_t>>&
MemoryTermReducer::GetReducedGroups() const {
  return reduced_groups_;
}
absl::flat_hash_set<int64_t> MemoryTermReducer::GetReducedTimes(
    int64_t num_primitives) {
  return GetReducedTimes(num_primitives, reduced_intervals_, reduced_groups_);
}
absl::flat_hash_set<int64_t> MemoryTermReducer::GetReducedTimes(
    int64_t num_primitives,
    const std::vector<std::pair<int64_t, int64_t>>& reduced_intervals,
    const std::vector<absl::btree_set<int64_t>>& reduced_groups) {
  std::vector<std::pair<int64_t, int64_t>> intervals;
  for (int64_t reduced_interval_idx = 0;
       reduced_interval_idx < reduced_intervals.size();
       ++reduced_interval_idx) {
    const Interval& reduced_interval = reduced_intervals[reduced_interval_idx];
    if (reduced_interval_idx < num_primitives) {
      intervals.push_back(reduced_interval);
      continue;
    }
    const GroupIdx group_idx = reduced_interval_idx - num_primitives;
    for (const PrimIdx prim_idx : reduced_groups[group_idx]) {
      Interval& interval = intervals[prim_idx];
      if (!IsValid(interval)) {
        interval.first = reduced_interval.first;
        interval.second = reduced_interval.second;
        continue;
      }
      interval.first = std::min(interval.first, reduced_interval.first);
      interval.second = std::max(interval.second, reduced_interval.second);
    }
  }
  absl::btree_set<std::pair<int64_t, bool>> times;
  for (const Interval& interval : intervals) {
    if (!IsValid(interval)) continue;
    times.insert({interval.first, false});
    times.insert({interval.second, true});
  }
  int64_t last_entering_time = -1;
  absl::flat_hash_set<int64_t> reduced_times;
  for (const auto& time : times) {
    if ( time.second) {
      reduced_times.insert(last_entering_time);
    } else {
      last_entering_time = time.first;
    }
  }
  reduced_times.insert(last_entering_time);
  return reduced_times;
}
}  
}  