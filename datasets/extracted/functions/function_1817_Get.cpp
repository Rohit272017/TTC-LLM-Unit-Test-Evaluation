#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include <utility>
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
namespace xla::gpu {
const HloFusionAnalysis& HloFusionAnalysisCache::Get(
    const HloInstruction& instruction) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = analyses_.find(instruction.unique_id());
    if (it != analyses_.end()) {
      return it->second;
    }
  }
  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(instruction, device_info_);
  absl::MutexLock lock(&mutex_);
  auto it = analyses_.find(instruction.unique_id());
  if (it != analyses_.end()) {
    return it->second;
  }
  return analyses_.emplace(instruction.unique_id(), std::move(analysis))
      .first->second;
}
const HloFusionAnalysis& HloFusionAnalysisCache::Get(
    const HloInstruction& producer, const HloInstruction& consumer) {
  std::pair<int, int> key{producer.unique_id(), consumer.unique_id()};
  {
    absl::MutexLock lock(&mutex_);
    auto it = producer_consumer_analyses_.find(key);
    if (it != producer_consumer_analyses_.end()) {
      return it->second;
    }
  }
  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(producer, consumer, device_info_);
  absl::MutexLock lock(&mutex_);
  auto it = producer_consumer_analyses_.find(key);
  if (it != producer_consumer_analyses_.end()) {
    return it->second;
  }
  producers_for_consumers_[consumer.unique_id()].push_back(
      producer.unique_id());
  consumers_for_producers_[producer.unique_id()].push_back(
      consumer.unique_id());
  return producer_consumer_analyses_.emplace(key, std::move(analysis))
      .first->second;
}
void HloFusionAnalysisCache::Invalidate(const HloInstruction& instruction) {
  analyses_.erase(instruction.unique_id());
  if (auto consumers =
          consumers_for_producers_.extract(instruction.unique_id())) {
    for (const auto consumer : consumers.mapped()) {
      producer_consumer_analyses_.erase({instruction.unique_id(), consumer});
    }
  }
  if (auto producers =
          producers_for_consumers_.extract(instruction.unique_id())) {
    for (const auto producer : producers.mapped()) {
      producer_consumer_analyses_.erase({producer, instruction.unique_id()});
    }
  }
}
void HloFusionAnalysisCache::Clear() {
  analyses_.clear();
  producer_consumer_analyses_.clear();
  consumers_for_producers_.clear();
  producers_for_consumers_.clear();
}
}  