#include "xla/tsl/profiler/utils/preprocess_xplane.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
namespace tsl {
namespace profiler {
namespace {
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;
void MutateXPlane(XPlane& plane,
                  const std::vector<std::unique_ptr<XplaneEventMutatorFactory>>&
                      mutator_factories) {
  XPlaneBuilder plane_builder(&plane);
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<XplaneEventMutator>>>
      mutators_from_event_metadata_id;
  std::vector<std::unique_ptr<XplaneEventMutator>> line_mutators;
  for (const auto& mutator_factory : mutator_factories) {
    auto mutators = mutator_factory->CreateMutators(plane_builder);
    for (auto& mutator : mutators) {
      if (mutator->event_metadata()) {
        auto id = mutator->event_metadata()->id();
        mutators_from_event_metadata_id[id].push_back(std::move(mutator));
      } else {
        line_mutators.push_back(std::move(mutator));
      }
    }
  }
  if (mutators_from_event_metadata_id.empty() && line_mutators.empty()) {
    return;
  }
  plane_builder.ForEachLine([&](XLineBuilder line_builder) {
    for (const auto& mutator : line_mutators) {
      mutator->MutateEventsInLine(line_builder);
    }
    if (mutators_from_event_metadata_id.empty()) return;
    line_builder.ForEachEvent([&](XEventBuilder event_builder) {
      auto event_mutators =
          mutators_from_event_metadata_id.find(event_builder.MetadataId());
      if (event_mutators != mutators_from_event_metadata_id.end()) {
        for (const auto& mutator : event_mutators->second) {
          mutator->Mutate(event_builder);
        }
      }
    });
  });
}
std::vector<std::unique_ptr<XplaneEventMutatorFactory>>
CreateMutatorFactories() {
  std::vector<std::unique_ptr<XplaneEventMutatorFactory>> mutator_factories;
  mutator_factories.push_back(ThreadpoolLineMutatorFactory::CreateFactory());
  mutator_factories.push_back(XplaneRootEventMutatorFactory::CreateFactory(
      HostEventType::kProcessBatch, 2));
  mutator_factories.push_back(XplaneRootEventMutatorFactory::CreateFactory(
      HostEventType::kBatchingSessionRun, 1));
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          HostEventType::kExecutorStateProcess,
          HostEventType::kTpuExecuteOp, ContextType::kLegacy,
          false,
          XContextStatsAccessor<uint64_t, StatType::kStepId>,
          XContextStatsAccessor<uint64_t,
                                StatType::kIterNum>>::CreateFactory());
#define ADD_QUEUE_CONNECTION(__enque_event__, __deque_event__)            \
  mutator_factories.push_back(                                            \
      XplaneConnectedEventMutatorFactory<                                 \
          HostEventType::__enque_event__, HostEventType::__deque_event__, \
          ContextType::kTpuStream, true,                 \
          XContextStatsAccessor<uint64, StatType::kRequestId>,            \
          XContextStatsAccessor<uint64,                                   \
                                StatType::kQueueAddr>>::CreateFactory())
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kRunProgramRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kHostCallbackRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferH2DRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferPreprocessedH2DRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferD2HRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceRecvRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRecvLocalRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kCustomWait);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRequestMulti);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceRecvRequestMulti);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kPjrtAsyncWait);
#undef ADD_QUEUE_CONNECTION
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kDoEnqueueProgram>::CreateFactory());
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kCompleteCallbacks>::CreateFactory());
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kDoEnqueueContinuationProgram>::CreateFactory());
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          HostEventType::kDoEnqueueProgram,
          HostEventType::kCompleteCallbacks,
          ContextType::kTpuLaunch,
          true,
          XContextStatsAccessor<uint64_t, StatType::kDeviceOrdinal>,
          XContextStatsAccessor<uint64_t, StatType::kQueueId>,
          XContextStatsAccessor<uint64_t, StatType::kRunId>,
          XContextStatsAccessorWithDefault<uint64_t, StatType::kCoreType,
                                           0ULL>>::CreateFactory());
  mutator_factories.push_back(TpuModuleLineMutatorFactory::CreateFactory());
  return mutator_factories;
}
}  
void PreprocessXPlane(XPlane* plane) {
  if (plane == nullptr) return;
  auto mutator_factories = CreateMutatorFactories();
  MutateXPlane(*plane, mutator_factories);
}
void PreprocessXSpace(XSpace* space) {
  if (space == nullptr) return;
  auto mutator_factories = CreateMutatorFactories();
  for (XPlane& plane : *space->mutable_planes()) {
    MutateXPlane(plane, mutator_factories);
  }
}
}  
}  