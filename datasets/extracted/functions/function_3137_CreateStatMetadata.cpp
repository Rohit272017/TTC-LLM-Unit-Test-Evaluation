#include "xla/tsl/profiler/utils/group_events.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/env.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace profiler {
void CreateStatMetadata(XPlane* plane) {
  XPlaneBuilder builder(plane);
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsEager));
}
std::optional<int64_t> GetKernelEventType(bool is_host_plane,
                                          const XEventVisitor& event) {
  if (event.GetStat(StatType::kCorrelationId).has_value()) {
    return is_host_plane ? HostEventType::kKernelLaunch
                         : HostEventType::kKernelExecute;
  }
  return std::nullopt;
}
int64_t GetEventType(bool is_host_plane, const XEventVisitor& event) {
  if (std::optional<int64_t> event_type = event.Type()) {
    return *event_type;
  } else if (std::optional<int64_t> kernel_event_type =
                 GetKernelEventType(is_host_plane, event)) {
    return *kernel_event_type;
  } else {
    return HostEventType::kUnknownHostEventType;
  }
}
bool IsLegacyRootEvent(const XEventVisitor& event) {
  return event.Type() == HostEventType::kTraceContext;
}
struct GroupingEventStats {
  explicit GroupingEventStats(const XEventVisitor& event);
  std::optional<int> producer_type;
  std::optional<uint64_t> producer_id;
  std::optional<int> consumer_type;
  std::optional<uint64_t> consumer_id;
  std::optional<int> root_level;
  bool is_async = false;
};
GroupingEventStats::GroupingEventStats(const XEventVisitor& event) {
  std::optional<int64_t> step_id;
  event.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (*stat.Type()) {
      case StatType::kProducerType:
        producer_type = stat.IntValue();
        break;
      case StatType::kProducerId:
        producer_id = stat.IntOrUintValue();
        break;
      case StatType::kConsumerType:
        consumer_type = stat.IntValue();
        break;
      case StatType::kConsumerId:
        consumer_id = stat.IntOrUintValue();
        break;
      case StatType::kIsRoot:
        root_level = stat.IntValue();
        break;
      case StatType::kIsAsync:
        is_async = stat.BoolValue();
        break;
      case StatType::kStepId:
        step_id = stat.IntValue();
        break;
      default:
        break;
    }
  });
  if (!root_level.has_value() && IsLegacyRootEvent(event)) {
    root_level = 1;
  }
}
void SetContextGroup(const GroupingEventStats& stats, EventNode* event,
                     ContextGroupMap* context_groups) {
  if (stats.producer_type.has_value() && stats.producer_id.has_value()) {
    ((*context_groups)[*stats.producer_type][*stats.producer_id])
        .producers.push_back(event);
  }
  if (stats.consumer_type.has_value() && stats.consumer_id.has_value()) {
    ((*context_groups)[*stats.consumer_type][*stats.consumer_id])
        .consumers.push_back(event);
  }
}
void ConnectContextGroups(const ContextGroupMap& context_groups) {
  for (auto& type_id_group : context_groups) {
    for (auto& id_group : type_id_group.second) {
      const ContextGroup& group = id_group.second;
      if (group.producers.size() >= 64 && group.consumers.size() >= 64) {
        LOG_EVERY_N(WARNING, 1000)
            << "id:" << id_group.first
            << " producers:" << group.producers.size() << " : "
            << group.producers[0]->GetEventVisitor().Name()
            << " consumers:" << group.consumers.size() << " : "
            << group.consumers[0]->GetEventVisitor().Name();
        continue;
      }
      for (EventNode* parent : group.producers) {
        for (EventNode* child : group.consumers) {
          parent->AddChild(child);
        }
      }
    }
  }
}
bool IsImplicitRootEvent(const XEventVisitor& event) {
  static const auto* const kImplicitRootEvents =
      new absl::flat_hash_set<int64_t>{
          HostEventType::kFunctionRun, HostEventType::kSessionRun,
          HostEventType::kRunGraph, HostEventType::kExecutorStateProcess};
  return event.Type().has_value() &&
         kImplicitRootEvents->contains(*event.Type());
}
void ProcessRootEvent(int64_t group_id, EventNode* root_event,
                      GroupMetadataMap* group_metadata_map) {
  root_event->PropagateGroupId(group_id, group_metadata_map);
  std::string group_name = root_event->GetGroupName();
  if (!IsImplicitRootEvent(root_event->GetEventVisitor())) {
    root_event->AddStepName(group_name);
  }
  (*group_metadata_map)[group_id].name = std::move(group_name);
}
using Comparator = std::function<bool(const EventNode*)>;
const EventNode* FindParentWithComparator(const Comparator& comparator,
                                          const EventNode* node,
                                          bool include_self) {
  std::queue<const EventNode*> nodes;
  absl::flat_hash_set<const EventNode*> seen = {node};
  if (include_self) {
    nodes.push(node);
  } else {
    for (const EventNode* parent : node->GetParents()) {
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  while (!nodes.empty()) {
    const EventNode* node = nodes.front();
    nodes.pop();
    if (comparator(node)) return node;
    for (const EventNode* parent : node->GetParents()) {
      if (seen.contains(parent)) continue;
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  return nullptr;
}
bool IsIteratorEventType(std::optional<int64_t> event_type) {
  return event_type == HostEventType::kIterator ||
         event_type == HostEventType::kDeviceInputPipelineSecondIterator;
}
bool CheckLoopOp(const XSpace& space) {
  for (const XPlane& plane : space.planes()) {
    for (const auto& event_metadata : plane.event_metadata()) {
      std::optional<int64_t> event_type =
          FindHostEventType(event_metadata.second.name());
      if (!event_type.has_value()) continue;
      switch (*event_type) {
        case HostEventType::kWhileOpEvalCond:
        case HostEventType::kWhileOpStartBody:
        case HostEventType::kForOp:
        case HostEventType::kParallelForOp:
        case HostEventType::kForeverOp:
          return true;
        default:
          break;
      }
    }
  }
  return false;
}
std::optional<XStatVisitor> EventNode::GetContextStat(int64_t stat_type) const {
  std::queue<const EventNode*> nodes;
  absl::flat_hash_set<const EventNode*> seen = {this};
  nodes.push(this);
  while (!nodes.empty()) {
    const EventNode* node = nodes.front();
    nodes.pop();
    if (std::optional<XStatVisitor> stat = node->visitor_.GetStat(stat_type)) {
      return stat;
    }
    for (const EventNode* parent : node->GetParents()) {
      if (seen.contains(parent)) continue;
      nodes.push(parent);
      seen.insert(parent);
    }
  }
  return std::nullopt;
}
std::string EventNode::GetGroupName() const {
  std::string name;
  if (std::optional<XStatVisitor> stat = GetContextStat(StatType::kGraphType)) {
    absl::StrAppend(&name, stat->StrOrRefValue(), " ");
  } else if (!(IsImplicitRootEvent(visitor_))) {
    absl::StrAppend(&name, GetEventVisitor().Name(), " ");
  }
  int64_t step_num = group_id_.value_or(0);
  if (std::optional<XStatVisitor> stat = GetContextStat(StatType::kIterNum)) {
    step_num = stat->IntValue();
  } else if (std::optional<XStatVisitor> stat =
                 GetContextStat(StatType::kStepNum)) {
    step_num = stat->IntValue();
  }
  absl::StrAppend(&name, step_num);
  return name;
}
XStat* EventNode::FindOrAddStatByType(int64_t stat_type) {
  const XPlaneVisitor& plane = visitor_.Plane();
  const XStatMetadata* stat_metadata = plane.GetStatMetadataByType(stat_type);
  DCHECK(stat_metadata != nullptr);
  auto* raw_event = const_cast<XEvent*>(&visitor_.RawEvent());  
  return FindOrAddMutableStat(*stat_metadata, raw_event);
}
void EventNode::SetGroupId(int64_t group_id) {
  group_id_ = group_id;
  FindOrAddStatByType(StatType::kGroupId)->set_int64_value(group_id);
}
void EventNode::PropagateGroupId(int64_t group_id,
                                 GroupMetadataMap* group_metadata_map) {
  std::queue<EventNode*> nodes;
  absl::flat_hash_set<EventNode*> seen = {this};
  nodes.push(this);
  while (!nodes.empty()) {
    EventNode* node = nodes.front();
    nodes.pop();
    std::optional<int64_t> node_group_id = node->GetGroupId();
    if (node_group_id.has_value()) {
      if (*node_group_id != group_id) {
        (*group_metadata_map)[group_id].children.insert(*node_group_id);
        (*group_metadata_map)[*node_group_id].parents.insert(group_id);
      }
    } else {
      node->SetGroupId(group_id);
      for (EventNode* child : node->GetChildren()) {
        if (seen.contains(child)) continue;
        nodes.push(child);
        seen.insert(child);
      }
    }
  }
}
void EventNode::AddStepName(absl::string_view step_name) {
  FindOrAddStatByType(StatType::kStepName)
      ->set_str_value(step_name.data(), step_name.size());
}
void EventNode::SetIsEager(bool is_eager) {
  FindOrAddStatByType(StatType::kIsEager)->set_int64_value(is_eager ? 1 : 0);
}
bool EventNode::IsCompiledFunc() const {
  auto is_func = visitor_.GetStat(StatType::kIsFunc);
  return !is_func || is_func->IntValue();
}
bool EventNode::IsEager() const {
  const EventNode* node = FindParent(HostEventType::kEagerKernelExecute);
  if (node == nullptr) {
    return false;
  }
  return !node->IsCompiledFunc();
}
const EventNode* EventNode::FindParent(int64_t event_type) const {
  return FindParentWithComparator(
      [event_type](const EventNode* node) {
        return node->GetEventVisitor().Type() == event_type;
      },
      this, true);
}
void EventForest::FindEventNodeAndApply(
    const int64_t event_type, const std::vector<int64_t>& stat_types,
    const std::function<void(EventNode&, const std::vector<uint64>&)>& cb) {
  if (auto* event_node_list = gtl::FindOrNull(event_node_map_, event_type)) {
    for (EventNode& event_node : *event_node_list) {
      std::vector<uint64> stats;
      for (const auto stat_type : stat_types) {
        std::optional<XStatVisitor> stat =
            event_node.GetEventVisitor().GetStat(stat_type);
        if (!stat) break;
        stats.push_back(stat->IntOrUintValue());
      }
      if (stats.size() == stat_types.size()) {
        cb(event_node, stats);
      }
    }
  }
}
void EventForest::ConnectIntraThread(XPlane* plane, XPlaneVisitor* visitor,
                                     ContextGroupMap* context_groups) {
  bool is_host_plane = (visitor->Name() == kHostThreadsPlaneName);
  for (auto& line : *plane->mutable_lines()) {
    std::vector<EventNode*> parent_nodes;
    for (auto& event : *line.mutable_events()) {
      XEventVisitor event_visitor(visitor, &line, &event);
      int64_t event_type = GetEventType(is_host_plane, event_visitor);
      EventNode* cur_node =
          &event_node_map_[event_type].emplace_back(std::move(event_visitor));
      GroupingEventStats stats(cur_node->GetEventVisitor());
      if (stats.root_level.has_value()) {
        cur_node->SetRootLevel(*stats.root_level);
      }
      SetContextGroup(stats, cur_node, context_groups);
      if (!stats.is_async) {
        while (!parent_nodes.empty()) {
          EventNode* parent_node = parent_nodes.back();
          if (parent_node->GetEventVisitor().GetTimespan().Includes(
                  cur_node->GetEventVisitor().GetTimespan())) {
            parent_node->AddChild(cur_node);
            break;
          } else {
            parent_nodes.pop_back();
          }
        }
        parent_nodes.push_back(cur_node);
      }
    }
  }
}
void EventForest::ConnectInterThread(
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  for (const auto& connect_info : connect_info_list) {
    absl::flat_hash_map<std::vector<uint64>, EventNode*> connect_map;
    const std::vector<int64_t>& parent_stat_types =
        connect_info.parent_stat_types;
    const std::vector<int64_t>* child_stat_types =
        &connect_info.child_stat_types;
    if (child_stat_types->empty()) {
      child_stat_types = &parent_stat_types;
    }
    FindEventNodeAndApply(connect_info.parent_event_type, parent_stat_types,
                          [&connect_map](EventNode& event_node,
                                         const std::vector<uint64>& stats) {
                            connect_map[stats] = &event_node;
                          });
    FindEventNodeAndApply(
        connect_info.child_event_type, *child_stat_types,
        [&connect_map](EventNode& event_node,
                       const std::vector<uint64>& stats) {
          if (auto parent_event_node = gtl::FindPtrOrNull(connect_map, stats)) {
            parent_event_node->AddChild(&event_node);
          }
        });
  }
}
bool RootNeedsGrouping(const EventNode* root) {
  if (root->GetGroupId().has_value()) return false;
  const EventNode* root_parent = FindParentWithComparator(
      [root](const EventNode* parent) {
        return parent->RootLevel() == root->RootLevel();
      },
      root,
      false);
  return root_parent == nullptr;
}
void SortRootEventList(EventList* event_list) {
  absl::c_sort(*event_list, [](const EventNode* e1, const EventNode* e2) {
    return e1->RootLevel() == e2->RootLevel()
               ? *e1 < *e2
               : e1->RootLevel() > e2->RootLevel();
  });
}
void EventForest::CreateEventGroups() {
  int64_t group_id = 0;
  if (!tf_loop_root_events_.empty()) {
    for (EventNode* root_event : tf_loop_root_events_) {
      ProcessRootEvent(group_id++, root_event, &group_metadata_map_);
    }
    return;
  }
  EventList root_events;
  for (auto& [event_type, events] : event_node_map_) {
    for (EventNode& event : events) {
      if (!event.RootLevel()) continue;
      std::optional<XStatVisitor> step_id_stat =
          event.GetEventVisitor().GetStat(StatType::kStepId);
      if (step_id_stat && tf_data_step_ids_.contains(step_id_stat->IntValue()))
        continue;
      root_events.push_back(&event);
    }
  }
  SortRootEventList(&root_events);
  for (EventNode* root_event : root_events) {
    if (RootNeedsGrouping(root_event)) {
      ProcessRootEvent(group_id++, root_event, &group_metadata_map_);
    }
  }
}
void EventForest::MarkEagerlyExecutedGpuKernels() {
  auto kernel_execute_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kKernelExecute);
  if (!kernel_execute_event_node_list) return;
  for (EventNode& kernel_execute_event_node : *kernel_execute_event_node_list) {
    kernel_execute_event_node.SetIsEager(kernel_execute_event_node.IsEager());
  }
}
void EventForest::MarkEagerlyExecutedCpuTfOps() {
  auto tf_op_run_event_node_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kTfOpRun);
  if (!tf_op_run_event_node_list) return;
  for (EventNode& tf_op_run_event_node : *tf_op_run_event_node_list) {
    tf_op_run_event_node.SetIsEager(tf_op_run_event_node.IsEager());
  }
}
void EventForest::ProcessTfDataSteps() {
  const int64_t tf_data_event_types[] = {
      HostEventType::kTfDataCapturedFunctionRun,
      HostEventType::kTfDataCapturedFunctionRunAsync,
      HostEventType::kTfDataCapturedFunctionRunInstantiated,
      HostEventType::kTfDataCapturedFunctionRunWithBorrowedArgs};
  for (const int64_t tf_data_event_type : tf_data_event_types) {
    auto tf_data_events = gtl::FindOrNull(event_node_map_, tf_data_event_type);
    if (!tf_data_events) continue;
    for (const EventNode& tf_data_event : *tf_data_events) {
      std::optional<XStatVisitor> step_id_stat =
          tf_data_event.GetEventVisitor().GetStat(StatType::kStepId);
      if (!step_id_stat) continue;
      tf_data_step_ids_.insert(step_id_stat->IntValue());
    }
  }
}
void EventForest::ProcessTensorFlowLoop() {
  struct TensorFlowLoopIteration {
    EventNode* first_event = nullptr;
    std::vector<EventNode*> events;
  };
  using TensorFlowLoop =
      absl::flat_hash_map<int64_t , TensorFlowLoopIteration>;
  absl::flat_hash_map<int64_t , TensorFlowLoop> tf_loops;
  auto executor_event_list =
      gtl::FindOrNull(event_node_map_, HostEventType::kExecutorStateProcess);
  if (!executor_event_list) return;
  for (EventNode& executor_event : *executor_event_list) {
    std::optional<XStatVisitor> step_id_stat =
        executor_event.GetEventVisitor().GetStat(StatType::kStepId);
    std::optional<XStatVisitor> iter_num_stat =
        executor_event.GetEventVisitor().GetStat(StatType::kIterNum);
    if (!step_id_stat || !iter_num_stat) continue;
    int64_t step_id = step_id_stat->IntValue();
    if (tf_data_step_ids_.contains(step_id)) continue;
    TensorFlowLoop& tf_loop = tf_loops[step_id];
    TensorFlowLoopIteration& iteration = tf_loop[iter_num_stat->IntValue()];
    if (!iteration.first_event || executor_event < *iteration.first_event) {
      iteration.first_event = &executor_event;
    }
    iteration.events.push_back(&executor_event);
  }
  std::vector<const TensorFlowLoopIteration*> iters;
  for (const auto& step_id_and_tf_loop : tf_loops) {
    const TensorFlowLoop& tf_loop = step_id_and_tf_loop.second;
    if (tf_loop.size() == 1 && tf_loop.contains(0)) continue;
    for (const auto& iter_num_and_iter : tf_loop) {
      iters.push_back(&iter_num_and_iter.second);
    }
  }
  absl::c_sort(iters, [](const auto& iter1, const auto& iter2) {
    return *iter1->first_event < *iter2->first_event;
  });
  for (const TensorFlowLoopIteration* iter : iters) {
    EventNode* root_event = iter->first_event;
    tf_loop_root_events_.push_back(root_event);
    for (EventNode* event : iter->events) {
      if (event == root_event) continue;
      root_event->AddChild(event);
    }
  }
}
void EventForest::AddPlane(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    XPlane* plane) {
  CreateStatMetadata(plane);
  planes_.push_back({plane, visitor_factory(plane)});
}
void EventForest::AddSpace(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    XSpace* space) {
  for (XPlane& plane : *space->mutable_planes()) {
    AddPlane(visitor_factory, &plane);
  }
}
void EventForest::AddPlanes(
    const std::function<XPlaneVisitor(const XPlane*)> visitor_factory,
    const std::vector<XPlane*>& planes) {
  for (XPlane* plane : planes) {
    AddPlane(visitor_factory, plane);
  }
}
void EventForest::ConnectEvents(
    const std::vector<InterThreadConnectInfo>& connect_info_list) {
  ContextGroupMap context_groups;
  for (auto& plane_visitor : planes_) {
    ConnectIntraThread(plane_visitor.first, &plane_visitor.second,
                       &context_groups);
  }
  ConnectInterThread(connect_info_list);
  ConnectContextGroups(context_groups);
}
void EventForest::ConnectTfDataEvents() {
  absl::flat_hash_map<
      std::pair<int64_t , int64_t >,
      std::vector<EventNode*>>
      produce_iterator_map;
  uint64 num_producers = 0;
  for (HostEventType event_type :
       {HostEventType::kPrefetchProduce,
        HostEventType::kParallelInterleaveProduce,
        HostEventType::kParallelMapProduce, HostEventType::kMapAndBatchProduce,
        HostEventType::kParseExampleProduce,
        HostEventType::kParallelBatchProduce}) {
    auto produce_event_list = gtl::FindOrNull(event_node_map_, event_type);
    if (!produce_event_list) continue;
    VLOG(1) << produce_event_list->size() << " "
            << GetHostEventTypeStr(event_type) << " events found.";
    for (EventNode& produce_event : *produce_event_list) {
      std::optional<XStatVisitor> element_id =
          produce_event.GetEventVisitor().GetStat(StatType::kElementId);
      if (!element_id.has_value()) continue;
      for (EventNode* produce_iterator : produce_event.GetChildren()) {
        if (IsIteratorEventType(produce_iterator->GetEventVisitor().Type())) {
          std::optional<XStatVisitor> iterator_id =
              produce_iterator->GetEventVisitor().GetStat(StatType::kParentId);
          if (!iterator_id.has_value()) break;
          produce_iterator_map[{iterator_id->IntValue(),
                                element_id->IntValue()}]
              .push_back(produce_iterator);
          ++num_producers;
          break;
        }
      }
    }
  }
  VLOG(1) << num_producers << " producer iterators found.";
  uint64 num_matched = 0;
  for (HostEventType event_type :
       {HostEventType::kPrefetchConsume,
        HostEventType::kParallelInterleaveConsume,
        HostEventType::kParallelMapConsume, HostEventType::kMapAndBatchConsume,
        HostEventType::kParseExampleConsume,
        HostEventType::kParallelBatchConsume}) {
    auto consume_event_list = gtl::FindOrNull(event_node_map_, event_type);
    if (!consume_event_list) continue;
    VLOG(1) << consume_event_list->size() << " "
            << GetHostEventTypeStr(event_type) << " events found.";
    for (EventNode& consume_event : *consume_event_list) {
      std::optional<XStatVisitor> element_id =
          consume_event.GetEventVisitor().GetStat(StatType::kElementId);
      if (!element_id.has_value()) continue;
      if (consume_event.GetParents().empty()) continue;
      EventNode* consume_iterator = consume_event.GetParents().at(0);
      if (!consume_iterator ||
          !IsIteratorEventType(consume_iterator->GetEventVisitor().Type())) {
        continue;
      }
      std::optional<XStatVisitor> iterator_id =
          consume_iterator->GetEventVisitor().GetStat(StatType::kStepId);
      if (!iterator_id.has_value()) continue;
      if (auto produce_iterators = gtl::FindOrNull(
              produce_iterator_map, std::make_pair(iterator_id->IntValue(),
                                                   element_id->IntValue()))) {
        for (EventNode* produce_iterator : *produce_iterators) {
          consume_iterator->AddChild(produce_iterator);
          ++num_matched;
        }
      }
    }
  }
  VLOG(1) << num_matched << " consumer iterators matched.";
}
void EventForest::GroupEvents() {
  ProcessTfDataSteps();
  ProcessTensorFlowLoop();
  CreateEventGroups();
  MarkEagerlyExecutedGpuKernels();
  MarkEagerlyExecutedCpuTfOps();
}
std::vector<InterThreadConnectInfo> CreateInterThreadConnectInfoList() {
  std::vector<InterThreadConnectInfo> connect_info_list = {
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kExecutorStateProcess,
       HostEventType::kIteratorGetNextAsOptionalOp,
       {StatType::kStepId, StatType::kIterNum}},
      {HostEventType::kKernelLaunch,
       HostEventType::kKernelExecute,
       {StatType::kCorrelationId}}};
  return connect_info_list;
}
void GroupTfEvents(XSpace* space, EventForest* event_forest) {
  if (CheckLoopOp(*space)) {
    return;
  }
  std::vector<InterThreadConnectInfo> connect_info_list =
      CreateInterThreadConnectInfoList();
  event_forest->AddSpace(CreateTfXPlaneVisitor, space);
  event_forest->ConnectEvents(connect_info_list);
  event_forest->GroupEvents();
}
void GroupTfEvents(XSpace* space) {
  EventForest event_forest;
  GroupTfEvents(space, &event_forest);
}
void AddGroupMetadataToStepEvents(const GroupMetadataMap& group_metadata_map,
                                  XLineBuilder& line) {
  if (group_metadata_map.empty()) return;
  XPlaneBuilder* plane = line.Plane();
  const XStatMetadata* group_id_stat_metadata =
      plane->GetStatMetadata(GetStatTypeStr(StatType::kGroupId));
  if (group_id_stat_metadata == nullptr) return;
  const XStatMetadata* step_name_stat_metadata =
      plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  line.ForEachEvent([&](XEventBuilder event) {
    const XStat* group_id_stat = event.GetStat(*group_id_stat_metadata);
    if (group_id_stat != nullptr) {
      int64_t group_id = group_id_stat->int64_value();
      if (const GroupMetadata* group_metadata =
              gtl::FindOrNull(group_metadata_map, group_id)) {
        event.AddStatValue(*step_name_stat_metadata, group_metadata->name);
      }
    }
  });
}
std::optional<int64_t> GetGroupId(const XEventVisitor& event,
                                  const XStatMetadata& group_id_stat_metadata) {
  if (auto group_id_stat =
          event.GetStat(StatType::kGroupId, group_id_stat_metadata)) {
    return group_id_stat->IntValue();
  }
  return std::nullopt;
}
class GroupQueue {
 public:
  GroupQueue(const XPlaneVisitor* plane, const XLine* line,
             const XStatMetadata* group_id_stat_metadata)
      : group_queue_(plane, line),
        group_id_stat_metadata_(group_id_stat_metadata) {}
  std::optional<int64_t> OverlappingGroupId(Timespan timespan) {
    if (!group_event_visitor_ ||
        !group_event_visitor_->GetTimespan().Overlaps(timespan)) {
      group_event_visitor_ = group_queue_.GetOverlappingEvent(timespan);
      if (group_event_visitor_) {
        group_id_ = GetGroupId(*group_event_visitor_, *group_id_stat_metadata_);
      } else {
        group_id_.reset();
      }
    }
    return group_id_;
  }
 private:
  XEventContextTracker group_queue_;
  std::optional<XEventVisitor> group_event_visitor_;
  std::optional<int64_t> group_id_;
  const XStatMetadata* group_id_stat_metadata_;
};
void MergeHostSteps(const XStatMetadata& group_id_stat_metadata,
                    const XPlaneVisitor& plane_visitor,
                    XPlaneBuilder* plane_builder, XLine* step_line) {
  std::optional<int64_t> merged_group_id;
  std::optional<XEventBuilder> merged_step_builder;
  absl::flat_hash_set<const XEvent*> events_to_remove;
  for (XEvent& step_event : *step_line->mutable_events()) {
    XEventVisitor step_visitor(&plane_visitor, step_line, &step_event);
    auto group_id = GetGroupId(step_visitor, group_id_stat_metadata);
    if (!group_id) {
      merged_group_id.reset();
      merged_step_builder.reset();
      events_to_remove.insert(&step_event);
    } else if (merged_group_id != group_id) {
      merged_group_id = group_id;
      merged_step_builder.emplace(step_line, plane_builder, &step_event);
    } else {
      merged_step_builder->SetEndTimestampPs(step_visitor.EndTimestampPs());
      events_to_remove.insert(&step_event);
    }
  }
  if (events_to_remove.size() < step_line->events_size()) {
    RemoveEvents(step_line, events_to_remove);
  }
}
void GroupLine(const XStatMetadata& group_id_stat_metadata,
               const XPlaneVisitor& plane_visitor, const XLine& group_line,
               XPlaneBuilder* plane_builder, XLine* line) {
  GroupQueue group_queue(&plane_visitor, &group_line, &group_id_stat_metadata);
  for (XEvent& event : *line->mutable_events()) {
    XEventBuilder event_builder(line, plane_builder, &event);
    if (auto group_id =
            group_queue.OverlappingGroupId(event_builder.GetTimespan())) {
      event_builder.AddStatValue(group_id_stat_metadata, *group_id);
    }
  }
}
void GroupHostAndPlanes(
    tensorflow::profiler::XSpace* space,
    const std::vector<tensorflow::profiler::XPlane*>& device_traces,
    EventForest* event_forest) {
  std::vector<InterThreadConnectInfo> connect_info_list =
      CreateInterThreadConnectInfoList();
  event_forest->AddSpace(CreateTfXPlaneVisitor, space);
  event_forest->AddPlanes(CreateTfXPlaneVisitor, device_traces);
  event_forest->ConnectEvents(connect_info_list);
  event_forest->GroupEvents();
}
void GroupXplaneEvents(tensorflow::profiler::XPlane* plane,
                       const GroupMetadataMap& group_metadata_map) {
  XLine* module_line = nullptr;
  XLine* step_line = nullptr;
  std::vector<XLine*> other_lines;
  for (XLine& line : *plane->mutable_lines()) {
    if (line.name() == "XLA Modules") {
      module_line = &line;
    } else if (line.name() == "Steps") {
      step_line = &line;
    } else {
      other_lines.push_back(&line);
    }
  }
  if (!module_line) return;
  XPlaneBuilder plane_builder(plane);
  const XStatMetadata* group_id_stat_metadata =
      plane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  const XLine* group_line = module_line;
  if (step_line) {
    bool device_loop = (step_line->events_size() > module_line->events_size());
    if (device_loop) {
      group_line = nullptr;
    } else {  
      if (group_line) {
        GroupLine(*group_id_stat_metadata, plane_visitor, *group_line,
                  &plane_builder, step_line);
        MergeHostSteps(*group_id_stat_metadata, plane_visitor, &plane_builder,
                       step_line);
        XLineBuilder step_line_builder(step_line, &plane_builder);
        AddGroupMetadataToStepEvents(group_metadata_map, step_line_builder);
      }
    }
  }
  if (group_line) {
    for (XLine* line : other_lines) {
      GroupLine(*group_id_stat_metadata, plane_visitor, *group_line,
                &plane_builder, line);
    }
  }
}
void GroupTpuEventsOSS(
    tensorflow::profiler::XSpace* space,
    const std::vector<tensorflow::profiler::XPlane*>& device_traces,
    EventForest* event_forest) {
  if (CheckLoopOp(*space)) {
    return;
  }
  GroupHostAndPlanes(space, device_traces, event_forest);
  if (device_traces.empty()) return;
  const GroupMetadataMap& group_metadata_map =
      event_forest->GetGroupMetadataMap();
  std::vector<std::unique_ptr<Thread>> threads;
  ThreadOptions thread_options;
  threads.reserve(device_traces.size());
  for (XPlane* plane : device_traces) {
    threads.emplace_back(Env::Default()->StartThread(
        thread_options, "group_xplane_events",
        absl::bind_front(GroupXplaneEvents, plane,
                         std::ref(group_metadata_map))));
  }
}
}  
}  