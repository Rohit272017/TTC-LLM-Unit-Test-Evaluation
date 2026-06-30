#include "tensorflow/core/profiler/convert/xplane_to_memory_profile.h"
#include <algorithm>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
namespace tensorflow {
namespace profiler {
namespace {
constexpr int64_t kInvalidStepId = -1;
using IndexMetaPair =
    std::pair<int64_t , const MemoryActivityMetadata*>;
bool IsMemoryAllocation(int64_t event_type) {
  return event_type == HostEventType::kMemoryAllocation;
}
bool IsMemoryDeallocation(int64_t event_type) {
  return event_type == HostEventType::kMemoryDeallocation;
}
void UpdateProfileSummary(const MemoryAggregationStats& stats,
                          int64_t time_offset_ps,
                          MemoryProfileSummary* summary) {
  summary->set_peak_bytes_usage_lifetime(stats.peak_bytes_in_use());
  MemoryAggregationStats* peak_stats = summary->mutable_peak_stats();
  if (stats.stack_reserved_bytes() + stats.heap_allocated_bytes() >=
      peak_stats->peak_bytes_in_use()) {
    *peak_stats = stats;
    peak_stats->set_peak_bytes_in_use(stats.stack_reserved_bytes() +
                                      stats.heap_allocated_bytes());
    summary->set_peak_stats_time_ps(time_offset_ps);
    summary->set_memory_capacity(stats.stack_reserved_bytes() +
                                 stats.heap_allocated_bytes() +
                                 stats.free_memory_bytes());
  }
}
MemoryProfile GenerateMemoryProfile(const XPlane* host_trace) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  MemoryProfile memory_profile;
  plane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      int64_t event_type =
          event.Type().value_or(HostEventType::kUnknownHostEventType);
      if (!(IsMemoryAllocation(event_type) ||
            IsMemoryDeallocation(event_type))) {
        return;
      }
      MemoryAggregationStats stats;
      MemoryActivityMetadata metadata;
      if (IsMemoryAllocation(event_type)) {
        metadata.set_memory_activity(ALLOCATION);
      } else if (IsMemoryDeallocation(event_type)) {
        metadata.set_memory_activity(DEALLOCATION);
      }
      metadata.set_step_id(kInvalidStepId);
      std::string memory_id;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        switch (stat.Type().value()) {
          case StatType::kIndexOnHost:
          case StatType::kDeviceOrdinal:
            memory_id = absl::StrCat(stat.IntValue());
            break;
          case StatType::kAllocatorName:
            memory_id = std::string(stat.StrOrRefValue());
            break;
          case StatType::kBytesReserved:
            stats.set_stack_reserved_bytes(stat.IntValue());
            break;
          case StatType::kBytesAllocated:
            stats.set_heap_allocated_bytes(stat.IntValue());
            break;
          case StatType::kBytesAvailable:
            stats.set_free_memory_bytes(stat.IntValue());
            break;
          case StatType::kFragmentation:
            stats.set_fragmentation(stat.DoubleValue());
            break;
          case StatType::kPeakBytesInUse:
            stats.set_peak_bytes_in_use(stat.IntValue());
            break;
          case StatType::kRequestedBytes:
            metadata.set_requested_bytes(stat.IntValue());
            break;
          case StatType::kAllocationBytes:
            metadata.set_allocation_bytes(stat.IntValue());
            break;
          case StatType::kAddress:
            metadata.set_address(stat.IntValue());
            break;
          case StatType::kTfOp:
            metadata.set_tf_op_name(std::string(stat.StrOrRefValue()));
            break;
          case StatType::kGroupId:
            metadata.set_step_id(stat.IntValue());
            break;
          case StatType::kRegionType:
            metadata.set_region_type(std::string(stat.StrOrRefValue()));
            break;
          case StatType::kDataType:
            metadata.set_data_type(tensorflow::DataTypeString(
                static_cast<tensorflow::DataType>(stat.IntValue())));
            break;
          case StatType::kTensorShapes:
            metadata.set_tensor_shape(std::string(stat.StrOrRefValue()));
            break;
        }
      });
      MemoryProfileSummary* summary =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .mutable_profile_summary();
      UpdateProfileSummary(stats, event.OffsetPs(), summary);
      MemoryProfileSnapshot* snapshot =
          (*memory_profile.mutable_memory_profile_per_allocator())[memory_id]
              .add_memory_profile_snapshots();
      snapshot->set_time_offset_ps(event.OffsetPs());
      *snapshot->mutable_aggregation_stats() = std::move(stats);
      *snapshot->mutable_activity_metadata() = std::move(metadata);
    });
  });
  return memory_profile;
}
void UpdateStepId(PerAllocatorMemoryProfile* memory_profile) {
  int64_t last_valid_step_id = -1;
  for (auto& snapshot : *memory_profile->mutable_memory_profile_snapshots()) {
    DCHECK(snapshot.has_activity_metadata());
    if (snapshot.mutable_activity_metadata()->step_id() == kInvalidStepId) {
      snapshot.mutable_activity_metadata()->set_step_id(last_valid_step_id + 1);
    } else {
      last_valid_step_id = snapshot.mutable_activity_metadata()->step_id();
    }
  }
}
void UpdateDeallocation(PerAllocatorMemoryProfile* memory_profile) {
  absl::flat_hash_map<uint64 , const MemoryActivityMetadata*>
      addr_metadata_map;
  for (auto& snapshot : *memory_profile->mutable_memory_profile_snapshots()) {
    uint64 address = snapshot.activity_metadata().address();
    if (snapshot.activity_metadata().memory_activity() == DEALLOCATION) {
      if (addr_metadata_map.contains(address)) {
        const MemoryActivityMetadata* alloc_meta = addr_metadata_map[address];
        snapshot.mutable_activity_metadata()->set_tf_op_name(
            alloc_meta->tf_op_name());
        snapshot.mutable_activity_metadata()->set_region_type(
            alloc_meta->region_type());
        snapshot.mutable_activity_metadata()->set_data_type(
            alloc_meta->data_type());
        snapshot.mutable_activity_metadata()->set_tensor_shape(
            alloc_meta->tensor_shape());
        addr_metadata_map.erase(address);
      } else {
        VLOG(2)
            << "Can't find matching memory allocation for this deallocation: "
            << snapshot.DebugString();
      }
    } else if (!addr_metadata_map.contains(address)) {  
      addr_metadata_map[address] = &snapshot.activity_metadata();
    } else {
      VLOG(2) << "There are two allocations recorded for the same address: "
              << address
              << ". The later allocation event is: " << snapshot.DebugString();
    }
  }
  VLOG(2) << "Number of allocations that cannot find matching dealloctions: "
          << addr_metadata_map.size();
}
int64_t GetPeakMemoryStep(int64_t peak_bytes_profile,
                          const PerAllocatorMemoryProfile* memory_profile) {
  int64_t peak_bytes_profile_step_id = 0;
  for (const auto& snapshot : memory_profile->memory_profile_snapshots()) {
    if (peak_bytes_profile ==
        snapshot.aggregation_stats().heap_allocated_bytes() +
            snapshot.aggregation_stats().stack_reserved_bytes()) {
      DCHECK(snapshot.has_activity_metadata());
      peak_bytes_profile_step_id = snapshot.activity_metadata().step_id();
    }
  }
  return peak_bytes_profile_step_id;
}
struct MetadataComparator {
  bool operator()(const IndexMetaPair& a, const IndexMetaPair& b) const {
    const MemoryActivityMetadata* a_meta = a.second;
    const MemoryActivityMetadata* b_meta = b.second;
    DCHECK_NE(a_meta, nullptr);
    DCHECK_NE(b_meta, nullptr);
    auto lhs =
        std::make_tuple(-a_meta->allocation_bytes(), -a_meta->requested_bytes(),
                        a_meta->tf_op_name(), a_meta->region_type(),
                        a_meta->data_type(), a_meta->tensor_shape());
    auto rhs =
        std::make_tuple(-b_meta->allocation_bytes(), -b_meta->requested_bytes(),
                        b_meta->tf_op_name(), b_meta->region_type(),
                        b_meta->data_type(), b_meta->tensor_shape());
    return lhs < rhs;
  }
};
void InsertSpecialAllocations(int64_t unmapped_allocation_bytes,
                              int64_t step_id,
                              PerAllocatorMemoryProfile* memory_profile,
                              std::vector<IndexMetaPair>* active_allocs) {
  int index = 0;
  if (unmapped_allocation_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    special_allocation->set_memory_activity(ALLOCATION);
    special_allocation->set_requested_bytes(unmapped_allocation_bytes);
    special_allocation->set_allocation_bytes(unmapped_allocation_bytes);
    special_allocation->set_address(0);
    special_allocation->set_tf_op_name("unused preallocated device memory");
    special_allocation->set_step_id(step_id);
    special_allocation->set_region_type("persist/dynamic");
    special_allocation->set_data_type(
        tensorflow::DataTypeString(static_cast<tensorflow::DataType>(0)));
    special_allocation->set_tensor_shape("unknown");
    active_allocs->push_back({--index, special_allocation});
  }
  int64_t stack_bytes =
      memory_profile->profile_summary().peak_stats().stack_reserved_bytes();
  if (stack_bytes > 0) {
    MemoryActivityMetadata* special_allocation =
        memory_profile->add_special_allocations();
    special_allocation->set_memory_activity(ALLOCATION);
    special_allocation->set_requested_bytes(stack_bytes);
    special_allocation->set_allocation_bytes(stack_bytes);
    special_allocation->set_address(0);
    special_allocation->set_tf_op_name("stack");
    special_allocation->set_step_id(step_id);
    special_allocation->set_region_type("stack");
    special_allocation->set_data_type(
        tensorflow::DataTypeString(static_cast<tensorflow::DataType>(0)));
    special_allocation->set_tensor_shape("unknown");
    active_allocs->push_back({--index, special_allocation});
  }
}
bool operator==(const IndexMetaPair& a, const IndexMetaPair& b) {
  const MemoryActivityMetadata* a_meta = a.second;
  const MemoryActivityMetadata* b_meta = b.second;
  return a_meta->allocation_bytes() == b_meta->allocation_bytes() &&
         a_meta->requested_bytes() == b_meta->requested_bytes() &&
         a_meta->tf_op_name() == b_meta->tf_op_name() &&
         a_meta->region_type() == b_meta->region_type() &&
         a_meta->data_type() == b_meta->data_type() &&
         a_meta->tensor_shape() == b_meta->tensor_shape();
}
void ProcessActiveAllocations(int64_t peak_bytes_profile_step_id,
                              PerAllocatorMemoryProfile* memory_profile) {
  int64_t unmapped_allocation_bytes =
      memory_profile->profile_summary().peak_stats().heap_allocated_bytes();
  int64_t unmapped_deallocation_bytes = 0;
  absl::flat_hash_map<int64_t , IndexMetaPair> active_alloc_map;
  for (int i = 0; i < memory_profile->memory_profile_snapshots_size(); i++) {
    const auto& snapshot = memory_profile->memory_profile_snapshots().at(i);
    DCHECK(snapshot.has_activity_metadata());
    const MemoryActivityMetadata& metadata = snapshot.activity_metadata();
    if (snapshot.time_offset_ps() >
        memory_profile->profile_summary().peak_stats_time_ps())
      break;
    if (metadata.step_id() != peak_bytes_profile_step_id) continue;
    if (metadata.memory_activity() == ALLOCATION) {
      active_alloc_map[metadata.address()] = {i, &metadata};
      unmapped_allocation_bytes -= metadata.allocation_bytes();
    } else {
      DCHECK_EQ(metadata.memory_activity(), DEALLOCATION);
      if (active_alloc_map.contains(metadata.address())) {
        active_alloc_map.erase(metadata.address());
      } else {
        unmapped_deallocation_bytes += metadata.allocation_bytes();
      }
      unmapped_allocation_bytes += metadata.allocation_bytes();
    }
  }
  unmapped_allocation_bytes -= unmapped_deallocation_bytes;
  VLOG(2) << "unmapped_allocation_bytes=" << unmapped_allocation_bytes
          << ", unmapped_deallocation_bytes=" << unmapped_deallocation_bytes;
  std::vector<IndexMetaPair> active_allocs;
  for (const auto& address_and_index_meta : active_alloc_map) {
    active_allocs.push_back(address_and_index_meta.second);
  }
  InsertSpecialAllocations(unmapped_allocation_bytes,
                           peak_bytes_profile_step_id, memory_profile,
                           &active_allocs);
  std::sort(active_allocs.begin(), active_allocs.end(), MetadataComparator());
  for (int i = 0, end = active_allocs.size(); i < end; i++) {
    ActiveAllocation* allocation = memory_profile->add_active_allocations();
    allocation->set_snapshot_index(active_allocs[i].first);
    if (active_allocs[i].first < 0) {
      allocation->set_special_index(-active_allocs[i].first - 1);
    } else {
      allocation->set_special_index(-1);
    }
    allocation->set_num_occurrences(1);
    const int last_alloc = active_allocs.size() - 1;
    while (i < last_alloc && active_allocs[i] == active_allocs[i + 1]) {
      allocation->set_num_occurrences(allocation->num_occurrences() + 1);
      i++;
    }
  }
  VLOG(2) << "Distinctive active allocation count="
          << memory_profile->active_allocations_size();
}
void SaveActiveAllocationSnapshots(
    protobuf::RepeatedPtrField<MemoryProfileSnapshot>* snapshots,
    protobuf::RepeatedPtrField<ActiveAllocation>* active_allocations) {
  std::vector<MemoryProfileSnapshot*> samples;
  for (const auto& allocation : *active_allocations) {
    auto orig_index = allocation.snapshot_index();
    if (orig_index < 0) continue;
    samples.push_back(&(*snapshots)[orig_index]);
  }
  int new_index = 0;
  for (auto& allocation : *active_allocations) {
    int64_t origin_index = allocation.snapshot_index();
    if (origin_index < 0) continue;
    allocation.set_snapshot_index(new_index);
    new_index++;
  }
  protobuf::RepeatedPtrField<MemoryProfileSnapshot> new_snapshots;
  new_snapshots.Reserve(samples.size());
  for (const auto& sample : samples) {
    *new_snapshots.Add() = std::move(*sample);
  }
  *snapshots = std::move(new_snapshots);
}
void SampleMemoryProfileTimeline(int64_t max_num_snapshots,
                                 PerAllocatorMemoryProfile* memory_profile) {
  const protobuf::RepeatedPtrField<MemoryProfileSnapshot>& original_snapshots =
      memory_profile->memory_profile_snapshots();
  protobuf::RepeatedPtrField<MemoryProfileSnapshot>* timeline_snapshots =
      memory_profile->mutable_sampled_timeline_snapshots();
  int64_t snapshot_count = original_snapshots.size();
  if (snapshot_count > max_num_snapshots) {
    auto max_box_filter = [&](int filter_width, int count, int start) {
      for (int i = 0; i < count; i++) {
        const MemoryProfileSnapshot* max_snapshot =
            &original_snapshots[start + filter_width * i];
        int64_t max_bytes =
            max_snapshot->aggregation_stats().heap_allocated_bytes() +
            max_snapshot->aggregation_stats().stack_reserved_bytes();
        for (int index = start + filter_width * i + 1;
             index < start + filter_width * (i + 1); index++) {
          int64_t bytes = original_snapshots[index]
                              .aggregation_stats()
                              .heap_allocated_bytes() +
                          original_snapshots[index]
                              .aggregation_stats()
                              .stack_reserved_bytes();
          if (bytes > max_bytes) {
            max_snapshot = &original_snapshots[index];
            max_bytes = bytes;
          }
        }
        *timeline_snapshots->Add() = *max_snapshot;
      }
    };
    int width = snapshot_count / max_num_snapshots;
    int count1 = max_num_snapshots * (width + 1) - snapshot_count;
    int count2 = max_num_snapshots - count1;
    max_box_filter(width, count1, 0);
    max_box_filter(width + 1, count2, width * count1);
  } else {
    *timeline_snapshots = original_snapshots;
  }
}
void ProcessMemoryProfileProto(int64_t max_num_snapshots,
                               MemoryProfile* memory_profile) {
  memory_profile->set_num_hosts(1);
  for (const auto& id_and_allocator_profile :
       memory_profile->memory_profile_per_allocator()) {
    if (!id_and_allocator_profile.second.memory_profile_snapshots().empty()) {
      memory_profile->add_memory_ids(id_and_allocator_profile.first);
    }
  }
  absl::c_sort(*memory_profile->mutable_memory_ids());
  for (auto& id_and_allocator_profile :
       *memory_profile->mutable_memory_profile_per_allocator()) {
    PerAllocatorMemoryProfile* allocator_memory_profile =
        &id_and_allocator_profile.second;
    protobuf::RepeatedPtrField<MemoryProfileSnapshot>* snapshots =
        allocator_memory_profile->mutable_memory_profile_snapshots();
    absl::c_sort(*snapshots, [](const MemoryProfileSnapshot& a,
                                const MemoryProfileSnapshot& b) {
      return a.time_offset_ps() < b.time_offset_ps();
    });
    UpdateStepId(allocator_memory_profile);
    UpdateDeallocation(allocator_memory_profile);
    SampleMemoryProfileTimeline(max_num_snapshots, allocator_memory_profile);
    int64_t peak_step_id =
        GetPeakMemoryStep(allocator_memory_profile->profile_summary()
                              .peak_stats()
                              .peak_bytes_in_use(),
                          allocator_memory_profile);
    ProcessActiveAllocations(peak_step_id, allocator_memory_profile);
    SaveActiveAllocationSnapshots(
        snapshots, allocator_memory_profile->mutable_active_allocations());
  }
}
template <typename Proto>
Status ConvertProtoToJson(const Proto& proto_output, std::string* json_output) {
  protobuf::util::JsonPrintOptions json_options;
  json_options.always_print_primitive_fields = true;
  auto status = protobuf::util::MessageToJsonString(proto_output, json_output,
                                                    json_options);
  if (!status.ok()) {
    auto error_msg = status.message();
    return errors::Internal(
        "Could not convert proto to JSON string: ",
        absl::string_view(error_msg.data(), error_msg.length()));
  }
  return absl::OkStatus();
}
}  
MemoryProfile ConvertXPlaneToMemoryProfile(const XPlane& host_plane,
                                           int64_t max_num_snapshots) {
  MemoryProfile memory_profile = GenerateMemoryProfile(&host_plane);
  ProcessMemoryProfileProto(max_num_snapshots, &memory_profile);
  memory_profile.set_version(1);
  return memory_profile;
}
Status ConvertXSpaceToMemoryProfileJson(const XSpace& xspace,
                                        std::string* json_output) {
  if (const XPlane* host_plane =
          FindPlaneWithName(xspace, kHostThreadsPlaneName)) {
    MemoryProfile memory_profile = ConvertXPlaneToMemoryProfile(*host_plane);
    TF_RETURN_IF_ERROR(ConvertProtoToJson(memory_profile, json_output));
  }
  return absl::OkStatus();
}
}  
}  