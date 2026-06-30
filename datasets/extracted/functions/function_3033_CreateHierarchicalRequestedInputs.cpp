#include "arolla/codegen/io/multi_loader.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>
#include "absl/log/check.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
namespace arolla::codegen::io {
namespace multi_loader_internal {
void CreateHierarchicalRequestedInputs(
    const std::vector<std::optional<TypedSlot>>& leaf_slots,
    const std::vector<std::vector<size_t>>& tree,
    HierarchicalRequestedInputsDataView output) {
  CHECK_LT(leaf_slots.size(), 1 << 16)
      << "Too many input leaves for generated code";
  std::vector<size_t> leaf_frame_offsets;
  std::vector<char> node_requested;
  node_requested.resize(tree.size(), false);
  for (size_t node_id = 0; node_id != tree.size(); ++node_id) {
    const std::vector<size_t>& children = tree[node_id];
    if (children.empty()) {
      size_t leaf_id = leaf_frame_offsets.size();
      const std::optional<TypedSlot>& slot = leaf_slots[leaf_id];
      size_t offset = slot.has_value() ? slot->byte_offset() : kSkippedOffset;
      leaf_frame_offsets.push_back(offset);
      node_requested[node_id] = slot.has_value();
    } else {
      node_requested[node_id] = false;
      for (size_t child : children) {
        CHECK_LT(child, node_id);
        node_requested[node_id] |= node_requested[child];
      }
    }
  }
  std::copy(leaf_frame_offsets.begin(), leaf_frame_offsets.end(),
            output.leaf_frame_offsets.begin());
  size_t intermediate_id = 0;
  for (size_t i = 0; i != tree.size(); ++i) {
    if (tree[i].empty()) {  
      continue;
    }
    CHECK_LT(intermediate_id, output.node_requested.size());
    output.node_requested[intermediate_id++] = node_requested[i];
  }
}
void CreateHierarchicalSingleValueRequestedInputs(
    const std::vector<std::optional<TypedSlot>>& leaf_slots,
    const std::vector<size_t>& size_leaves,
    const std::vector<std::vector<size_t>>& tree,
    HierarchicalSingleValueRequestedInputsDataView output) {
  CHECK_LT(leaf_slots.size(), 1 << 16)
      << "Too many input leaves for generated code";
  std::vector<HierarchicalSingleValueClearInfo> node_optional_clear_infos;
  std::vector<HierarchicalSingleValueClearInfo> node_size_clear_infos;
  std::vector<size_t> presence_offsets;
  std::vector<size_t> size_offsets;
  node_optional_clear_infos.resize(tree.size(),
                                   HierarchicalSingleValueClearInfo{});
  node_size_clear_infos.resize(tree.size(), HierarchicalSingleValueClearInfo{});
  size_t leaf_id = 0;
  for (size_t node_id = 0; node_id != tree.size(); ++node_id) {
    const std::vector<size_t>& children = tree[node_id];
    auto& node_optional_clear_info = node_optional_clear_infos[node_id];
    auto& node_size_clear_info = node_size_clear_infos[node_id];
    if (children.empty()) {  
      const std::optional<TypedSlot>& slot = leaf_slots[leaf_id];
      size_t offset = slot.has_value() ? slot->byte_offset() : kSkippedOffset;
      node_optional_clear_info.range_begin = presence_offsets.size();
      node_size_clear_info.range_begin = size_offsets.size();
      if (offset != kSkippedOffset) {
        if (std::binary_search(size_leaves.begin(), size_leaves.end(),
                               leaf_id)) {
          size_offsets.push_back(offset);
        } else if (::arolla::IsOptionalQType(slot->GetType())) {
          presence_offsets.push_back(offset);
        }
      }
      node_optional_clear_info.range_end = presence_offsets.size();
      node_size_clear_info.range_end = size_offsets.size();
      ++leaf_id;
    } else {
      node_optional_clear_info.range_begin =
          node_optional_clear_infos[children.front()].range_begin;
      node_optional_clear_info.range_end =
          node_optional_clear_infos[children.back()].range_end;
      node_size_clear_info.range_begin =
          node_size_clear_infos[children.front()].range_begin;
      node_size_clear_info.range_end =
          node_size_clear_infos[children.back()].range_end;
    }
  }
  CHECK_GE(output.requested_offsets.size(),
           presence_offsets.size() + size_offsets.size());
  std::copy(presence_offsets.begin(), presence_offsets.end(),
            output.requested_offsets.begin());
  std::copy(size_offsets.begin(), size_offsets.end(),
            output.requested_offsets.begin() + presence_offsets.size());
  std::fill(output.requested_offsets.begin() + presence_offsets.size() +
                size_offsets.size(),
            output.requested_offsets.end(), kSkippedOffset);
  size_t leaf_count = 0;
  for (size_t i = 0; i != tree.size(); ++i) {
    if (tree[i].empty()) {  
      ++leaf_count;
      continue;
    }
    size_t intermediate_id = i - leaf_count;
    output.node_optional_clear_infos[intermediate_id] =
        node_optional_clear_infos[i];
    output.node_size_clear_infos[intermediate_id] = node_size_clear_infos[i];
    output.node_size_clear_infos[intermediate_id].range_begin +=
        presence_offsets.size();
    output.node_size_clear_infos[intermediate_id].range_end +=
        presence_offsets.size();
  }
}
}  
}  