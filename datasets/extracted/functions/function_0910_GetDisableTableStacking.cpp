#include "tensorflow/core/tpu/kernels/sparse_core_layout.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"
#include "tsl/platform/stringpiece.h"
namespace tensorflow {
ABSL_ATTRIBUTE_WEAK bool GetDisableTableStacking(bool disable_table_stacking) {
  bool should_disable_stacking = false;
  XlaSparseCoreFlags *sparse_core_flags = GetXlaSparseCoreFlags();
  should_disable_stacking =
      sparse_core_flags->tf_xla_sparse_core_disable_table_stacking;
  return should_disable_stacking || disable_table_stacking;
}
ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingMemLimit() {
  XlaSparseCoreFlags *sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_mem_limit_bytes;
}
ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingTableShardLimit() {
  XlaSparseCoreFlags *sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_table_shard_limit_bytes;
}
namespace tpu {
static int64_t NextLargestMultiple(int64_t n, int64_t factor) {
  int64_t extra = n % factor;
  if (extra == 0) return n;
  return n + factor - extra;
}
SparseCoreLayoutStacker::SparseCoreLayoutStacker(int num_partitions,
                                                 bool disable_table_stacking,
                                                 int sparse_cores_per_partition)
    : num_partitions_(num_partitions),
      sparse_cores_per_partition_(sparse_cores_per_partition),
      num_sparse_cores_(num_partitions_ * sparse_cores_per_partition_),
      stacking_enabled_(!GetDisableTableStacking(disable_table_stacking)),
      activation_mem_bytes_limit_(GetXlaSparseCoreStackingMemLimit()),
      variable_shard_bytes_limit_(GetXlaSparseCoreStackingTableShardLimit()) {}
absl::Status SparseCoreLayoutStacker::AddTable(absl::string_view table_name,
                                               int64_t table_height,
                                               int64_t table_width,
                                               absl::string_view group,
                                               int64_t output_samples) {
  if (stacks_by_group_.empty()) {  
    VLOG(1) << "Stacking parameters: stacking_enabled_ = " << stacking_enabled_
            << ", activation_mem_bytes_limit_ = " << activation_mem_bytes_limit_
            << ", variable_shard_bytes_limit_ = " << variable_shard_bytes_limit_
            << ", row_limit_ = " << row_limit_
            << ", table_limit_ = " << table_limit_;
  }
  VLOG(2) << "Table " << table_name << ":";
  int64_t samples_per_sparse_core =
      output_samples / sparse_cores_per_partition_;
  int64_t padded_width = NextLargestMultiple(table_width, 8);
  int64_t padded_height =
      NextLargestMultiple(table_height, num_sparse_cores_ * 8);
  VLOG(2) << "  Original size: " << table_height << "x" << table_width
          << " padded size: " << padded_height << "x" << padded_width;
  int64_t activation_mem_bytes =
      sizeof(float) * padded_width * samples_per_sparse_core;
  int64_t variable_shard_bytes =
      sizeof(float) * padded_width * padded_height / num_partitions_;
  VLOG(2) << "  activation mem = " << activation_mem_bytes
          << ", variable shard bytes = " << variable_shard_bytes;
  std::vector<TableStack> &candidate_stacks =
      stacks_by_group_[std::make_pair(padded_width, std::string(group))];
  TableStack *stack = nullptr;  
  if (stacking_enabled_) {
    for (TableStack &ts : candidate_stacks) {
      if (ts.incomplete_tables.size() >= table_limit_) continue;
      if (activation_mem_bytes_limit_ != 0 &&
          ts.total_activation_mem_bytes + activation_mem_bytes >=
              activation_mem_bytes_limit_) {
        continue;
      }
      if (variable_shard_bytes_limit_ != 0 &&
          ts.total_variable_shard_bytes + variable_shard_bytes >=
              variable_shard_bytes_limit_) {
        continue;
      }
      if (row_limit_ != 0 &&
          ts.unsharded_height + padded_height >= row_limit_) {
        continue;
      }
      stack = &ts;
      break;
    }
  }
  if (stack == nullptr) {
    candidate_stacks.emplace_back();
    stack = &candidate_stacks.back();
    stack->padded_width = padded_width;
    stack->temporary_name = absl::Substitute("w$0_i$1_$2", padded_width,
                                             candidate_stacks.size(), group);
  }
  stack->incomplete_tables.emplace_back();
  SparseCoreTableLayout &layout = stack->incomplete_tables.back();
  layout.set_table_name(std::string(table_name));
  layout.set_num_sparse_cores(num_sparse_cores_);
  layout.set_num_partitions(num_partitions_);
  layout.add_unsharded_shape(table_height);
  layout.add_unsharded_shape(table_width);
  layout.add_unsharded_padded_shape(padded_height);
  layout.add_unsharded_padded_shape(padded_width);
  layout.set_sparse_core_shard_row_offset(stack->unsharded_height /
                                          num_sparse_cores_);
  layout.set_sparse_core_shard_rotation(((stack->incomplete_tables.size() - 1) *
                                         num_sparse_cores_ / num_partitions_) %
                                        num_sparse_cores_);
  stack->unsharded_height += padded_height;
  stack->total_variable_shard_bytes += variable_shard_bytes;
  stack->total_activation_mem_bytes += activation_mem_bytes;
  return absl::OkStatus();
}
absl::StatusOr<SparseCoreTableLayouts> SparseCoreLayoutStacker::GetLayouts() {
  SparseCoreTableLayouts layouts;
  for (const auto &[key, stacks] : stacks_by_group_) {
    VLOG(1) << "Stack group: padded width " << key.first
            << ", name = " << key.second;
    for (const TableStack &stack : stacks) {
      VLOG(1) << "  Stack " << stack.temporary_name
              << ": unsharded_height = " << stack.unsharded_height
              << ", total_activation_mem_bytes = "
              << stack.total_activation_mem_bytes
              << ", total_variable_shard_bytes = "
              << stack.total_variable_shard_bytes;
      std::string stacked_table_name;
      for (const SparseCoreTableLayout &incomplete_layout :
           stack.incomplete_tables) {
        if (!stacked_table_name.empty()) stacked_table_name += "_";
        absl::StrAppend(&stacked_table_name, incomplete_layout.table_name());
      }
      for (const SparseCoreTableLayout &incomplete_layout :
           stack.incomplete_tables) {
        SparseCoreTableLayout *out_layout = layouts.add_tables();
        *out_layout = incomplete_layout;
        out_layout->set_stacked_table_name(stacked_table_name);
        VLOG(1) << "    Contains " << out_layout->table_name();
        out_layout->set_total_rows_per_sparse_core_shard(
            stack.unsharded_height / num_sparse_cores_);
      }
    }
  }
  return layouts;
}
}  
}  