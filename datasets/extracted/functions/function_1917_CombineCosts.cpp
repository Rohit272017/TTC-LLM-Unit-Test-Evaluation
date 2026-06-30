#include "tensorflow/core/grappler/costs/cost_estimator.h"
namespace tensorflow {
namespace grappler {
Costs CombineCosts(const Costs& left, const Costs& right) {
  CHECK_NE(left.max_memory, kMemoryUnknown);
  CHECK_NE(left.max_per_op_buffers, kMemoryUnknown);
  CHECK_NE(left.max_per_op_streaming, kMemoryUnknown);
  Costs result = left;
  result.execution_time += right.execution_time;
  result.compute_time += right.compute_time;
  result.memory_time += right.memory_time;
  result.network_time += right.network_time;
  result.intermediate_memory_time += right.intermediate_memory_time;
  result.intermediate_memory_read_time += right.intermediate_memory_read_time;
  result.intermediate_memory_write_time += right.intermediate_memory_write_time;
  if (right.max_per_op_buffers != kMemoryUnknown) {
    result.max_per_op_buffers =
        std::max(left.max_per_op_buffers, right.max_per_op_buffers);
  }
  if (right.max_per_op_streaming != kMemoryUnknown) {
    result.max_per_op_streaming =
        std::max(left.max_per_op_streaming, right.max_per_op_streaming);
  }
  result.num_ops_total += right.num_ops_total;
  if (right.inaccurate) {
    result.inaccurate = true;
  }
  result.num_ops_with_unknown_shapes += right.num_ops_with_unknown_shapes;
  if (right.max_memory != kMemoryUnknown) {
    result.max_memory += right.max_memory;
  }
  return result;
}
Costs MultiplyCosts(const Costs& costs, int multiplier) {
  CHECK_GE(multiplier, 0);
  if (multiplier == 0) {
    return Costs::ZeroCosts();
  }
  if (multiplier == 1) {
    return costs;
  }
  Costs result = costs;
  result.execution_time *= multiplier;
  result.compute_time *= multiplier;
  result.memory_time *= multiplier;
  result.network_time *= multiplier;
  result.intermediate_memory_time *= multiplier;
  result.intermediate_memory_read_time *= multiplier;
  result.intermediate_memory_write_time *= multiplier;
  return result;
}
}  
}  