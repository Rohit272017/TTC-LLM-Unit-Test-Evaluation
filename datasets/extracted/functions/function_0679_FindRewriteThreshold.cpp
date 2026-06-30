#ifndef TENSORFLOW_CORE_UTIL_MKL_HEURISTICS_H_
#define TENSORFLOW_CORE_UTIL_MKL_HEURISTICS_H_
#ifdef INTEL_MKL
#include <vector>
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/cpu_info.h"
namespace tensorflow {
struct RewriteThreshold {
  std::string op;
  int cpu_family;
  int cpu_model_num;
  struct PerformanceParameters {
    double thread_sync_cost;
    double framework_cost;
  } params;
};
static const RewriteThreshold rewrite_thresholds[] = {
#ifdef DNNL_AARCH64_USE_ACL
    {"Conv2D", 0x41, 0xd40, {0.9349, 22.603}},
    {"_FusedConv2D", 0x41, 0xd40, {0.9349, 22.603}},
    {"FusedBatchNormV3", 0x41, 0xd40, {0.3223, -0.8822}},
    {"Sigmoid", 0x41, 0xd40, {0.0, 0.064736}},
#endif  
    {"", 0x0, 0x0, {0, 0}}};
static double FindRewriteThreshold(const string node_name, int threads) {
  int cpu_family_ = tsl::port::CPUFamily();
  int cpu_model_num_ = tsl::port::CPUModelNum();
  if (threads == 0) {
    return 0;
  }
  for (const RewriteThreshold* i = rewrite_thresholds;
       i->op != "" && threads > 0; i++) {
    if (node_name == i->op && cpu_family_ == i->cpu_family &&
        cpu_model_num_ == i->cpu_model_num) {
      return i->params.thread_sync_cost * threads + i->params.framework_cost;
    }
  }
  return 0;
}
static double CalculateNodeMFlops(const AttrSlice& attrs,
                                  const string node_name) {
  std::vector<const TensorShapeProto*> shape_attrs;
  if (!TryGetNodeAttr(attrs, "_input_shapes", &shape_attrs)) {
    return -1;
  }
  if ((node_name == "Conv2D" || node_name == "_FusedConv2D") &&
      shape_attrs.size() == 2) {
    TensorShape input_shape, filter_shape;
    if (TensorShape::BuildTensorShape(*shape_attrs[0], &input_shape) !=
        tsl::OkStatus()) {
      return -1;
    }
    if (TensorShape::BuildTensorShape(*shape_attrs[1], &filter_shape) !=
        tsl::OkStatus()) {
      return -1;
    }
    return input_shape.dim_size(0) * input_shape.dim_size(1) *
           input_shape.dim_size(2) * input_shape.dim_size(3) *
           filter_shape.dim_size(0) * filter_shape.dim_size(1) *
           filter_shape.dim_size(3) / (double)1e6;
  } else if ((node_name == "FusedBatchNormV3" || node_name == "Sigmoid") &&
             shape_attrs.size() >= 1) {
    TensorShape input_shape;
    if (TensorShape::BuildTensorShape(*shape_attrs[0], &input_shape) !=
        tsl::OkStatus()) {
      return -1;
    }
    return input_shape.dim_size(0) * input_shape.dim_size(1) *
           input_shape.dim_size(2) * input_shape.dim_size(3) / (double)1e6;
  }
  return -1;
}
static bool MatMulHeuristic(const Node* n) {
  if (!tsl::port::TestAarch64CPU(tsl::port::Aarch64CPU::ARM_NEOVERSE_V1)) {
    return true;
  }
  std::vector<const TensorShapeProto*> shape_attrs;
  if (!TryGetNodeAttr(n->attrs(), "_input_shapes", &shape_attrs)) {
    return true;
  }
  if ((n->type_string() == "MatMul" || n->type_string() == "_FusedMatMul")) {
    TensorShape lhs_shape, rhs_shape;
    if (TensorShape::BuildTensorShape(*shape_attrs[0], &lhs_shape) !=
        tsl::OkStatus()) {
      return true;
    }
    if (TensorShape::BuildTensorShape(*shape_attrs[1], &rhs_shape) !=
        tsl::OkStatus()) {
      return true;
    }
    auto M = lhs_shape.dim_size(0);
    auto K = lhs_shape.dim_size(1);
    auto N = rhs_shape.dim_size(1);
    auto ops = M * N * K;
    std::array<int, 3> n_threshold = {7560, 250, 1536};
    std::array<int, 2> m_threshold = {378, 80};
    std::array<int, 2> ops_threshold = {5242880, 1090519040};
    if (N <= n_threshold.at(0)) {
      if (ops <= ops_threshold.at(0)) {
        if (M <= m_threshold.at(0)) {
          return false;
        } else {
          if (N <= n_threshold.at(1)) {
            return false;
          } else {
            return true;
          }
        }
      } else {
        if (M <= m_threshold.at(1)) {
          if (N <= n_threshold.at(2)) {
            return true;
          } else {
            return false;
          }
        } else {
          if (ops <= ops_threshold.at(1)) {
            return true;
          } else {
            return false;
          }
        }
      }
    } else {
      return false;
    }
  }
  return true;
}
}  
#endif  
#endif  