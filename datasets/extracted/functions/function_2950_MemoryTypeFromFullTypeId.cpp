#include "tensorflow/core/common_runtime/arg_ret_placement.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow::full_type {
MemoryType MemoryTypeFromFullTypeId(FullTypeId id) {
  if (id == TFT_SHAPE_TENSOR) {
    return HOST_MEMORY;
  }
  return DEVICE_MEMORY;
}
bool LogMemoryTypeMismatch(bool use_host_memory, const FullTypeDef& ft) {
  FullTypeId id = ft.type_id();
  if (id == TFT_PRODUCT) {
    LOG(ERROR) << "Unexpected full type information for tensor, which should "
                  "not start with TFT_PRODUCT\n"
               << ft.DebugString();
    return false;
  }
  MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
  if (use_host_memory != (mt_from_ft == HOST_MEMORY)) {
    VLOG(1) << "use_host_memory=" << use_host_memory
            << "but full type information is\n"
            << ft.DebugString();
    return false;
  }
  return true;
}
Status CheckMemoryType(bool use_host_memory, const FullTypeDef& ft) {
  FullTypeId id = ft.type_id();
  MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
  if (id == TFT_PRODUCT) {
    return errors::Internal(
        "Unexpected full type information for tensor, which should not start "
        "with TFT_PRODUCT\n",
        ft.DebugString());
  }
  if (use_host_memory != (mt_from_ft == HOST_MEMORY)) {
    return errors::Internal("use_host_memory=", use_host_memory,
                            " but full type information is\n",
                            ft.DebugString());
  }
  return absl::OkStatus();
}
static Status SetMemoryTypeForNode(
    const Node* node, const DataType dtype, bool is_arg, bool weak_flag,
    bool ints_on_device, MemoryTypeVector* memory_types,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  const Node* n;
  int output_idx;
  if (is_arg) {
    DCHECK(node->op_def().name() == "_Arg" ||
           node->op_def().name() == "_DeviceArg");
    output_idx = 0;
    n = node;
  } else {
    DCHECK(node->op_def().name() == "_Retval" ||
           node->op_def().name() == "_DeviceRetval");
    const Edge* edge;
    TF_RETURN_IF_ERROR(node->input_edge(0, &edge));
    n = edge->src();
    output_idx = edge->src_output();
  }
  MemoryType mt_from_dtype = ints_on_device ? MTypeFromDTypeIntsOnDevice(dtype)
                                            : MTypeFromDType(dtype);
  if (dtype == DT_INT32) {
    if (n->def().has_experimental_type()) {
      bool valid_full_type_information = false;
      auto ft = n->def().experimental_type();
      if (ft.type_id() == TFT_PRODUCT) {
        FullTypeId id = GetArgDefaultUnset(ft, output_idx).type_id();
        MemoryType mt_from_ft = MemoryTypeFromFullTypeId(id);
        if ((id == TFT_TENSOR) || (id == TFT_SHAPE_TENSOR)) {
          valid_full_type_information = mt_from_dtype == mt_from_ft;
        } else if (id == TFT_UNSET) {
          valid_full_type_information = mt_from_dtype != HOST_MEMORY;
        }
      }
      if (!valid_full_type_information) {
        if (weak_flag) {
          VLOG(1) << "node=" << n->name() << " (op=" << n->def().op()
                  << ") has an int32 output with unexpected full type "
                  << "information with ints_on_device=" << ints_on_device
                  << "\n"
                  << n->def().DebugString();
        } else {
          return errors::Internal(
              "node=", n->name(), " (op=", n->def().op(),
              ") has an int32 output with unexpected full type information ",
              "with ints_on_device=", ints_on_device, "\n",
              n->def().DebugString());
        }
      }
    } else if (mt_from_dtype == HOST_MEMORY) {
      if (weak_flag) {
        VLOG(1) << "node=" << n->name() << " (op=" << n->def().op()
                << ") has a HOST_MEMORY int32 output but does not have "
                << "(TFT_SHAPE_TENSOR) full type information.";
      } else {
        return errors::Internal(
            "node=", n->name(), " (op=", n->def().op(),
            ")  has a HOST_MEMORY int32 output but does not have "
            "(TFT_SHAPE_TENSOR) full type information.");
      }
    }
  }
  if (memory_types != nullptr) {
    memory_types->push_back(mt_from_dtype);
  }
  if (alloc_attrs != nullptr) {
    AllocatorAttributes aa;
    aa.set_on_host(mt_from_dtype == HOST_MEMORY);
    alloc_attrs->push_back(aa);
  }
  return absl::OkStatus();
}
static Status SetMemoryTypeHelper(
    const absl::InlinedVector<Node*, 4UL>& nodes, const DataTypeVector& dtypes,
    bool is_arg, bool weak_flag, MemoryTypeVector* memory_types,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK_EQ(nodes.size(), dtypes.size());
  if (alloc_attrs != nullptr) {
    alloc_attrs->reserve(nodes.size());
  }
  for (int i = 0; i < nodes.size(); ++i) {
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(nodes[i], dtypes[i], is_arg,
                                            weak_flag, false,
                                            memory_types, alloc_attrs));
  }
  return absl::OkStatus();
}
static Status SetMemoryTypeHelper(
    const std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool weak_flag, bool ints_on_device,
    std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK(alloc_attrs != nullptr);
  alloc_attrs->reserve(arg_nodes.size());
  for (const auto& arg : arg_nodes) {
    const AttrValue* attr_value = arg.first->attrs().Find("T");
    if (attr_value == nullptr) {
      return errors::Internal("Arg node missing T attribute");
    }
    DataType dtype = attr_value->type();
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(
        arg.first, dtype, true, weak_flag, ints_on_device,
        nullptr, alloc_attrs));
  }
  return absl::OkStatus();
}
static Status SetMemoryTypeHelper(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool weak_flag,
    bool ints_on_device, std::vector<AllocatorAttributes>* alloc_attrs) {
  DCHECK(alloc_attrs != nullptr);
  alloc_attrs->reserve(ret_nodes.size());
  for (const auto& ret : ret_nodes) {
    const AttrValue* attr_value = ret.first->attrs().Find("T");
    if (attr_value == nullptr) {
      return errors::Internal("Ret node missing T attribute");
    }
    DataType dtype = attr_value->type();
    TF_RETURN_IF_ERROR(SetMemoryTypeForNode(
        ret.first, dtype, false, weak_flag, ints_on_device,
        nullptr, alloc_attrs));
  }
  return absl::OkStatus();
}
Status SetMemoryTypeForArgs(const absl::InlinedVector<Node*, 4UL>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, true,
                             false, &memory_types, nullptr);
}
Status WeakSetMemoryTypeForArgs(const absl::InlinedVector<Node*, 4UL>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, true,
                             true, &memory_types, nullptr);
}
Status SetMemoryTypeForRets(const absl::InlinedVector<Node*, 4UL>& nodes,
                            const DataTypeVector& dtypes,
                            MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, false,
                             false, &memory_types, nullptr);
}
Status WeakSetMemoryTypeForRets(const absl::InlinedVector<Node*, 4UL>& nodes,
                                const DataTypeVector& dtypes,
                                MemoryTypeVector& memory_types) {
  return SetMemoryTypeHelper(nodes, dtypes, false,
                             true, &memory_types, nullptr);
}
Status SetAllocAttrsForArgs(const absl::InlinedVector<Node*, 4UL>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, true,
                             false, nullptr, &alloc_attrs);
}
Status WeakSetAllocAttrsForArgs(const absl::InlinedVector<Node*, 4UL>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, true,
                             true, nullptr, &alloc_attrs);
}
Status SetAllocAttrsForRets(const absl::InlinedVector<Node*, 4UL>& nodes,
                            const DataTypeVector& dtypes,
                            std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, false,
                             false, nullptr, &alloc_attrs);
}
Status WeakSetAllocAttrsForRets(const absl::InlinedVector<Node*, 4UL>& nodes,
                                const DataTypeVector& dtypes,
                                std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(nodes, dtypes, false,
                             true, nullptr, &alloc_attrs);
}
Status SingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(arg_nodes, false, ints_on_device,
                             &alloc_attrs);
}
Status WeakSingleDeviceSetAllocAttrsForArgs(
    std::vector<std::pair<Node*, FunctionArgIndex>> arg_nodes,
    bool ints_on_device, std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(arg_nodes, true, ints_on_device,
                             &alloc_attrs);
}
Status SingleDeviceSetAllocAttrsForRets(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(ret_nodes, false, ints_on_device,
                             &alloc_attrs);
}
Status WeakSingleDeviceSetAllocAttrsForRets(
    const std::vector<std::pair<Node*, int>> ret_nodes, bool ints_on_device,
    std::vector<AllocatorAttributes>& alloc_attrs) {
  return SetMemoryTypeHelper(ret_nodes, true, ints_on_device,
                             &alloc_attrs);
}
}  