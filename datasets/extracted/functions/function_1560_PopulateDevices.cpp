#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
namespace tensorflow {
namespace dtensor {
namespace {
void PopulateDevices(absl::Span<const int64_t> permutation,
                     absl::Span<const int64_t> sizes,
                     absl::Span<const int64_t> cum_sizes,
                     std::vector<int64_t>* out_devices, int64_t base = 0) {
  int expanding_dim = permutation[0];
  int expanding_dim_size = sizes[expanding_dim];
  int expanding_cum_dim_size = cum_sizes[expanding_dim];
  for (int i = 0; i < expanding_dim_size; ++i) {
    if (permutation.size() == 1) {
      out_devices->push_back(base + i * expanding_cum_dim_size);
    } else {
      PopulateDevices(permutation.subspan(1), sizes, cum_sizes, out_devices,
                      base + i * expanding_cum_dim_size);
    }
  }
}
}  
std::vector<int64_t> MeshMajorToMinor::ToDeviceList() {
  std::vector<int64_t> cum_sizes(sizes.size());
  int64_t cum_size = 1;
  for (int i = sizes.size() - 1; i >= 0; --i) {
    cum_sizes[i] = cum_size;
    cum_size *= sizes[i];
  }
  std::vector<int64_t> devices;
  devices.reserve(cum_size * sizes[0]);
  PopulateDevices(permutation, sizes, cum_sizes, &devices);
  return devices;
}
StatusOr<MeshMajorToMinor> ConvertMeshMajorToMinor(const Layout& layout,
                                                   const Mesh& mesh) {
  MeshMajorToMinor major_to_minor;
  major_to_minor.permutation.reserve(mesh.dims().size());
  major_to_minor.sizes.reserve(mesh.dims().size());
  absl::flat_hash_map<std::string, int64_t> dim_name_to_index_map;
  for (const auto& [index, mesh_dim] : llvm::enumerate(mesh.dims())) {
    major_to_minor.sizes.push_back(mesh_dim.size);
    dim_name_to_index_map[mesh_dim.name] = index;
  }
  for (const auto& spec : layout.sharding_spec_strs()) {
    if (mesh.IsMeshDim(spec)) {
      const auto it = dim_name_to_index_map.find(spec);
      TF_RET_CHECK(it != dim_name_to_index_map.end());
      const auto& dimension_index = it->second;
      major_to_minor.permutation.push_back(dimension_index);
      dim_name_to_index_map.erase(it);
    }
  }
  for (const auto& [name, unused_size] : mesh.dims()) {
    if (const auto it = dim_name_to_index_map.find(name);
        it != dim_name_to_index_map.end()) {
      const auto& dimension_index = it->second;
      major_to_minor.permutation.push_back(dimension_index);
    }
  }
  TF_RET_CHECK(major_to_minor.permutation.size() ==
               major_to_minor.sizes.size());
  return major_to_minor;
}
StatusOr<::xla::OpSharding> ConvertLayoutToXlaOpSharding(const Layout& layout) {
  ::xla::OpSharding xla_sharding;
  if (layout.IsSingleDevice()) {
    xla_sharding.set_type(::xla::OpSharding::MAXIMAL);
    return xla_sharding;
  } else if (layout.IsFullyReplicated()) {
    xla_sharding.set_type(::xla::OpSharding::REPLICATED);
    return xla_sharding;
  }
  xla_sharding.set_type(::xla::OpSharding::OTHER);
  const Mesh& mesh = layout.mesh();
  {
    int32 product_of_sharded_dimensions = 1;
    for (int32 dim_size : layout.num_shards()) {
      product_of_sharded_dimensions *= dim_size;
      xla_sharding.add_tile_assignment_dimensions(dim_size);
    }
    if (product_of_sharded_dimensions != mesh.num_devices()) {
      xla_sharding.add_tile_assignment_dimensions(
          mesh.num_devices() / product_of_sharded_dimensions);
      xla_sharding.set_replicate_on_last_tile_dim(true);
    }
  }
  TF_ASSIGN_OR_RETURN(auto major_to_minor,
                      ConvertMeshMajorToMinor(layout, mesh));
  std::vector<int64_t> tile_assignment_devices = major_to_minor.ToDeviceList();
  *(xla_sharding.mutable_tile_assignment_devices()) = {
      tile_assignment_devices.begin(), tile_assignment_devices.end()};
  return xla_sharding;
}
}  
}  