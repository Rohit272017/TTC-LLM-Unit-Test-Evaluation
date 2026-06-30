#include "xla/python/ifrt/support/sharding_conversions.h"
#include <cstdint>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace ifrt {
namespace support {
absl::StatusOr<OpSharding> ToOpSharding(const Sharding& sharding) {
  if (auto* sharding_param_sharding =
          llvm::dyn_cast<xla::ifrt::ShardingParamSharding>(&sharding)) {
    return ToOpSharding(sharding_param_sharding->sharding_param(),
                        sharding_param_sharding->devices());
  } else {
    return absl::InvalidArgumentError(
        "Only conversion from `ShardingParamSharding` to `OpSharding` is "
        "supported.");
  }
}
absl::StatusOr<OpSharding> ToOpSharding(
    const ShardingParam& sharding_param,
    const tsl::RCReference<xla::ifrt::DeviceList>& device_mapping) {
  OpSharding op_sharding;
  {
    bool all_dim_replicated = true;
    for (const int64_t dim_shard : sharding_param.dim_shards()) {
      if (dim_shard != 1) {
        all_dim_replicated = false;
        break;
      }
    }
    if (all_dim_replicated) {
      op_sharding.set_type(OpSharding::REPLICATED);
      return op_sharding;
    }
  }
  op_sharding.set_type(OpSharding::OTHER);
  auto* tile_assignment_dims = op_sharding.mutable_tile_assignment_dimensions();
  int64_t cum_size = 1;
  tile_assignment_dims->Reserve(sharding_param.dim_shards().size() + 1);
  for (const int64_t dim_shard : sharding_param.dim_shards()) {
    cum_size *= dim_shard;
    tile_assignment_dims->Add(dim_shard);
  }
  int device_count = 1;
  for (const int axis_size : sharding_param.minor_to_major().axis_sizes) {
    device_count *= axis_size;
  }
  if (device_count != cum_size) {
    op_sharding.set_replicate_on_last_tile_dim(true);
    tile_assignment_dims->Add(device_count / cum_size);
  }
  llvm::SmallVector<int> logical_device_ids;
  sharding_param.minor_to_major().ToDeviceList(logical_device_ids);
  auto* tile_assignment_devices = op_sharding.mutable_tile_assignment_devices();
  tile_assignment_devices->Reserve(logical_device_ids.size());
  const absl::Span<Device* const> device_mapping_devices =
      device_mapping->devices();
  for (const int logical_device_id : logical_device_ids) {
    if (logical_device_id < 0 ||
        logical_device_id >= device_mapping_devices.size()) {
      return absl::OutOfRangeError(
          absl::StrCat("Can't map device with logical id ", logical_device_id,
                       ". The logical device id should be within [0, ",
                       device_mapping_devices.size(), ")."));
    }
    tile_assignment_devices->Add(
        device_mapping_devices[logical_device_id]->Id().value());
  }
  return op_sharding;
}
absl::StatusOr<HloSharding> ToHloSharding(const ShardingParam& sharding_param) {
  auto axis_sizes = sharding_param.minor_to_major().axis_sizes;
  llvm::SmallVector<int64_t> reshape_dims;
  reshape_dims.reserve(axis_sizes.size());
  int device_count = 1;
  for (auto axis_size : llvm::reverse(axis_sizes)) {
    reshape_dims.push_back(axis_size);
    device_count *= axis_size;
  }
  if (device_count == 1) {
    return HloSharding::Replicate();
  }
  int64_t cum_size = 1;
  llvm::SmallVector<int64_t> dims;
  dims.reserve(sharding_param.dim_shards().size());
  for (const int64_t dim_shard : sharding_param.dim_shards()) {
    cum_size *= dim_shard;
    dims.push_back(dim_shard);
  }
  llvm::SmallVector<int, 4> permutation;
  int num_axis = sharding_param.minor_to_major().permutation.size();
  permutation.reserve(num_axis);
  for (const int axis_id :
       llvm::reverse(sharding_param.minor_to_major().permutation)) {
    permutation.push_back(num_axis - axis_id - 1);
  }
  if (device_count != cum_size) {
    dims.push_back(device_count / cum_size);
    return HloSharding::PartialTile(
        TileAssignment(dims, reshape_dims, permutation));
  } else {
    return HloSharding::IotaTile(dims, reshape_dims, permutation);
  }
}
absl::StatusOr<ShardingParam> ToShardingParam(const HloSharding& hlo_sharding,
                                              int rank, int num_devices) {
  ShardingParam::MinorToMajor minor_to_major;
  if (hlo_sharding.IsReplicated() ||
      (hlo_sharding.IsTileMaximal() && hlo_sharding.HasUniqueDevice() &&
       num_devices == 1)) {
    llvm::SmallVector<int64_t> dim_shards(rank, 1);
    minor_to_major.permutation.push_back(0);
    minor_to_major.axis_sizes.push_back(num_devices);
    return ShardingParam(dim_shards, std::move(minor_to_major));
  } else if (hlo_sharding.IsTiled()) {
    const xla::TileAssignment& tile_assignment = hlo_sharding.tile_assignment();
    if (!tile_assignment.iota()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Conversion from `HloSharding` without `IotaTileAssignment` is not "
          "supported; sharding=",
          hlo_sharding.ToString()));
    }
    if (rank != hlo_sharding.TiledDataRank()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "`TiledData` expected to have have %d dimensions, but has %d "
          "dimensions; sharding=%s",
          rank, hlo_sharding.TiledDataRank(), hlo_sharding.ToString()));
    }
    if (hlo_sharding.subgroup_types().size() > 1 ||
        (hlo_sharding.subgroup_types().size() == 1 &&
         hlo_sharding.subgroup_types()[0] != xla::OpSharding::REPLICATED)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported conversion to `ShardingParam` from `HloSharding` that "
          "has more than a subgroup or a subgroup that is not REPLICATED; "
          "sharding=",
          hlo_sharding.ToString()));
    }
    llvm::SmallVector<int64_t> dim_shards(tile_assignment.dimensions().begin(),
                                          tile_assignment.dimensions().end());
    if (hlo_sharding.ReplicateOnLastTileDim() ||
        (hlo_sharding.subgroup_types().size() == 1 &&
         hlo_sharding.subgroup_types()[0] == xla::OpSharding::REPLICATED)) {
      dim_shards.pop_back();
    }
    if (tile_assignment.iota()->reshape_dims().empty()) {
      minor_to_major.permutation.push_back(0);
      minor_to_major.axis_sizes.push_back(num_devices);
    } else {
      for (auto reshape_dim :
           llvm::reverse(tile_assignment.iota()->reshape_dims())) {
        minor_to_major.axis_sizes.push_back(reshape_dim);
      }
      int num_axis = tile_assignment.iota()->transpose_perm().size();
      for (int axis_id :
           llvm::reverse(tile_assignment.iota()->transpose_perm())) {
        minor_to_major.permutation.push_back(num_axis - axis_id - 1);
      }
    }
    return ShardingParam(dim_shards, std::move(minor_to_major));
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported conversion to `ShardingParam` from "
                   "`HloSharding`; sharding=",
                   hlo_sharding.ToString()));
}
}  
}  
}  