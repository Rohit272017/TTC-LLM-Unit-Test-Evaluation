#include "xla/service/gpu/runtime/outfeed_thunk.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_transfer_manager.h"
#include "xla/service/gpu/outfeed_manager.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace gpu {
OutfeedThunk::OutfeedThunk(ThunkInfo thunk_info,
                           std::vector<ShapedSlice> source_slices)
    : Thunk(Kind::kOutfeed, thunk_info),
      source_slices_(std::move(source_slices)) {}
absl::Status OutfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& buffer_allocations = *params.buffer_allocations;
  VLOG(2) << "Outfeeding from GPU";
  OutfeedManager* outfeed_manager =
      GpuTransferManager::GetOrCreateOutfeedManager(stream.parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* output_buffers =
      outfeed_manager->BlockingGetNextDestination();
  if (source_slices_.empty()) {
    return absl::OkStatus();
  }
  const int64_t leaf_count = output_buffers->leaf_count();
  TF_RET_CHECK(source_slices_.size() == leaf_count)
      << "Mismatch between number of outfeed inputs (" << source_slices_.size()
      << ") and outputs (" << leaf_count << ")";
  auto output_leaf_it = output_buffers->leaf_begin();
  for (int64_t index = 0; index < leaf_count; ++index) {
    const ShapeIndex& shape_index = output_leaf_it->first;
    std::unique_ptr<OutfeedBuffer>& buffer = output_leaf_it->second;
    ++output_leaf_it;
    const Shape& output_shape =
        ShapeUtil::GetSubshape(output_buffers->shape(), shape_index);
    TF_RET_CHECK(
        ShapeUtil::ReshapeIsBitcast(source_slices_[index].shape, output_shape))
        << "Mismatch between outfeed output buffer shape "
        << ShapeUtil::HumanStringWithLayout(output_shape)
        << " and outfeed source buffer shape "
        << ShapeUtil::HumanStringWithLayout(source_slices_[index].shape);
    BufferAllocation::Slice source_slice = source_slices_[index].slice;
    if (!source_slice.allocation())
      return Internal("outfeed source missing buffer allocation");
    se::DeviceMemoryBase data_address =
        buffer_allocations.GetDeviceAddress(source_slice);
    TF_RETURN_IF_ERROR(stream.Memcpy(buffer->destination()->untyped_data(),
                                     data_address, buffer->length()));
    TF_RETURN_IF_ERROR(stream.DoHostCallback([&buffer]() { buffer->Done(); }));
  }
  absl::Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return Internal("Failed to complete data transfer on stream %p: %s",
                    &stream, block_status.message());
  }
  VLOG(2) << "Outfeeding from GPU complete";
  return absl::OkStatus();
}
}  
}  