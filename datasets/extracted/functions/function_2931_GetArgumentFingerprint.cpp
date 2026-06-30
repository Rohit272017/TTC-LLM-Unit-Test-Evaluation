#include "xla/service/gpu/kernel_reuse_cache.h"
#include <functional>
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
namespace xla {
namespace gpu {
namespace {
std::string GetArgumentFingerprint(
    absl::Span<const KernelArgument> kernel_arguments) {
  return absl::StrJoin(
      kernel_arguments, ",", [](std::string* s, const KernelArgument& arg) {
        if (arg.first_with_same_slice().has_value()) {
          absl::StrAppend(s, "=", arg.first_with_same_slice().value());
          return;
        }
        absl::StrAppend(s, arg.alignment());
        if (arg.aliased()) {
          absl::StrAppend(s, "a");
        }
        if (arg.written()) {
          absl::StrAppend(s, "w");
        }
      });
}
}  
std::string GetComputationFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const KernelArgument> kernel_arguments,
    absl::string_view discriminator) {
  auto print_options = HloPrintOptions::Fingerprint()
                           .set_print_only_essential_constants(false)
                           .set_print_operand_shape(false);
  return absl::StrCat(discriminator, "(",
                      GetArgumentFingerprint(kernel_arguments), ")",
                      fused_computation->ToString(print_options));
}
absl::Status KernelReuseCache::Load(const CompilationCacheProto& proto) {
  for (const auto& [name, entry] : proto.entries()) {
    std::optional<se::ClusterDim> cluster_dim;
    if (entry.has_cluster_dim()) {
      cluster_dim =
          se::ClusterDim{entry.cluster_dim().x(), entry.cluster_dim().y(),
                         entry.cluster_dim().z()};
    }
    TF_RET_CHECK(
        cache_
            .insert(
                {entry.fingerprint(),
                 Entry{name,
                       LaunchDimensions{
                           entry.launch_dimensions().num_blocks(),
                           entry.launch_dimensions().num_threads_per_block()},
                       cluster_dim, entry.shmem_bytes(), entry.binary()}})
            .second);
  }
  return absl::OkStatus();
}
CompilationCacheProto KernelReuseCache::Export() const {
  CompilationCacheProto proto;
  for (const auto& [fingerprint, cache_entry] : cache_) {
    if (!hits_.contains(fingerprint)) {
      VLOG(5) << "Not exporting unused " << cache_entry.kernel_name;
      continue;
    }
    auto [it, inserted] = proto.mutable_entries()->emplace(
        cache_entry.kernel_name, CompilationCacheEntryProto{});
    CHECK(inserted) << cache_entry.kernel_name;
    CompilationCacheEntryProto& proto_entry = it->second;
    proto_entry.set_fingerprint(fingerprint);
    LaunchDimensionsProto launch_dimensions_proto;
    launch_dimensions_proto.set_num_blocks(
        cache_entry.launch_dimensions.num_blocks());
    launch_dimensions_proto.set_num_threads_per_block(
        cache_entry.launch_dimensions.num_threads_per_block());
    *proto_entry.mutable_launch_dimensions() = launch_dimensions_proto;
    if (cache_entry.cluster_dim.has_value()) {
      ClusterDimProto cluster_dim_proto;
      cluster_dim_proto.set_x(cache_entry.cluster_dim->x);
      cluster_dim_proto.set_y(cache_entry.cluster_dim->y);
      cluster_dim_proto.set_z(cache_entry.cluster_dim->z);
      *proto_entry.mutable_cluster_dim() = cluster_dim_proto;
    }
    proto_entry.set_shmem_bytes(cache_entry.shmem_bytes);
    proto_entry.set_binary(cache_entry.binary);
  }
  return proto;
}
absl::Status UpdateDiskKernelCache(
    absl::string_view path, const bool do_append,
    const CompilationCacheProto& current_cache,
    absl::Span<const KernelReuseCache::NamedBinary> binaries_to_cache) {
  CompilationCacheProto disk_cache;
  if (do_append) {
    std::string serialized;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                             std::string(path), &serialized));
    if (!disk_cache.ParseFromString(std::string(serialized))) {
      return Internal("Failed to parse serialized CompilationCacheProto.");
    }
  }
  auto entries = disk_cache.mutable_entries();
  int stored_kernel_count = 0;
  for (const auto& [name, binary] : binaries_to_cache) {
    auto it_current = current_cache.entries().find(name);
    TF_RET_CHECK(it_current != current_cache.entries().end());
    auto [it_disk, inserted] = entries->insert({name, it_current->second});
    TF_RET_CHECK(inserted);
    TF_RET_CHECK(!binary.empty());
    it_disk->second.set_binary(reinterpret_cast<const char*>(binary.data()),
                               binary.size());
    VLOG(5) << "Cached kernel: " << name << ": " << binary.size();
    ++stored_kernel_count;
  }
  if (stored_kernel_count > 0) {
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(),
                                              std::string(path),
                                              disk_cache.SerializeAsString()));
    VLOG(2) << "Stored " << stored_kernel_count << " / "
            << binaries_to_cache.size() << " kernels in the cache file.";
  }
  return absl::OkStatus();
}
std::pair<absl::StatusOr<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    const HloComputation* fused_computation,
    absl::Span<const KernelArgument> kernel_arguments,
    absl::string_view discriminator,
    const std::function<absl::StatusOr<KernelReuseCache::Entry>()>& generator) {
  std::string fingerprint = GetComputationFingerprint(
      fused_computation, kernel_arguments, discriminator);
  VLOG(4) << "Fingerprint: ";
  XLA_VLOG_LINES(4, fingerprint);
  return GetWithStatus(std::move(fingerprint), generator);
}
std::pair<absl::StatusOr<const KernelReuseCache::Entry*>, bool>
KernelReuseCache::GetWithStatus(
    std::string fingerprint,
    const std::function<absl::StatusOr<KernelReuseCache::Entry>()>& generator) {
  hits_.insert(fingerprint);
  auto it = cache_.find(fingerprint);
  if (it != cache_.end()) {
    return {&it->second, true};
  }
  absl::StatusOr<Entry> entry = generator();
  if (entry.ok()) {
    it =
        cache_.insert({std::move(fingerprint), std::move(entry.value())}).first;
    return {&it->second, false};
  }
  return {entry.status(), false};
}
}  
}  