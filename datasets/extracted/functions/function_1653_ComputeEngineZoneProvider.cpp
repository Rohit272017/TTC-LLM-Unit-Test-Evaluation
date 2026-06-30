#include "tsl/platform/cloud/compute_engine_zone_provider.h"
#include <utility>
#include "tsl/platform/str_util.h"
namespace tsl {
namespace {
constexpr char kGceMetadataZonePath[] = "instance/zone";
}  
ComputeEngineZoneProvider::ComputeEngineZoneProvider(
    std::shared_ptr<ComputeEngineMetadataClient> google_metadata_client)
    : google_metadata_client_(std::move(google_metadata_client)) {}
absl::Status ComputeEngineZoneProvider::GetZone(string* zone) {
  if (!cached_zone.empty()) {
    *zone = cached_zone;
    return absl::OkStatus();
  }
  std::vector<char> response_buffer;
  TF_RETURN_IF_ERROR(google_metadata_client_->GetMetadata(kGceMetadataZonePath,
                                                          &response_buffer));
  absl::string_view location(&response_buffer[0], response_buffer.size());
  std::vector<string> elems = str_util::Split(location, "/");
  if (elems.size() == 4) {
    cached_zone = elems.back();
    *zone = cached_zone;
  } else {
    LOG(ERROR) << "Failed to parse the zone name from location: "
               << string(location);
  }
  return absl::OkStatus();
}
ComputeEngineZoneProvider::~ComputeEngineZoneProvider() {}
}  