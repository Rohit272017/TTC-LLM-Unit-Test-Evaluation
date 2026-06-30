#include <libaddressinput/region_data.h>
#include <cstddef>
#include <string>
#include <vector>
namespace i18n {
namespace addressinput {
RegionData::RegionData(const std::string& region_code)
    : key_(region_code),
      name_(region_code),
      parent_(nullptr),
      sub_regions_() {}
RegionData::~RegionData() {
  for (auto ptr : sub_regions_) {
    delete ptr;
  }
}
RegionData* RegionData::AddSubRegion(const std::string& key,
                                     const std::string& name) {
  auto* sub_region = new RegionData(key, name, this);
  sub_regions_.push_back(sub_region);
  return sub_region;
}
RegionData::RegionData(const std::string& key,
                       const std::string& name,
                       RegionData* parent)
    : key_(key), name_(name), parent_(parent), sub_regions_() {}
}  
}  