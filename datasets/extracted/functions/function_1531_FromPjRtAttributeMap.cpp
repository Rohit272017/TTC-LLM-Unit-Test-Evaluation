#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/attribute_map.h"
namespace xla {
namespace ifrt {
AttributeMap FromPjRtAttributeMap(
    absl::flat_hash_map<std::string, xla::PjRtValueType> attributes) {
  AttributeMap::Map result;
  result.reserve(attributes.size());
  for (auto& item : attributes) {
    std::visit(
        [&](auto& value) {
          using T = std::decay_t<decltype(value)>;
          const auto& key = item.first;
          if constexpr (std::is_same_v<T, std::string>) {
            result.insert({key, AttributeMap::StringValue(std::move(value))});
          } else if constexpr (std::is_same_v<T, bool>) {
            result.insert({key, AttributeMap::BoolValue(value)});
          } else if constexpr (std::is_same_v<T, int64_t>) {
            result.insert({key, AttributeMap::Int64Value(value)});
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            result.insert(
                {key, AttributeMap::Int64ListValue(std::move(value))});
          } else if constexpr (std::is_same_v<T, float>) {
            result.insert({key, AttributeMap::FloatValue(value)});
          }
        },
        item.second);
  }
  return AttributeMap(std::move(result));
}
absl::flat_hash_map<std::string, xla::PjRtValueType> ToPjRtAttributeMap(
    AttributeMap attributes) {
  absl::flat_hash_map<std::string, xla::PjRtValueType> result;
  result.reserve(attributes.map().size());
  for (auto& item : attributes.map()) {
    std::visit(
        [&](auto& value) {
          using T = std::decay_t<decltype(value)>;
          const auto& key = item.first;
          if constexpr (std::is_same_v<T, AttributeMap::StringValue>) {
            result.insert({key, std::move(value.value)});
          } else if constexpr (std::is_same_v<T, AttributeMap::BoolValue>) {
            result.insert({key, value.value});
          } else if constexpr (std::is_same_v<T, AttributeMap::Int64Value>) {
            result.insert({key, value.value});
          } else if constexpr (std::is_same_v<T,
                                              AttributeMap::Int64ListValue>) {
            result.insert({key, std::move(value.value)});
          } else if constexpr (std::is_same_v<T, AttributeMap::FloatValue>) {
            result.insert({key, value.value});
          }
        },
        item.second);
  }
  return result;
}
}  
}  