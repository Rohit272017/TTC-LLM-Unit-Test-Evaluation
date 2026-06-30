#ifndef AROLLA_UTIL_MAP_H_
#define AROLLA_UTIL_MAP_H_
#include <algorithm>
#include <vector>
namespace arolla {
template <class Map>
std::vector<typename Map::key_type> SortedMapKeys(const Map& map) {
  std::vector<typename Map::key_type> result;
  result.reserve(map.size());
  for (const auto& item : map) {
    result.push_back(item.first);
  }
  std::sort(result.begin(), result.end());
  return result;
}
}  
#endif  