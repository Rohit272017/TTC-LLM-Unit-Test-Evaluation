#ifndef TENSORSTORE_UTIL_SPAN_JSON_H_
#define TENSORSTORE_UTIL_SPAN_JSON_H_
#include <cstddef>
#include <nlohmann/json.hpp>
#include "tensorstore/util/span.h"
namespace tensorstore {
template <typename T, ptrdiff_t Extent>
void to_json(::nlohmann::json& out,  
             tensorstore::span<T, Extent> s) {
  out = ::nlohmann::json::array_t(s.begin(), s.end());
}
}  
#endif  