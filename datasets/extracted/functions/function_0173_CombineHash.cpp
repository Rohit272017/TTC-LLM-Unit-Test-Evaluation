#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_HASH_TOOLS_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_HASH_TOOLS_H_
#include <cstddef>
namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
inline void CombineHash(size_t from, size_t* to) {
  *to ^= from + 0x9e3779b9 + (*to << 6) + (*to >> 2);
}
inline void CombineHashCommutative(size_t from, size_t* to) {
  *to = *to + from + 0x9e3779b9;
}
}  
}  
}  
#endif  