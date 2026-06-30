#include "tensorstore/util/element_pointer.h"
#include <string>
#include "tensorstore/data_type.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_element_pointer {
std::string DescribeForCast(DataType dtype) {
  return tensorstore::StrCat("pointer with ",
                             StaticCastTraits<DataType>::Describe(dtype));
}
}  
}  