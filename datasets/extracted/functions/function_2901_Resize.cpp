#include "tensorflow/lite/experimental/ml_adjacent/data/owning_vector_ref.h"
#include <cstddef>
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
namespace ml_adj {
namespace data {
void OwningVectorRef::Resize(dims_t&& dims) {
  dims_ = dims;
  num_elements_ = 0;
  for (dim_t d : dims_) {
    if (d <= 0) {
      break;
    }
    if (num_elements_ == 0) {
      num_elements_ = d;
    } else {
      num_elements_ *= d;
    }
  }
  raw_data_buffer_.resize(num_elements_ * TypeWidth(Type()));
}
const void* OwningVectorRef::Data() const { return raw_data_buffer_.data(); }
void* OwningVectorRef::Data() { return raw_data_buffer_.data(); }
ind_t OwningVectorRef::NumElements() const { return num_elements_; }
size_t OwningVectorRef::Bytes() const {
  return NumElements() * TypeWidth(Type());
}
}  
}  