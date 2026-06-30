#ifndef TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
#define TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/bcast.h"
namespace tensorflow {
class MatMulBCast {
 public:
  using Vec = BCast::Vec;
  MatMulBCast(const Vec& x, const Vec& y) {
    if (std::max(x.size(), y.size()) == 2) return;
    const Vec x_resized(x.begin(), x.end() - 2);
    const Vec y_resized(y.begin(), y.end() - 2);
    batch_bcast_ =
        std::make_unique<BCast>(std::move(x_resized), std::move(y_resized));
    if (!batch_bcast_->IsValid()) {
      broadcasting_required_ = true;
      return;
    }
    x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
    y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
    output_batch_shape_ = TensorShape(batch_bcast_->output_shape());
    output_batch_size_ = output_batch_shape_.num_elements();
    broadcasting_required_ =
        std::min(x_batch_size_, y_batch_size_) != output_batch_size_;
    if (broadcasting_required_) {
      ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                          batch_bcast_->x_bcast(), &x_batch_indices_);
      ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                          batch_bcast_->y_bcast(), &y_batch_indices_);
    }
  }
  bool IsValid() const {
    return !broadcasting_required_ || (batch_bcast_ && batch_bcast_->IsValid());
  }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }
  int64_t output_batch_size() const { return output_batch_size_; }
  int64_t x_batch_size() const { return x_batch_size_; }
  int64_t y_batch_size() const { return y_batch_size_; }
  const TensorShape& output_batch_shape() const { return output_batch_shape_; }
  const std::vector<int64_t>& x_batch_indices() const {
    return x_batch_indices_;
  }
  const std::vector<int64_t>& y_batch_indices() const {
    return y_batch_indices_;
  }
 private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64_t x_batch_size_ = 1;
  int64_t y_batch_size_ = 1;
  TensorShape output_batch_shape_;
  int64_t output_batch_size_ = 1;
  std::vector<int64_t> x_batch_indices_;
  std::vector<int64_t> y_batch_indices_;
};
}  
#endif  