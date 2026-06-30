#include <cmath>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/threadpool.h"
namespace {
using ::int64_t;
using tensorflow::int32;
constexpr int kCostMultiplier = 100;
class Segment {
 public:
  explicit Segment(int col_index)
      : col_start_(col_index), col_limit_(col_index + 1) {}
  int num_points() const { return col_limit_ - col_start_; }
  void merge_with(const Segment& other) {
    col_start_ = std::min(col_start_, other.col_start());
    col_limit_ = std::max(col_limit_, other.col_limit());
  }
  int col_start() const { return col_start_; }
  int col_limit() const { return col_limit_; }
 private:
  int col_start_;
  int col_limit_;
};
template <typename T>
class L2PavaSegment : public Segment {
 public:
  L2PavaSegment(T y, int col_index)
      : Segment(col_index), y_sum_(y), minimum_(y) {}
  void merge_with(const L2PavaSegment& other) {
    Segment::merge_with(other);
    y_sum_ += other.y_sum_;
    minimum_ = y_sum_ / static_cast<T>(num_points());
  }
  T minimum() const { return minimum_; }
 private:
  T y_sum_;    
  T minimum_;  
};
template <typename SegmentType, typename FloatTensor, typename IntTensor>
void solve_pava(const std::function<SegmentType(int, int)>& make_segment,
                FloatTensor* solution, IntTensor* segments, int row_index) {
  const size_t n = solution->dimensions()[1];
  std::vector<SegmentType> pools;
  pools.reserve(n);
  for (size_t col_index = 0; col_index < n; ++col_index) {
    pools.push_back(make_segment(row_index, col_index));
    while (pools.size() > 1 &&
           pools.rbegin()->minimum() > (pools.rbegin() + 1)->minimum()) {
      (pools.rbegin() + 1)->merge_with(*pools.rbegin());
      pools.pop_back();
    }
  }
  int segment_id = 0;
  for (const auto& pool : pools) {
    const auto pool_minimum = pool.minimum();
    auto* solution_ptr = &(*solution)(row_index, pool.col_start());
    auto* segments_ptr = &(*segments)(row_index, pool.col_start());
    for (int i = pool.col_start(); i < pool.col_limit(); ++i) {
      *solution_ptr++ = pool_minimum;
      *segments_ptr++ = segment_id;
    }
    ++segment_id;
  }
}
template <typename SegmentType, typename FloatTensor, typename IntTensor>
void solve_pava_batch(const std::function<SegmentType(int, int)>& make_segment,
                      FloatTensor* solution, IntTensor* segments,
                      tensorflow::OpKernelContext* context) {
  const int batch_size = solution->dimensions()[0];
  const int problem_size = solution->dimensions()[1];
  auto thread_pool =
      context->device()->tensorflow_cpu_worker_threads()->workers;
  thread_pool->ParallelFor(
      batch_size, kCostMultiplier * problem_size,
      [&make_segment, &solution, &segments](int64_t row_start,
                                            int64_t row_limit) {
        for (int row_index = static_cast<int>(row_start);
             row_index < static_cast<int>(row_limit); ++row_index) {
          solve_pava(make_segment, solution, segments, row_index);
        }
      });
}
}  
template <typename Tin, typename Tout>
class IsotonicRegressionOp : public tensorflow::OpKernel {
 public:
  explicit IsotonicRegressionOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}
  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input_tensor = context->input(0);
    const auto input = input_tensor.flat_inner_dims<Tin, 2>();
    int int_max = std::numeric_limits<int32_t>::max();
    OP_REQUIRES(context,
                tensorflow::FastBoundsCheck(input.dimensions()[0], int_max) &&
                    tensorflow::FastBoundsCheck(input.dimensions()[1], int_max),
                tensorflow::errors::InvalidArgument("Tensor too large"));
    const auto shape = input_tensor.shape();
    tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, shape, &output_tensor));
    auto output = output_tensor->flat_inner_dims<Tout, 2>();
    tensorflow::Tensor* segments_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape, &segments_tensor));
    auto segments = segments_tensor->flat_inner_dims<int>();
    auto make_l2_segment = [&input](int row_index, int col_index) {
      return L2PavaSegment<Tout>(input(row_index, col_index), col_index);
    };
    solve_pava_batch<L2PavaSegment<Tout>>(make_l2_segment, &output, &segments,
                                          context);
  }
};
#define REGISTER_CPU_KERNEL(Tin, Tout)                               \
  REGISTER_KERNEL_BUILDER(Name("IsotonicRegression")                 \
                              .Device(tensorflow::DEVICE_CPU)        \
                              .TypeConstraint<Tin>("T")              \
                              .TypeConstraint<Tout>("output_dtype"), \
                          IsotonicRegressionOp<Tin, Tout>);
#define REGISTER_CPU_SAME_KERNEL(T) REGISTER_CPU_KERNEL(T, T)
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SAME_KERNEL);
#define REGISTER_CPU_KERNEL_FLOAT(Tin) REGISTER_CPU_KERNEL(Tin, float)
TF_CALL_int16(REGISTER_CPU_KERNEL_FLOAT);
TF_CALL_int8(REGISTER_CPU_KERNEL_FLOAT);
#define REGISTER_CPU_KERNEL_DOUBLE(Tin) REGISTER_CPU_KERNEL(Tin, double)
TF_CALL_int64(REGISTER_CPU_KERNEL_DOUBLE);
TF_CALL_int32(REGISTER_CPU_KERNEL_DOUBLE);