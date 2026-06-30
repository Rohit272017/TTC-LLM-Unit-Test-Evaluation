#include "xla/tsl/lib/monitoring/sampler.h"
#include "absl/log/check.h"
#ifdef IS_MOBILE_PLATFORM
#else
namespace tsl {
namespace monitoring {
namespace {
class ExplicitBuckets : public Buckets {
 public:
  ~ExplicitBuckets() override = default;
  explicit ExplicitBuckets(std::vector<double> bucket_limits)
      : bucket_limits_(std::move(bucket_limits)) {
    CHECK_GT(bucket_limits_.size(), 0);
    for (size_t i = 1; i < bucket_limits_.size(); i++) {
      CHECK_GT(bucket_limits_[i], bucket_limits_[i - 1]);
    }
    if (bucket_limits_.back() != DBL_MAX) {
      bucket_limits_.push_back(DBL_MAX);
    }
  }
  const std::vector<double>& explicit_bounds() const override {
    return bucket_limits_;
  }
 private:
  std::vector<double> bucket_limits_;
  ExplicitBuckets(const ExplicitBuckets&) = delete;
  void operator=(const ExplicitBuckets&) = delete;
};
class ExponentialBuckets : public Buckets {
 public:
  ~ExponentialBuckets() override = default;
  ExponentialBuckets(double scale, double growth_factor, int bucket_count)
      : explicit_buckets_(
            ComputeBucketLimits(scale, growth_factor, bucket_count)) {}
  const std::vector<double>& explicit_bounds() const override {
    return explicit_buckets_.explicit_bounds();
  }
 private:
  static std::vector<double> ComputeBucketLimits(double scale,
                                                 double growth_factor,
                                                 int bucket_count) {
    CHECK_GT(bucket_count, 0);
    std::vector<double> bucket_limits;
    double bound = scale;
    for (int i = 0; i < bucket_count; i++) {
      bucket_limits.push_back(bound);
      bound *= growth_factor;
    }
    return bucket_limits;
  }
  ExplicitBuckets explicit_buckets_;
  ExponentialBuckets(const ExponentialBuckets&) = delete;
  void operator=(const ExponentialBuckets&) = delete;
};
}  
std::unique_ptr<Buckets> Buckets::Explicit(std::vector<double> bucket_limits) {
  return std::unique_ptr<Buckets>(
      new ExplicitBuckets(std::move(bucket_limits)));
}
std::unique_ptr<Buckets> Buckets::Explicit(
    std::initializer_list<double> bucket_limits) {
  return std::unique_ptr<Buckets>(new ExplicitBuckets(bucket_limits));
}
std::unique_ptr<Buckets> Buckets::Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count) {
  return std::unique_ptr<Buckets>(
      new ExponentialBuckets(scale, growth_factor, bucket_count));
}
}  
}  
#endif  