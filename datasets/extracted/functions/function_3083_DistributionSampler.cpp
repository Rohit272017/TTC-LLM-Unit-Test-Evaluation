#include "xla/tsl/lib/random/distribution_sampler.h"
#include <memory>
#include <vector>
#include "absl/types/span.h"
namespace tsl {
namespace random {
DistributionSampler::DistributionSampler(
    const absl::Span<const float> weights) {
  DCHECK(!weights.empty());
  int n = weights.size();
  num_ = n;
  data_.reset(new std::pair<float, int>[n]);
  std::unique_ptr<double[]> pr(new double[n]);
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += weights[i];
    set_alt(i, -1);
  }
  std::vector<int> high;
  high.reserve(n);
  std::vector<int> low;
  low.reserve(n);
  for (int i = 0; i < n; i++) {
    double p = (weights[i] * n) / sum;
    pr[i] = p;
    if (p < 1.0) {
      low.push_back(i);
    } else {
      high.push_back(i);
    }
  }
  while (!high.empty() && !low.empty()) {
    int l = low.back();
    low.pop_back();
    int h = high.back();
    high.pop_back();
    set_alt(l, h);
    DCHECK_GE(pr[h], 1.0);
    double remaining = pr[h] - (1.0 - pr[l]);
    pr[h] = remaining;
    if (remaining < 1.0) {
      low.push_back(h);
    } else {
      high.push_back(h);
    }
  }
  for (int i = 0; i < n; i++) {
    set_prob(i, pr[i]);
  }
  for (size_t i = 0; i < high.size(); i++) {
    int idx = high[i];
    set_prob(idx, 1.0);
    set_alt(idx, idx);
  }
  for (size_t i = 0; i < low.size(); i++) {
    int idx = low[i];
    set_prob(idx, 1.0);
    set_alt(idx, idx);
  }
}
}  
}  