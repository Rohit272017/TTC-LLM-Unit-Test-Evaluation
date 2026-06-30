#include "tensorflow/compiler/aot/benchmark.h"
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace tfcompile {
namespace benchmark {
static uint64 NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
void DumpStatsToStdout(const Stats& stats) {
  std::vector<int64_t> sorted_us(stats.per_iter_us);
  std::sort(sorted_us.begin(), sorted_us.end());
  const size_t count_us = sorted_us.size();
  double sum_us = 0;
  size_t count_us_trimmed = 0;
  double sum_us_trimmed = 0;
  size_t count_us_best = 0;
  double sum_us_best = 0;
  static constexpr float trim_ratio = 0.25;
  static constexpr float best_ratio = 0.1;
  const size_t count_trimmed = count_us * trim_ratio;
  const size_t count_best = count_us * best_ratio;
  for (size_t i = 0; i < sorted_us.size(); ++i) {
    const int64_t us = sorted_us[i];
    sum_us += us;
    if (i >= count_trimmed && i < count_us - count_trimmed) {
      sum_us_trimmed += us;
      ++count_us_trimmed;
    }
    if (i < count_best) {
      sum_us_best += us;
      ++count_us_best;
    }
  }
  const int kBufSize = 1000;
  char buf[kBufSize];
  snprintf(buf, kBufSize, "Mean with %2.0f%% trimmed:", trim_ratio * 100);
  std::string label_trimmed(buf);
  snprintf(buf, kBufSize, "Mean of %2.0f%% best:", best_ratio * 100);
  std::string label_best(buf);
  std::vector<std::pair<std::string, double>> groups = {
      {"Best:", sorted_us.front()},
      {"Worst:", sorted_us.back()},
      {"Median:", sorted_us[count_us / 2]},
      {"Mean:", sum_us / count_us},
      {std::move(label_trimmed), sum_us_trimmed / count_us_trimmed},
      {std::move(label_best), sum_us_best / count_us_best},
  };
  int max_label_size = 0;
  double max_us = 0;
  for (const auto& g : groups) {
    if (g.first.size() > max_label_size) {
      max_label_size = g.first.size();
    }
    if (g.second > max_us) {
      max_us = g.second;
    }
  }
  int max_digits = 1;
  while (max_us >= 10.0) {
    max_us /= 10.0;
    ++max_digits;
  }
  printf("Benchmark ran %zu iterations over %lld us\n", count_us,
         static_cast<long long>(stats.total_us));  
  for (const auto& g : groups) {
    printf("  %-*s %*.3f us\n", max_label_size, g.first.c_str(), max_digits + 4,
           g.second);
  }
}
void Benchmark(const Options& options, const BenchmarkFn& fn, Stats* stats) {
  const int64_t max_us = (options.max_micros <= 0 && options.max_iters <= 0)
                             ? Options::kDefaultMicros
                             : options.max_micros;
  printf("Running benchmark for %lld us\n", static_cast<long long>(max_us));
  const int64_t start_us = NowMicros();
  int64_t iters = 0;
  while (true) {
    const int64_t iter_start_us = NowMicros();
    fn();
    const int64_t end_us = NowMicros();
    stats->per_iter_us.push_back(end_us - iter_start_us);
    const int64_t total_us = end_us - start_us;
    ++iters;
    if ((max_us > 0 && total_us >= max_us) ||
        (options.max_iters > 0 && iters >= options.max_iters)) {
      stats->total_us = total_us;
      break;
    }
  }
}
}  
}  
}  