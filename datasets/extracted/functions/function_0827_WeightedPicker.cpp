#include "xla/tsl/lib/random/weighted_picker.h"
#include <string.h>
#include <algorithm>
#include "xla/tsl/lib/random/simple_philox.h"
namespace tsl {
namespace random {
WeightedPicker::WeightedPicker(int N) {
  CHECK_GE(N, 0);
  N_ = N;
  num_levels_ = 1;
  while (LevelSize(num_levels_ - 1) < N) {
    num_levels_++;
  }
  level_ = new int32*[num_levels_];
  for (int l = 0; l < num_levels_; l++) {
    level_[l] = new int32[LevelSize(l)];
  }
  SetAllWeights(1);
}
WeightedPicker::~WeightedPicker() {
  for (int l = 0; l < num_levels_; l++) {
    delete[] level_[l];
  }
  delete[] level_;
}
static int32 UnbiasedUniform(SimplePhilox* r, int32_t n) {
  CHECK_LE(0, n);
  const uint32 range = ~static_cast<uint32>(0);
  if (n == 0) {
    return r->Rand32() * n;
  } else if (0 == (n & (n - 1))) {
    return r->Rand32() & (n - 1);
  } else {
    uint32 rem = (range % n) + 1;
    uint32 rnd;
    do {
      rnd = r->Rand32();  
    } while (rnd < rem);  
    return rnd % n;
  }
}
int WeightedPicker::Pick(SimplePhilox* rnd) const {
  if (total_weight() == 0) return -1;
  return PickAt(UnbiasedUniform(rnd, total_weight()));
}
int WeightedPicker::PickAt(int32_t weight_index) const {
  if (weight_index < 0 || weight_index >= total_weight()) return -1;
  int32_t position = weight_index;
  int index = 0;
  for (int l = 1; l < num_levels_; l++) {
    const int32_t left_weight = level_[l][2 * index];
    if (position < left_weight) {
      index = 2 * index;
    } else {
      index = 2 * index + 1;
      position -= left_weight;
    }
  }
  CHECK_GE(index, 0);
  CHECK_LT(index, N_);
  CHECK_LE(position, level_[num_levels_ - 1][index]);
  return index;
}
void WeightedPicker::set_weight(int index, int32_t weight) {
  assert(index >= 0);
  assert(index < N_);
  const int32_t delta = weight - get_weight(index);
  for (int l = num_levels_ - 1; l >= 0; l--) {
    level_[l][index] += delta;
    index >>= 1;
  }
}
void WeightedPicker::SetAllWeights(int32_t weight) {
  int32* leaves = level_[num_levels_ - 1];
  for (int i = 0; i < N_; i++) leaves[i] = weight;
  for (int i = N_; i < LevelSize(num_levels_ - 1); i++) leaves[i] = 0;
  RebuildTreeWeights();
}
void WeightedPicker::SetWeightsFromArray(int N, const int32* weights) {
  Resize(N);
  int32* leaves = level_[num_levels_ - 1];
  for (int i = 0; i < N_; i++) leaves[i] = weights[i];
  for (int i = N_; i < LevelSize(num_levels_ - 1); i++) leaves[i] = 0;
  RebuildTreeWeights();
}
void WeightedPicker::RebuildTreeWeights() {
  for (int l = num_levels_ - 2; l >= 0; l--) {
    int32* level = level_[l];
    int32* children = level_[l + 1];
    for (int i = 0; i < LevelSize(l); i++) {
      level[i] = children[2 * i] + children[2 * i + 1];
    }
  }
}
void WeightedPicker::Append(int32_t weight) {
  Resize(num_elements() + 1);
  set_weight(num_elements() - 1, weight);
}
void WeightedPicker::Resize(int new_size) {
  CHECK_GE(new_size, 0);
  if (new_size <= LevelSize(num_levels_ - 1)) {
    for (int i = new_size; i < N_; i++) {
      set_weight(i, 0);
    }
    N_ = new_size;
    return;
  }
  assert(new_size > N_);
  WeightedPicker new_picker(new_size);
  int32* dst = new_picker.level_[new_picker.num_levels_ - 1];
  int32* src = this->level_[this->num_levels_ - 1];
  memcpy(dst, src, sizeof(dst[0]) * N_);
  memset(dst + N_, 0, sizeof(dst[0]) * (new_size - N_));
  new_picker.RebuildTreeWeights();
  std::swap(new_picker.N_, this->N_);
  std::swap(new_picker.num_levels_, this->num_levels_);
  std::swap(new_picker.level_, this->level_);
  assert(this->N_ == new_size);
}
}  
}  