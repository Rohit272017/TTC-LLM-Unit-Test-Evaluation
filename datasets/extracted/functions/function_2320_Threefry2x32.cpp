#include "tensorflow/lite/kernels/rng_util.h"
#include <array>
#include <cstdint>
namespace tflite {
namespace rng {
static constexpr uint32_t kThreefryParity = 0x1BD11BDA;
static constexpr uint64_t kPhiloxM4x32A = 0xD2511F53;
static constexpr uint64_t kPhiloxM4x32B = 0xCD9E8D57;
static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
std::array<uint32_t, 2> Threefry2x32(uint32_t key_0, uint32_t key_1,
                                     std::array<uint32_t, 2> ctr) {
  constexpr std::array<std::array<int, 4>, 2> rotations{
      std::array<int, 4>{13, 15, 26, 6}, std::array<int, 4>{17, 29, 16, 24}};
  uint32_t key_2 = key_0 ^ key_1 ^ kThreefryParity;
  ctr[0] += key_0;
  ctr[1] += key_1;
  auto apply_round = [&](int r, uint32_t ks0, uint32_t ks1, int b) {
    for (int rot : rotations[r]) {
      ctr[0] += ctr[1];
      ctr[1] = (ctr[1] << rot) | (ctr[1] >> (32 - rot));
      ctr[1] ^= ctr[0];
    }
    ctr[0] += ks0;
    ctr[1] += ks1 + b;
  };
  apply_round(0, key_1, key_2, 1);
  apply_round(1, key_2, key_0, 2);
  apply_round(0, key_0, key_1, 3);
  apply_round(1, key_1, key_2, 4);
  apply_round(0, key_2, key_0, 5);
  return ctr;
}
std::array<uint32_t, 4> Philox4x32(uint32_t key_0, uint32_t key_1,
                                   std::array<uint32_t, 4> ctr) {
  struct u32pair {
    uint32_t low;
    uint32_t high;
  };
  union prod {
    u32pair hilo;
    uint64_t prod;
  };
  for (int i = 0; i < 10; ++i) {
    prod p0, p1;
    p0.prod = kPhiloxM4x32A * static_cast<uint64_t>(ctr[0]);
    p1.prod = kPhiloxM4x32B * static_cast<uint64_t>(ctr[2]);
    ctr = {{p1.hilo.high ^ ctr[1] ^ key_0, p1.hilo.low,
            p0.hilo.high ^ ctr[3] ^ key_1, p0.hilo.low}};
    key_0 += kPhiloxW32A;
    key_1 += kPhiloxW32B;
  }
  return ctr;
}
}  
}  