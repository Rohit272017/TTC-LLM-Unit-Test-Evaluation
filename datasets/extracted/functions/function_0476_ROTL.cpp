#include "tensorflow/core/kernels/random_index_shuffle.h"
#include <assert.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace random {
constexpr int kMinBlockSize = 16;
namespace impl {
#define ROTL(x, r, W) (((x) << (r)) | (x >> (W - (r))))
#define ROTR(x, r, W) (((x) >> (r)) | ((x) << (W - (r))))
#define SIMON_F(x, W) ((ROTL(x, 1, W) & ROTL(x, 8, W) ^ ROTL(x, 2, W)))
#define SIMON_Rx2(x, y, k1, k2, W) \
  (y ^= SIMON_F(x, W), y ^= k1, x ^= SIMON_F(y, W), x ^= k2)
template <int W>
std::vector<std::bitset<W>> simon_key_schedule(
    const std::array<uint32_t, 3>& key, const int32_t rounds) {
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  const auto c = std::bitset<W>(0xfffffffc);
  auto z = std::bitset<W>(0x7369f885192c0ef5LL);
  std::vector<std::bitset<W>> rk({key[0], key[1], key[2]});
  rk.reserve(rounds);
  for (int i = 3; i < rounds; i++) {
    rk.push_back(c ^ (z & std::bitset<W>(1)) ^ rk[i - 3] ^
                 ROTR(rk[i - 1], 3, W) ^ ROTR(rk[i - 1], 4, W));
    z >>= 1;
  }
  return rk;
}
template <int W>
uint64_t simon_encrypt(const uint64_t value,
                       const std::vector<std::bitset<W>>& round_keys) {
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  std::bitset<W> left(value >> W);
  std::bitset<W> right(value);
  for (int i = 0; i < round_keys.size();) {
    SIMON_Rx2(right, left, round_keys[i++], round_keys[i++], W);
  }
  return (left.to_ullong() << W) | right.to_ullong();
}
template <int B>
uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index, const int32_t rounds) {
  const auto round_keys = simon_key_schedule<B / 2>(key, rounds);
  uint64_t new_index = index;
  while (true) {
    new_index = simon_encrypt<B / 2>(new_index, round_keys);
    if (new_index <= max_index) {
      return new_index;
    }
  }
}
#undef ROTL
#undef ROTR
#undef SIMON_F
#undef SIMON_RxC
}  
uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index, const int32_t rounds) {
  int block_size = static_cast<int>(std::ceil(std::log2(max_index)));
  block_size = std::max(block_size + block_size % 2, kMinBlockSize);
  assert(block_size > 0 && block_size % 2 == 0 && block_size <= 64);
  assert(rounds >= 4 && rounds % 2 == 0);
#define HANDLE_BLOCK_SIZE(B) \
  case B:                    \
    return impl::index_shuffle<B>(index, key, max_index, rounds);
  switch (block_size) {
    HANDLE_BLOCK_SIZE(16);
    HANDLE_BLOCK_SIZE(18);
    HANDLE_BLOCK_SIZE(20);
    HANDLE_BLOCK_SIZE(22);
    HANDLE_BLOCK_SIZE(24);
    HANDLE_BLOCK_SIZE(26);
    HANDLE_BLOCK_SIZE(28);
    HANDLE_BLOCK_SIZE(30);
    HANDLE_BLOCK_SIZE(32);
    HANDLE_BLOCK_SIZE(34);
    HANDLE_BLOCK_SIZE(36);
    HANDLE_BLOCK_SIZE(38);
    HANDLE_BLOCK_SIZE(40);
    HANDLE_BLOCK_SIZE(42);
    HANDLE_BLOCK_SIZE(44);
    HANDLE_BLOCK_SIZE(46);
    HANDLE_BLOCK_SIZE(48);
    HANDLE_BLOCK_SIZE(50);
    HANDLE_BLOCK_SIZE(52);
    HANDLE_BLOCK_SIZE(54);
    HANDLE_BLOCK_SIZE(56);
    HANDLE_BLOCK_SIZE(58);
    HANDLE_BLOCK_SIZE(60);
    HANDLE_BLOCK_SIZE(62);
    default:
      return impl::index_shuffle<64>(index, key, max_index, rounds);
  }
#undef HANDLE_BLOCK_SIZE
}
}  
}  