#include "tsl/platform/random.h"
#include <memory>
#include <random>
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace random {
namespace {
std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}
std::mt19937_64 InitRngWithDefaultSeed() { return std::mt19937_64(); }
}  
uint64 New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  static mutex mu(LINKER_INITIALIZED);
  mutex_lock l(mu);
  return (*rng)();
}
uint64 ThreadLocalNew64() {
  static thread_local std::unique_ptr<std::mt19937_64> rng =
      std::unique_ptr<std::mt19937_64>(InitRngWithRandomSeed());
  return (*rng)();
}
uint64 New64DefaultSeed() {
  static std::mt19937_64 rng = InitRngWithDefaultSeed();
  static mutex mu(LINKER_INITIALIZED);
  mutex_lock l(mu);
  return rng();
}
}  
}  