#include "xla/tsl/lib/random/distribution_sampler.h"
#include "xla/tsl/lib/random/philox_random.h"
namespace tsl {
namespace random {
template <>
void SingleSampleAdapter<PhiloxRandom>::SkipFromGenerator(uint64 num_skips) {
  generator_->Skip(num_skips);
}
}  
}  