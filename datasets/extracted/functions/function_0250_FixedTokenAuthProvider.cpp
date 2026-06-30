#include "tensorstore/internal/oauth2/fixed_token_auth_provider.h"
#include "absl/time/time.h"
#include "tensorstore/internal/oauth2/bearer_token.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
namespace internal_oauth2 {
FixedTokenAuthProvider::FixedTokenAuthProvider(std::string token)
    : token_(token) {}
Result<BearerTokenWithExpiration> FixedTokenAuthProvider::GetToken() {
  return BearerTokenWithExpiration{token_, absl::InfiniteFuture()};
}
}  
}  