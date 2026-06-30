#include "tensorflow/compiler/tf2xla/lib/random.h"
#include <cmath>
#include <limits>
#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
namespace tensorflow {
xla::XlaOp TruncatedNormal(xla::XlaOp uniform) {
  const double kA = -2.0;
  const double kB = 2.0;
  const double kMu = 0.0;
  const double kSigma = 1.0;
  return ParameterizedTruncatedNormal(
      uniform, xla::ScalarLike(uniform, kMu), xla::ScalarLike(uniform, kSigma),
      xla::ScalarLike(uniform, kA), xla::ScalarLike(uniform, kB));
}
xla::XlaOp ParameterizedTruncatedNormal(xla::XlaOp uniform, xla::XlaOp mu,
                                        xla::XlaOp sigma, xla::XlaOp a,
                                        xla::XlaOp b) {
  xla::XlaOp one = xla::ScalarLike(uniform, 1.0);
  xla::XlaOp two = xla::ScalarLike(uniform, 2.0);
  xla::XlaOp sqrt_2 = xla::ScalarLike(uniform, std::sqrt(2.0));
  auto normal_cdf = [&](xla::XlaOp x) {
    return (one + xla::Erf(x / sqrt_2)) / two;
  };
  xla::XlaOp alpha = (a - mu) / sigma;
  xla::XlaOp beta = (b - mu) / sigma;
  xla::XlaOp alpha_normal_cdf = normal_cdf(alpha);
  xla::XlaOp beta_normal_cdf = normal_cdf(beta);
  xla::XlaOp p =
      alpha_normal_cdf + (beta_normal_cdf - alpha_normal_cdf) * uniform;
  xla::XlaOp v = two * p - one;
  xla::PrimitiveType primitive_type =
      uniform.builder()->GetShape(uniform).value().element_type();
  xla::XlaOp epsilon = xla::Epsilon(uniform.builder(), primitive_type);
  v = xla::Clamp(-one + epsilon, v, one - epsilon);
  xla::XlaOp x = mu + sigma * sqrt_2 * xla::ErfInv(v);
  x = xla::Clamp(a, x, b);
  return x;
}
}  