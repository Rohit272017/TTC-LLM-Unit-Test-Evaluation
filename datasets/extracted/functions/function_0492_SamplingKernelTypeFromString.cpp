#include "tensorflow/core/kernels/image/sampling_kernels.h"
#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
namespace tensorflow {
namespace functor {
SamplingKernelType SamplingKernelTypeFromString(const StringPiece str) {
  const string lower_case = absl::AsciiStrToLower(str);
  if (lower_case == "lanczos1") return Lanczos1Kernel;
  if (lower_case == "lanczos3") return Lanczos3Kernel;
  if (lower_case == "lanczos5") return Lanczos5Kernel;
  if (lower_case == "gaussian") return GaussianKernel;
  if (lower_case == "box") return BoxKernel;
  if (lower_case == "triangle") return TriangleKernel;
  if (lower_case == "keyscubic") return KeysCubicKernel;
  if (lower_case == "mitchellcubic") return MitchellCubicKernel;
  return SamplingKernelTypeEnd;
}
}  
}  