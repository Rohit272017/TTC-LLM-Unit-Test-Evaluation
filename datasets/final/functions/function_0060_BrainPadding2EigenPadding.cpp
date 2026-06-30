#include <algorithm>
#include <cmath>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/padding.h"
namespace tensorflow {
Eigen::PaddingType BrainPadding2EigenPadding(Padding padding) {
  switch (padding) {
    case Padding::VALID:
      return Eigen::PADDING_VALID;
    case Padding::SAME:
      return Eigen::PADDING_SAME;
    case Padding::EXPLICIT:
      LOG(FATAL) << "Eigen does not have explicit padding enum "  
                    "value";
  }
  return Eigen::PADDING_SAME;  
}
Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize) {
  if (index * stride > in_size) {
    return errors::InvalidArgument(
        "index * stride must be less than or equal to input size");
  }
  *bindex = index * stride;
  *bsize = ksize;
  if (*bindex < pad_size) {
    *bsize = ksize + *bindex - pad_size;
    *bindex = 0;
  } else {
    *bindex -= pad_size;
  }
  if (*bindex + ksize > in_size) {
    *bsize = std::min((in_size - *bindex), ksize);
  }
  return absl::OkStatus();
}
string SanitizeThreadSuffix(string suffix) {
  string clean;
  for (int i = 0; i < suffix.size(); ++i) {
    const char ch = suffix[i];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
      clean += ch;
    } else {
      clean += '_';
    }
  }
  return clean;
}
}  