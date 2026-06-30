#include "tensorstore/progress.h"
#include <ostream>
namespace tensorstore {
bool operator==(const ReadProgress& a, const ReadProgress& b) {
  return a.total_elements == b.total_elements &&
         a.copied_elements == b.copied_elements;
}
bool operator!=(const ReadProgress& a, const ReadProgress& b) {
  return !(a == b);
}
std::ostream& operator<<(std::ostream& os, const ReadProgress& a) {
  return os << "{ total_elements=" << a.total_elements
            << ", copied_elements=" << a.copied_elements << " }";
}
bool operator==(const WriteProgress& a, const WriteProgress& b) {
  return a.total_elements == b.total_elements &&
         a.copied_elements == b.copied_elements &&
         a.committed_elements == b.committed_elements;
}
bool operator!=(const WriteProgress& a, const WriteProgress& b) {
  return !(a == b);
}
std::ostream& operator<<(std::ostream& os, const WriteProgress& a) {
  return os << "{ total_elements=" << a.total_elements
            << ", copied_elements=" << a.copied_elements
            << ", committed_elements=" << a.committed_elements << " }";
}
bool operator==(const CopyProgress& a, const CopyProgress& b) {
  return a.total_elements == b.total_elements &&
         a.read_elements == b.read_elements &&
         a.copied_elements == b.copied_elements &&
         a.committed_elements == b.committed_elements;
}
bool operator!=(const CopyProgress& a, const CopyProgress& b) {
  return !(a == b);
}
std::ostream& operator<<(std::ostream& os, const CopyProgress& a) {
  return os << "{ total_elements=" << a.total_elements
            << ", read_elements=" << a.read_elements
            << ", copied_elements=" << a.copied_elements
            << ", committed_elements=" << a.committed_elements << " }";
}
}  