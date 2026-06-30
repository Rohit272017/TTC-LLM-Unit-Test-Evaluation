#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_CLEANUP_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_CLEANUP_H_
#include <type_traits>
#include <utility>
namespace tf_gcs_filesystem {
template <typename F>
class Cleanup {
 public:
  Cleanup() : released_(true), f_() {}
  template <typename G>
  explicit Cleanup(G&& f)          
      : f_(std::forward<G>(f)) {}  
  Cleanup(Cleanup&& src)  
      : released_(src.is_released()), f_(src.release()) {}
  template <typename G>
  Cleanup(Cleanup<G>&& src)  
      : released_(src.is_released()), f_(src.release()) {}
  Cleanup& operator=(Cleanup&& src) {  
    if (!released_) f_();
    released_ = src.released_;
    f_ = src.release();
    return *this;
  }
  ~Cleanup() {
    if (!released_) f_();
  }
  F release() {
    released_ = true;
    return std::move(f_);
  }
  bool is_released() const { return released_; }
 private:
  static_assert(!std::is_reference<F>::value, "F must not be a reference");
  bool released_ = false;
  F f_;
};
template <int&... ExplicitParameterBarrier, typename F,
          typename DecayF = typename std::decay<F>::type>
Cleanup<DecayF> MakeCleanup(F&& f) {
  return Cleanup<DecayF>(std::forward<F>(f));
}
}  
#endif  