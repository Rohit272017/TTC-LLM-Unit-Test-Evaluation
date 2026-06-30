#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OVERLOAD_H_
namespace shlo_ref {
template <class... Ts>
class Overload : public Ts... {
 public:
  explicit Overload(Ts&&... ts) : Ts(static_cast<Ts&&>(ts))... {}
  using Ts::operator()...;
};
template <class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;
}  
#endif  