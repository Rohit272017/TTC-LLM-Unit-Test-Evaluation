#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_WRAPPER_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_WRAPPER_H_
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
namespace tflite {
namespace shim {
namespace op_wrapper {
using ::tflite::shim::OpKernelShim;
using ::tflite::shim::Runtime;
template <typename N, typename... T>
struct Attr {
  const char* Name() const { return N::Name(); }
};
template <char const* str>
struct AttrName {
  static const char* Name() { return str; }
};
template <typename T>
struct AttrType {
  using type = T;
};
template <typename T, typename... Us>
static constexpr std::tuple<T, Us...> prependTypeInner(T, std::tuple<Us...>);
template <typename T, typename... Us>
static constexpr auto prependType(T, std::tuple<Us...>)
    -> std::tuple<decltype(prependTypeInner(std::declval<T>(),
                                            std::declval<Us>()))...>;
template <typename Name, typename... Ts>
static constexpr std::tuple<std::tuple<Ts>...> getCombinations(
    Attr<Name, Ts...>);
template <typename Name, typename Head, typename... Attrs>
static constexpr auto getCombinations(Attr<Name, Head>, Attrs...)
    -> decltype(prependType(std::declval<Head>(),
                            getCombinations(std::declval<Attrs>()...)));
template <typename Name, typename Head, typename... Tail, typename... Attrs>
static constexpr auto getCombinations(Attr<Name, Head, Tail...>, Attrs...)
    -> decltype(std::tuple_cat(
        prependType(std::declval<Head>(),
                    getCombinations(std::declval<Attrs>()...)),
        getCombinations(std::declval<Attr<Name, Tail...>>(),
                        std::declval<Attrs>()...)));
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... Ts>
static constexpr Op<Rt, Ts...> convertTuplesToOpsInner(std::tuple<Ts...>);
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... Ts>
static constexpr auto convertTuplesToOps(std::tuple<Ts...>) -> std::tuple<
    decltype(convertTuplesToOpsInner<Rt, Op>(std::declval<Ts>()))...>;
template <typename... Ts>
static constexpr std::variant<Ts...> convertTupleToVariant(std::tuple<Ts...>);
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename FirstAttr, typename... OtherAttrs>
struct VariantOp {
  using type =
      decltype(convertTupleToVariant(convertTuplesToOps<Rt, Op>(getCombinations(
          std::declval<FirstAttr>(), std::declval<OtherAttrs>()...))));
};
template <Runtime Rt>
class OpWrapperExtension : public OpKernelShim<OpWrapperExtension, Rt> {};
template <Runtime Rt, template <Runtime, typename...> typename Op,
          typename... As>
class OpWrapper : public OpWrapperExtension<Rt> {
 public:
  using TmplOpType = typename VariantOp<Rt, Op, As...>::type;
  using TmplOpType0 = typename std::variant_alternative<0, TmplOpType>::type;
  using typename OpKernelShim<OpWrapperExtension, Rt>::InitContext;
  using typename OpKernelShim<OpWrapperExtension, Rt>::InvokeContext;
  using typename OpKernelShim<OpWrapperExtension, Rt>::ShapeInferenceContext;
  OpWrapper() = default;
  static const char* OpName() { return TmplOpType0::OpName(); }
  static const char* Doc() { return TmplOpType0::Doc(); }
  static std::vector<std::string> Attrs() { return TmplOpType0::Attrs(); }
  static std::vector<std::string> Inputs() { return TmplOpType0::Inputs(); }
  static std::vector<std::string> Outputs() { return TmplOpType0::Outputs(); }
  static absl::Status ShapeInference(ShapeInferenceContext* context) {
    return TmplOpType0::ShapeInference(context);
  }
  absl::Status Init(InitContext* context) {
    SH_RETURN_IF_ERROR(SetVariantOp<As...>(context));
    return std::visit(
        [context](auto&& op) -> absl::Status { return op.Init(context); },
        *op_);
  }
  absl::Status Invoke(InvokeContext* context) {
    return std::visit(
        [context](auto&& op) -> absl::Status { return op.Invoke(context); },
        *op_);
  }
 private:
  template <typename FirstAttr, typename... Attrs>
  absl::Status SetVariantOp(InitContext* c) {
    return CombineAttributeTypes(this, c, FirstAttr{}, Attrs{}...);
  }
  template <typename F, typename Name, typename T>
  struct Forwarder {
   public:
    explicit Forwarder(F* f) : inner(f) {}
    template <typename... Args>
    absl::Status SetOpCombination(Args... args) {
      return inner->SetOpCombination(Name::Name(), AttrType<T>{}, args...);
    }
   private:
    F* inner;
  };
  template <typename F, typename Name, typename Head, typename... Tail,
            typename... Attrs>
  absl::Status CombineAttributeTypes(F* obj, InitContext* c,
                                     Attr<Name, Head, Tail...>, Attrs... rest) {
    SH_RETURN_IF_ERROR(
        ApplyAttrType(obj, c, Name{}, AttrType<Head>{}, rest...));
    return CombineAttributeTypes(obj, c, Attr<Name, Tail...>{}, rest...);
  }
  template <typename F, typename Name, typename... Attrs>
  absl::Status CombineAttributeTypes(F*, InitContext*, Attr<Name>, Attrs...) {
    return absl::OkStatus();
  }
  template <typename F, typename Name, typename T, typename Attr,
            typename... Attrs>
  absl::Status ApplyAttrType(F* obj, InitContext* c, Name, AttrType<T>, Attr a,
                             Attrs... rest) {
    Forwarder<F, Name, T> forwarder(obj);
    return CombineAttributeTypes(&forwarder, c, a, rest...);
  }
  template <typename F, typename Name, typename T>
  absl::Status ApplyAttrType(F* obj, InitContext* c, Name, AttrType<T> t) {
    return obj->SetOpCombination(Name::Name(), t, c);
  }
  template <typename T>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                InitContext* context) {
    int64_t datatype_1;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    if (datatype_1 == typeToTfLiteType<T>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T>());
    }
    return absl::OkStatus();
  }
  template <typename T, typename U>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U>());
    }
    return absl::OkStatus();
  }
  template <typename T, typename U, typename V>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V>());
    }
    return absl::OkStatus();
  }
  template <typename T, typename U, typename V, typename W>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                std::string Name4, AttrType<W>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3, datatype_4;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    SH_RETURN_IF_ERROR(context->GetAttr(Name4, &datatype_4));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>() &&
        datatype_4 == typeToTfLiteType<W>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V, W>());
    }
    return absl::OkStatus();
  }
  template <typename T, typename U, typename V, typename W, typename X>
  absl::Status SetOpCombination(std::string Name1, AttrType<T>,
                                std::string Name2, AttrType<U>,
                                std::string Name3, AttrType<V>,
                                std::string Name4, AttrType<W>,
                                std::string Name5, AttrType<X>,
                                InitContext* context) {
    int64_t datatype_1, datatype_2, datatype_3, datatype_4, datatype_5;
    SH_RETURN_IF_ERROR(context->GetAttr(Name1, &datatype_1));
    SH_RETURN_IF_ERROR(context->GetAttr(Name2, &datatype_2));
    SH_RETURN_IF_ERROR(context->GetAttr(Name3, &datatype_3));
    SH_RETURN_IF_ERROR(context->GetAttr(Name4, &datatype_4));
    SH_RETURN_IF_ERROR(context->GetAttr(Name5, &datatype_5));
    if (datatype_1 == typeToTfLiteType<T>() &&
        datatype_2 == typeToTfLiteType<U>() &&
        datatype_3 == typeToTfLiteType<V>() &&
        datatype_4 == typeToTfLiteType<W>() &&
        datatype_5 == typeToTfLiteType<X>()) {
      this->op_ = std::make_unique<TmplOpType>(Op<Rt, T, U, V, W, X>());
    }
    return absl::OkStatus();
  }
 protected:
  std::unique_ptr<TmplOpType> op_;
};
}  
}  
}  
#endif  