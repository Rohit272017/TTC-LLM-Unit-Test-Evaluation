#ifndef AROLLA_SERVING_EMBEDDED_MODEL_H_
#define AROLLA_SERVING_EMBEDDED_MODEL_H_
#include <functional>   
#include <type_traits>  
#include "absl/base/no_destructor.h"  
#include "absl/status/status.h"  
#include "absl/status/statusor.h"  
#include "absl/strings/str_format.h"  
#include "absl/strings/string_view.h"  
#include "arolla/util/init_arolla.h"
#include "arolla/util/meta.h"  
#include "arolla/util/status_macros_backport.h"
#define AROLLA_DEFINE_EMBEDDED_MODEL_FN(fn_name, model_or)                    \
  namespace {                                                                 \
  const decltype(model_or)& _arolla_embed_model_or_status_##fn_name() {       \
    using ModelT = decltype(model_or);                                        \
    static const absl::NoDestructor<ModelT> model(model_or);                  \
    return *model;                                                            \
  }                                                                           \
  }                                                                           \
                                                                              \
  const ::arolla::meta::strip_template_t<absl::StatusOr, decltype(model_or)>& \
  fn_name() {                                                                 \
    const auto& model = _arolla_embed_model_or_status_##fn_name();            \
                                                       \
    if (!model.ok()) {                                                        \
      static ::arolla::meta::strip_template_t<absl::StatusOr,                 \
                                              decltype(model_or)>             \
          error_fn =                                                          \
              [status_(model.status())](const auto&...) { return status_; };  \
      return error_fn;                                                        \
    }                                                                         \
    return *model;                                                            \
  }                                                                           \
                                                                              \
  AROLLA_INITIALIZER(                                                         \
          .deps =                                                             \
              {                                                               \
                  "@phony/serving_compiler_optimizer",                        \
                  ::arolla::initializer_dep::kOperators,                      \
                  ::arolla::initializer_dep::kS11n,                           \
              },                                                              \
          .init_fn = []() -> absl::Status {                                   \
            RETURN_IF_ERROR(                                                  \
                _arolla_embed_model_or_status_##fn_name().status())           \
                << "while initializing embedded model " << #fn_name << " at " \
                << __FILE__ << ":" << __LINE__;                               \
            return absl::OkStatus();                                          \
          })
#define AROLLA_DEFINE_EMBEDDED_MODEL_SET_FN(fn_name, model_set_or)             \
  namespace {                                                                  \
  const decltype(model_set_or)&                                                \
      _arolla_embed_model_set_or_status_##fn_name() {                          \
    using ModelSetT = decltype(model_set_or);                                  \
    static const absl::NoDestructor<ModelSetT> model_set(model_set_or);        \
    return *model_set;                                                         \
  }                                                                            \
  }                                                                            \
                                                                               \
  absl::StatusOr<std::reference_wrapper<                                       \
      const std::decay_t<decltype(model_set_or->at(""))>>>                     \
  fn_name(absl::string_view model_name) {                                      \
    const auto& model_set = _arolla_embed_model_set_or_status_##fn_name();     \
    RETURN_IF_ERROR(model_set.status());                                       \
    auto it = model_set->find(model_name);                                     \
    if (it == model_set->end()) {                                              \
      return absl::NotFoundError(                                              \
          absl::StrFormat("model \"%s\" not found in " #fn_name, model_name)); \
    }                                                                          \
    return it->second;                                                         \
  }                                                                            \
                                                                               \
  AROLLA_INITIALIZER(                                                          \
          .deps =                                                              \
              {                                                                \
                  "@phony/serving_compiler_optimizer",                         \
                  ::arolla::initializer_dep::kOperators,                       \
                  ::arolla::initializer_dep::kS11n,                            \
              },                                                               \
          .init_fn = []() -> absl::Status {                                    \
            RETURN_IF_ERROR(                                                   \
                _arolla_embed_model_set_or_status_##fn_name().status())        \
                << "while initializing embedded model " << #fn_name << " at "  \
                << __FILE__ << ":" << __LINE__;                                \
            return absl::OkStatus();                                           \
          })
#endif  