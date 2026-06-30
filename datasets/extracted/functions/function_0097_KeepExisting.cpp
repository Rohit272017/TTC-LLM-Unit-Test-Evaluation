#include "tensorflow/core/framework/full_type_inference_util.h"
#include <functional>
#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
namespace tensorflow {
namespace full_type {
TypeInferenceFn KeepExisting() { return nullptr; }
TypeInferenceFn Tensor(FullTypeId t) {
  return [t](const TypeRefVector& input_types,
             const FunctionTypeInferrer& infer_function_rets) {
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    ret_type.add_args()->set_type_id(TFT_TENSOR);
    ret_type.mutable_args(0)->add_args()->set_type_id(t);
    return ret_type;
  };
}
TypeInferenceFn ReplicateInput(int i, int n) {
  return [i, n](const TypeRefVector& input_types,
                const FunctionTypeInferrer& infer_function_rets) {
    const FullTypeDef& in_type = input_types.at(i).get();
    FullTypeDef ret_type;
    if (in_type.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      for (int k = 0; k < n; k++) {
        *(ret_type.add_args()) = in_type;
      }
    }
    return ret_type;
  };
}
TypeInferenceFn Merge() {
  return [](const TypeRefVector& input_types,
            const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK(!input_types.empty());
    FullTypeDef merged;
    for (int i = 0; i < input_types.size(); i++) {
      const auto& t = input_types[i].get();
      if (t.type_id() == TFT_UNSET) {
        continue;
      }
      if (IsSubtype(t, merged)) {
        merged = t;
        continue;
      }
      if (IsSubtype(merged, t)) {
        continue;
      }
      return Status(absl::StatusCode::kInvalidArgument,
                    absl::StrCat("expected compatible input types, but input ",
                                 i, ":\n", t.DebugString(),
                                 " is neither a subtype nor a supertype of the "
                                 "combined inputs preceding it:\n",
                                 merged.DebugString()));
    }
    FullTypeDef ret_type;
    if (merged.type_id() != TFT_UNSET) {
      ret_type.set_type_id(TFT_PRODUCT);
      *(ret_type.add_args()) = merged;
    }
    return ret_type;
  };
}
TypeInferenceFn Encode(FullTypeId t, int i) {
  return [t, i](const TypeRefVector& input_types,
                const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= i);
    FullTypeDef ret_type;
    const FullTypeDef& in_t = input_types[i].get();
    if (in_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    ret_type.set_type_id(TFT_PRODUCT);
    auto* enc_type = ret_type.add_args();
    enc_type->set_type_id(TFT_ENCODED);
    *enc_type->add_args() = in_t;
    enc_type->add_args()->set_type_id(t);
    return ret_type;
  };
}
TypeInferenceFn Decode(FullTypeId t, int i) {
  return [t, i](const TypeRefVector& input_types,
                const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= i);
    const FullTypeDef& in_t = input_types[i].get();
    const FullTypeId enc_tid = GetArgDefaultUnset(in_t, 1).type_id();
    if ((enc_tid != TFT_UNSET) && (enc_tid != t)) {
      return Status(absl::StatusCode::kInvalidArgument,
                    absl::StrCat("expected encoded type ", t, " for input ", i,
                                 ", got ", in_t.DebugString()));
    }
    FullTypeDef ret_type;
    const FullTypeDef& out_t = GetArgDefaultUnset(in_t, 0);
    if (in_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    ret_type.set_type_id(TFT_PRODUCT);
    *ret_type.add_args() = out_t;
    return ret_type;
  };
}
TypeInferenceFn UnaryContainerCreate(FullTypeId t, int element_idx) {
  return [t, element_idx](const TypeRefVector& input_types,
                          const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= element_idx);
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* arg_t = ret_type.add_args();
    arg_t->set_type_id(t);
    *(arg_t->add_args()) = input_types[element_idx].get();
    return ret_type;
  };
}
TypeInferenceFn UnaryContainerAdd(FullTypeId t, int container_idx,
                                  int element_idx, bool homogeneous) {
  return [t, container_idx, element_idx, homogeneous](
             const TypeRefVector& input_types,
             const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK(input_types.size() >= container_idx);
    DCHECK(input_types.size() >= element_idx);
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);
    const FullTypeDef& in_cont_t = input_types[container_idx].get();
    const FullTypeDef& in_el_t = input_types[element_idx].get();
    if (in_cont_t.type_id() != TFT_UNSET) {
      if (in_cont_t.type_id() != t) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            absl::StrCat("expected container type ", t, " for input ",
                         container_idx, ", got ", in_cont_t.DebugString()));
      }
      *cont_t = in_cont_t;
    }
    VLOG(1) << "ContainerAddUnary: " << cont_t->DebugString() << ", "
            << in_el_t.DebugString() << ", " << container_idx << "; "
            << element_idx;
    for (const auto& tmp : input_types) {
      VLOG(1) << "  input: " << tmp.get().DebugString();
    }
    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    const FullTypeDef& el_t = GetArgDefaultUnset(*cont_t, 0);
    if (el_t.type_id() == TFT_UNSET) {
      cont_t->clear_args();
      *(cont_t->add_args()) = in_el_t;
      return ret_type;
    }
    if (IsSubtype(in_el_t, el_t)) {
      return ret_type;
    }
    if (homogeneous) {
      return Status(absl::StatusCode::kInvalidArgument,
                    absl::StrCat("expected a subtype of ", el_t.DebugString(),
                                 " for input ", element_idx,
                                 " of a homogeneous container ", t, ", got ",
                                 in_el_t.DebugString()));
    } else {
      return Status(
          absl::StatusCode::kUnimplemented,
          absl::StrCat("need union types for heterogeneous containers.\n"
                       "A homogeneous container would expect a subtype of ",
                       el_t.DebugString(), " for input ", element_idx,
                       ", but got ", in_el_t.DebugString()));
    }
  };
}
TypeInferenceFn MultiaryUnstack(
    FullTypeId t, std::function<FullTypeDef(const FullTypeDef&)> unstack) {
  return [t, unstack](const TypeRefVector& input_types,
                      const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* cont_t = ret_type.add_args();
    cont_t->set_type_id(t);
    FullTypeDef* el_t = cont_t->add_args();
    el_t->set_type_id(TFT_PRODUCT);
    for (int element_idx = 0; element_idx < input_types.size(); ++element_idx) {
      *(el_t->add_args()) = unstack(input_types[element_idx].get());
    }
    return ret_type;
  };
}
FullTypeDef UnstackTensor(const FullTypeDef& t) {
  DCHECK((t.type_id() == TFT_TENSOR) || (t.type_id() == TFT_RAGGED) ||
         (t.type_id() == TFT_UNSET));
  DCHECK_LE(t.args_size(), 1);
  return t;
}
TypeInferenceFn ContainerMap(
    FullTypeId t, int input_idx,
    std::function<FullTypeDef(const FullTypeDef&)> map) {
  return [t, input_idx, map](const TypeRefVector& input_types,
                             const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    DCHECK_GE(input_types.size(), input_idx);
    const FullTypeDef& in_cont_t = input_types.at(input_idx).get();
    FullTypeDef ret_type;
    if (in_cont_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_cont_t.type_id() != t) {
      return Status(absl::StatusCode::kInvalidArgument,
                    absl::StrCat("expected type ", t, " for input ", input_idx,
                                 ", got ", in_cont_t.DebugString()));
    }
    ret_type.set_type_id(TFT_PRODUCT);
    FullTypeDef* out_cont_t = ret_type.add_args();
    out_cont_t->set_type_id(t);
    const FullTypeDef& in_el_t = GetArgDefaultUnset(in_cont_t, 0);
    if (in_el_t.type_id() == TFT_UNSET) {
      return ret_type;
    }
    if (in_el_t.type_id() != TFT_PRODUCT) {
      return Status(absl::StatusCode::kInvalidArgument,
                    absl::StrCat("expected PRODUCT element type for input ",
                                 input_idx, ", got ", in_el_t.DebugString()));
    }
    FullTypeDef* out_el_t = out_cont_t->add_args();
    out_el_t->set_type_id(TFT_PRODUCT);
    for (int k = 0; k < in_el_t.args_size(); k++) {
      *(out_el_t->add_args()) = map(in_el_t.args(k));
    }
    return ret_type;
  };
}
TypeInferenceFn MapCovariant(FullTypeId t, FullTypeId u, int input_idx) {
  return
      [t, u, input_idx](const TypeRefVector& input_types,
                        const FunctionTypeInferrer& infer_function_rets)
          -> absl::StatusOr<FullTypeDef> {
        DCHECK_GE(input_types.size(), input_idx);
        const FullTypeDef& in_t = input_types.at(input_idx).get();
        FullTypeDef ret_type;
        if (in_t.type_id() == TFT_UNSET) {
          return ret_type;
        }
        if (in_t.type_id() != t) {
          return Status(absl::StatusCode::kInvalidArgument,
                        absl::StrCat("expected type ", t, " for input ",
                                     input_idx, ", got ", in_t.DebugString()));
        }
        ret_type.set_type_id(TFT_PRODUCT);
        FullTypeDef* t = ret_type.add_args();
        t->set_type_id(u);
        *t->mutable_args() = in_t.args();
        return ret_type;
      };
}
TypeInferenceFn FunctionCall(const string& func_attr_name) {
  return [func_attr_name](const TypeRefVector& input_types,
                          const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    return infer_function_rets(func_attr_name, input_types);
  };
}
TypeInferenceFn Tuple(const std::vector<TypeInferenceFn>& func_list) {
  return [func_list](const TypeRefVector& input_types,
                     const FunctionTypeInferrer& infer_function_rets)
             -> absl::StatusOr<FullTypeDef> {
    FullTypeDef ret_type;
    ret_type.set_type_id(TFT_PRODUCT);
    for (const auto& func : func_list) {
      const auto& status_or_t = func(input_types, infer_function_rets);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          status_or_t.status(),
          absl::StrCat("for Tuple type infernce function ",
                       ret_type.args_size()));
      const FullTypeDef& t = status_or_t.value();
      if (t.type_id() == TFT_UNSET) {
        VLOG(1) << "For Tuple type inference function, function "
                << ret_type.args_size() << " is unset.";
        FullTypeDef unset_type;
        return unset_type;
      }
      if (t.type_id() != TFT_PRODUCT) {
        return Status(
            absl::StatusCode::kInvalidArgument,
            absl::StrCat("for Tuple type inference function, expected result "
                         "of type inference function ",
                         ret_type.args_size(),
                         " to start with TFT_PRODUCT not ", t.DebugString()));
      }
      for (int i = 0; i < t.args_size(); i++) {
        *(ret_type.add_args()) = t.args(i);
      }
    }
    return ret_type;
  };
}
FullTypeDef BatchTensor(const FullTypeDef& t) {
  return t;
}
FullTypeDef ShardTensor(const FullTypeDef& t) {
  return t;
}
}  
}  