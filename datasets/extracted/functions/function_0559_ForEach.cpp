#ifndef AROLLA_OPERATORS_EXPERIMENTAL_DICT_H_
#define AROLLA_OPERATORS_EXPERIMENTAL_DICT_H_
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/dense_array/qtype/types.h"
#include "arolla/memory/optional_value.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qtype/dict/dict_types.h"
#include "arolla/util/view_types.h"
namespace arolla {
struct MakeKeyToRowDictOp {
  template <typename Key>
  absl::StatusOr<KeyToRowDict<Key>> operator()(
      const DenseArray<Key>& keys) const {
    typename KeyToRowDict<Key>::Map dict;
    dict.reserve(keys.size());
    absl::Status status;
    keys.ForEach([&](int64_t id, bool present, view_type_t<Key> key) {
      if (present) {
        if constexpr (std::is_floating_point_v<Key>) {
          if (std::isnan(key)) {
            status = absl::InvalidArgumentError("NaN dict keys are prohibited");
            return;
          }
        }
        auto [iter, inserted] = dict.emplace(Key{key}, id);
        if (!inserted) {
          status = absl::InvalidArgumentError(
              absl::StrFormat("duplicated key %s in the dict", Repr(Key{key})));
        }
      } else {
      }
    });
    if (status.ok()) {
      return KeyToRowDict<Key>(std::move(dict));
    } else {
      return status;
    }
  }
};
class DictGetRowOp {
 public:
  template <typename Key>
  OptionalValue<int64_t> operator()(const KeyToRowDict<Key>& dict,
                                    view_type_t<Key> key) const {
    if (auto iter = dict.map().find(key); iter != dict.map().end()) {
      return iter->second;
    } else {
      return std::nullopt;
    }
  }
};
class DictContainsOp {
 public:
  template <typename Key>
  OptionalUnit operator()(const KeyToRowDict<Key>& dict,
                          view_type_t<Key> key) const {
    return OptionalUnit{dict.map().contains(key)};
  }
};
class DictKeysOp {
 public:
  template <typename Key>
  absl::StatusOr<DenseArray<Key>> operator()(
      EvaluationContext* ctx, const KeyToRowDict<Key>& dict) const {
    DenseArrayBuilder<Key> result_builder(dict.map().size(),
                                          &ctx->buffer_factory());
    for (const auto& [key, row] : dict.map()) {
      if (row < 0 || row >= dict.map().size()) {
        return absl::InternalError(
            "unexpected row ids in the key-to-row mapping in the dict");
      }
      result_builder.Set(row, key);
    }
    DenseArray<Key> result = std::move(result_builder).Build();
    if (!result.IsFull()) {
      return absl::InternalError("incomplete key-to-row mapping in the dict");
    }
    return result;
  }
};
}  
#endif  