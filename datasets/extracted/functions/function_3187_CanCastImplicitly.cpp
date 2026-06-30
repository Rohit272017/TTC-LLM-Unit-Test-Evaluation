#include "arolla/qexpr/casting.h"
#include <cstddef>
#include <utility>
#include <vector>
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/qexpr/operator_errors.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qtype/derived_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/standard_type_properties/common_qtype.h"
namespace arolla {
namespace {
bool CanCastImplicitly(absl::Span<const QTypePtr> from_types,
                       absl::Span<const QTypePtr> to_types) {
  if (from_types.size() != to_types.size()) {
    return false;
  }
  for (size_t i = 0; i < to_types.size(); ++i) {
    if (!CanCastImplicitly(from_types[i], to_types[i],
                           true)) {
      return false;
    }
  }
  return true;
}
struct SignatureFormatter {
  void operator()(std::string* out,
                  const QExprOperatorSignature* signature) const {
    absl::StrAppend(out, signature);
  }
};
}  
absl::StatusOr<const QExprOperatorSignature*> FindMatchingSignature(
    absl::Span<const QTypePtr> input_types, QTypePtr output_type,
    absl::Span<const QExprOperatorSignature* const> supported_signatures,
    absl::string_view op_name) {
  const QTypePtr decayed_output_type = DecayDerivedQType(output_type);
  absl::InlinedVector<QTypePtr, 6> decayed_input_types(input_types.size());
  for (size_t i = 0; i < input_types.size(); ++i) {
    decayed_input_types[i] = DecayDerivedQType(input_types[i]);
  }
  absl::InlinedVector<const QExprOperatorSignature*, 8> frontier;
  for (const auto& candidate : supported_signatures) {
    if (decayed_output_type != DecayDerivedQType(candidate->output_type())) {
      continue;
    }
    if (!CanCastImplicitly(input_types, candidate->input_types())) {
      continue;
    }
    if (decayed_input_types == candidate->input_types()) {
      return candidate;
    }
    bool dominates = false;
    bool dominated = false;
    auto out_it = frontier.begin();
    for (auto* previous : frontier) {
      if (CanCastImplicitly(candidate->input_types(),
                            previous->input_types())) {
        dominates = true;  
      } else if (dominates || !CanCastImplicitly(previous->input_types(),
                                                 candidate->input_types())) {
        *out_it++ = previous;  
      } else {
        dominated = true;  
        break;
      }
    }
    if (dominates) {
      frontier.erase(out_it, frontier.end());
    }
    if (!dominated) {
      frontier.push_back(candidate);
    }
  }
  if (frontier.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "QExpr operator %s%v not found; %s\n%s", op_name,
        QExprOperatorSignature::Get(input_types, output_type),
        SuggestMissingDependency(),
        SuggestAvailableOverloads(op_name, supported_signatures)));
  }
  if (frontier.size() > 1) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "ambiguous overloads for the QExpr operator %s%v: provided argument "
        "types can be cast to the following supported signatures: %s ",
        op_name, QExprOperatorSignature::Get(input_types, output_type),
        absl::StrJoin(frontier, ", ", SignatureFormatter())));
  }
  return frontier[0];
}
}  