#include "arolla/qexpr/operator_errors.h"
#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/typed_ref.h"
#include "arolla/qtype/typed_value.h"
namespace arolla {
namespace {
absl::Status SlotTypesMismatchError(absl::string_view slots_kind,
                                    absl::Span<const QTypePtr> expected_types,
                                    absl::Span<const QTypePtr> got_types) {
  return absl::FailedPreconditionError(absl::StrFormat(
      "incorrect %s types: expected %s, got %s", slots_kind,
      FormatTypeVector(expected_types), FormatTypeVector(got_types)));
}
template <typename T>
std::vector<QTypePtr> GetQTypes(absl::Span<const T> objects) {
  std::vector<QTypePtr> types;
  types.reserve(objects.size());
  for (const auto& o : objects) {
    types.push_back(o.GetType());
  }
  return types;
}
template <typename T>
absl::Status VerifyTypes(absl::Span<const T> objects,
                         absl::Span<const QTypePtr> expected_types,
                         absl::string_view slots_kind) {
  if (objects.size() != expected_types.size()) {
    return SlotTypesMismatchError(slots_kind, expected_types,
                                  GetQTypes(objects));
  }
  for (size_t i = 0; i < objects.size(); ++i) {
    if (objects[i].GetType() != expected_types[i]) {
      return SlotTypesMismatchError(slots_kind, expected_types,
                                    GetQTypes(objects));
    }
  }
  return absl::OkStatus();
}
}  
absl::Status OperatorNotDefinedError(absl::string_view operator_name,
                                     absl::Span<const QTypePtr> input_types,
                                     absl::string_view extra_message) {
  return absl::NotFoundError(absl::StrCat(
      "operator ", operator_name, " is not defined for argument types ",
      FormatTypeVector(input_types), extra_message.empty() ? "" : ": ",
      extra_message));
}
absl::Status VerifyInputSlotTypes(absl::Span<const TypedSlot> slots,
                                  absl::Span<const QTypePtr> expected_types) {
  return VerifyTypes(slots, expected_types, "input");
}
absl::Status VerifyOutputSlotType(TypedSlot slot, QTypePtr expected_type) {
  return VerifyTypes<TypedSlot>({slot}, {expected_type}, "output");
}
absl::Status VerifyInputValueTypes(absl::Span<const TypedValue> values,
                                   absl::Span<const QTypePtr> expected_types) {
  return VerifyTypes(values, expected_types, "input");
}
absl::Status VerifyInputValueTypes(absl::Span<const TypedRef> values,
                                   absl::Span<const QTypePtr> expected_types) {
  return VerifyTypes(values, expected_types, "input");
}
absl::Status VerifyOutputValueType(const TypedValue& value,
                                   QTypePtr expected_type) {
  return VerifyTypes<TypedValue>({value}, {expected_type}, "output");
}
std::string GuessLibraryName(absl::string_view operator_name) {
  std::string path = absl::StrReplaceAll(
      operator_name.substr(0, operator_name.rfind('.')), {{".", "/"}});
  return absl::StrCat("
}
std::string GuessOperatorLibraryName(absl::string_view operator_name) {
  return absl::StrFormat("%s:operator_%s", GuessLibraryName(operator_name),
                         absl::AsciiStrToLower(operator_name.substr(
                             operator_name.rfind('.') + 1)));
}
std::string SuggestMissingDependency() {
    return "adding \"@arolla:
           "build dependency may help";
}
std::string SuggestAvailableOverloads(
    absl::string_view operator_name,
    absl::Span<const QExprOperatorSignature* const> supported_qtypes) {
  std::vector<std::string> available_overloads;
  for (const auto type : supported_qtypes) {
    available_overloads.push_back(absl::StrFormat(
        "%s(%s) -> %s", operator_name, JoinTypeNames(type->input_types()),
        type->output_type()->name()));
  }
  return absl::StrFormat("available overloads:\n  %s",
                         absl::StrJoin(available_overloads, ",\n  "));
}
}  