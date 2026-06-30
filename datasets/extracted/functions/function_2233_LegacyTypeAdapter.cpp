#ifndef THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_STRUCTS_LEGACY_TYPE_ADPATER_H_
#define THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_STRUCTS_LEGACY_TYPE_ADPATER_H_
#include <cstdint>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "base/attribute.h"
#include "common/memory.h"
#include "eval/public/cel_options.h"
#include "eval/public/cel_value.h"
namespace google::api::expr::runtime {
class LegacyTypeMutationApis {
 public:
  virtual ~LegacyTypeMutationApis() = default;
  virtual bool DefinesField(absl::string_view field_name) const = 0;
  virtual absl::StatusOr<CelValue::MessageWrapper::Builder> NewInstance(
      cel::MemoryManagerRef memory_manager) const = 0;
  virtual absl::StatusOr<CelValue> AdaptFromWellKnownType(
      cel::MemoryManagerRef memory_manager,
      CelValue::MessageWrapper::Builder instance) const = 0;
  virtual absl::Status SetField(
      absl::string_view field_name, const CelValue& value,
      cel::MemoryManagerRef memory_manager,
      CelValue::MessageWrapper::Builder& instance) const = 0;
  virtual absl::Status SetFieldByNumber(
      int64_t field_number, const CelValue& value,
      cel::MemoryManagerRef memory_manager,
      CelValue::MessageWrapper::Builder& instance) const {
    return absl::UnimplementedError("SetFieldByNumber is not yet implemented");
  }
};
class LegacyTypeAccessApis {
 public:
  struct LegacyQualifyResult {
    CelValue value;
    int qualifier_count;
  };
  virtual ~LegacyTypeAccessApis() = default;
  virtual absl::StatusOr<bool> HasField(
      absl::string_view field_name,
      const CelValue::MessageWrapper& value) const = 0;
  virtual absl::StatusOr<CelValue> GetField(
      absl::string_view field_name, const CelValue::MessageWrapper& instance,
      ProtoWrapperTypeOptions unboxing_option,
      cel::MemoryManagerRef memory_manager) const = 0;
  virtual absl::StatusOr<LegacyQualifyResult> Qualify(
      absl::Span<const cel::SelectQualifier>,
      const CelValue::MessageWrapper& instance, bool presence_test,
      cel::MemoryManagerRef memory_manager) const {
    return absl::UnimplementedError("Qualify unsupported.");
  }
  virtual bool IsEqualTo(const CelValue::MessageWrapper&,
                         const CelValue::MessageWrapper&) const {
    return false;
  }
  virtual std::vector<absl::string_view> ListFields(
      const CelValue::MessageWrapper& instance) const = 0;
};
class LegacyTypeAdapter {
 public:
  LegacyTypeAdapter(const LegacyTypeAccessApis* access,
                    const LegacyTypeMutationApis* mutation)
      : access_apis_(access), mutation_apis_(mutation) {}
  const LegacyTypeAccessApis* access_apis() { return access_apis_; }
  const LegacyTypeMutationApis* mutation_apis() { return mutation_apis_; }
 private:
  const LegacyTypeAccessApis* access_apis_;
  const LegacyTypeMutationApis* mutation_apis_;
};
}  
#endif  