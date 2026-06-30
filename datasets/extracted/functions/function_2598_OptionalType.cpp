#include <cstddef>
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "common/type.h"
namespace cel {
namespace common_internal {
namespace {
struct OptionalTypeData final {
  const absl::string_view name;
  const size_t parameters_size;
  const Type parameter;
};
union DynOptionalTypeData final {
  OptionalTypeData optional;
  OpaqueTypeData opaque;
};
static_assert(offsetof(OptionalTypeData, name) ==
              offsetof(OpaqueTypeData, name));
static_assert(offsetof(OptionalTypeData, parameters_size) ==
              offsetof(OpaqueTypeData, parameters_size));
static_assert(offsetof(OptionalTypeData, parameter) ==
              offsetof(OpaqueTypeData, parameters));
ABSL_CONST_INIT const DynOptionalTypeData kDynOptionalTypeData = {
    .optional =
        {
            .name = OptionalType::kName,
            .parameters_size = 1,
            .parameter = DynType(),
        },
};
}  
}  
OptionalType::OptionalType()
    : opaque_(&common_internal::kDynOptionalTypeData.opaque) {}
Type OptionalType::GetParameter() const { return GetParameters().front(); }
}  