#include "xla/service/cpu_gpu_shape_verifier.h"
#include <array>
#include <string_view>
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
bool IsAllowedS4U4CustomCall(const HloInstruction* instruction) {
  static constexpr std::array<std::string_view, 1> kMetadataCustomCalls = {
      "Sharding",
  };
  return absl::c_any_of(kMetadataCustomCalls, [&](std::string_view target) {
    return target == instruction->custom_call_target();
  });
}
absl::Status VerifyS4U4Usage(HloInstruction* instruction) {
  auto verify_subshape = [](const HloInstruction* instruction) {
    return ShapeUtil::ForEachSubshapeWithStatus(
        instruction->shape(), [&](const Shape& shape, const ShapeIndex&) {
          if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "%s is currently only supported in allow-listed instructions, "
                "but got instruction: %s",
                primitive_util::LowercasePrimitiveTypeName(
                    shape.element_type()),
                instruction->ToString()));
          }
          return absl::OkStatus();
        });
  };
  switch (instruction->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFusion:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kSlice:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
      break;
    case HloOpcode::kCustomCall:
      if (IsAllowedS4U4CustomCall(instruction)) {
        break;
      }
      ABSL_FALLTHROUGH_INTENDED;
    default:
      return verify_subshape(instruction);
  }
  return absl::OkStatus();
}
}  
absl::Status CpuGpuShapeVerifier::Preprocess(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      hlo->shape(), [&](const Shape& shape, const ShapeIndex&) {
        if (shape.has_layout()) {
          if (LayoutUtil::IsSparseArray(shape)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support sparse shapes: %s",
                hlo->ToString()));
          }
          if (!primitive_util::IsSubByteNonPredType(shape.element_type()) &&
              shape.layout().element_size_in_bits() != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "The XLA CPU/GPU backend does not support custom element sizes "
                "on non-sub-byte-bit types: %s",
                hlo->ToString()));
          }
        }
        return absl::OkStatus();
      }));
  TF_RETURN_IF_ERROR(VerifyS4U4Usage(hlo));
  return ShapeVerifier::Preprocess(hlo);
}
}  