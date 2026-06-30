#include "xla/service/cpu_gpu_shape_verifier.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

class CpuGpuShapeVerifierTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<HloModule> ParseModuleOrDie(const std::string& hlo) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    CHECK_OK(module.status());
    return std::move(module).value();
  }

  HloInstruction* RootInstruction(HloModule* module) {
    return module->entry_computation()->root_instruction();
  }

  absl::Status VerifyInstruction(HloInstruction* instruction) {
    CpuGpuShapeVerifier verifier(/*layout_sensitive=*/true,
                                 /*allow_mixed_precision=*/true);
    return verifier.Preprocess(instruction);
  }
};

TEST_F(CpuGpuShapeVerifierTest, F32ParameterWithDefaultLayoutIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = f32[2,3]{1,0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S32ParameterWithDefaultLayoutIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = s32[2,3]{1,0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, PredicateParameterWithDefaultLayoutIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = pred[2,3]{1,0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, NonSubByteTypeWithZeroCustomElementSizeIsAccepted) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  shape.mutable_layout()->set_element_size_in_bits(0);
  auto instruction = HloInstruction::CreateParameter(0, shape, "p0");

  EXPECT_THAT(VerifyInstruction(instruction.get()), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, NonSubByteTypeWithCustomElementSizeOneBitIsRejected) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  shape.mutable_layout()->set_element_size_in_bits(1);
  auto instruction = HloInstruction::CreateParameter(0, shape, "p0");

  EXPECT_THAT(
      VerifyInstruction(instruction.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "does not support custom element sizes on non-sub-byte-bit "
                   "types")));
}

TEST_F(CpuGpuShapeVerifierTest, NonSubByteTypeWithCustomElementSizeEightBitsIsRejected) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  shape.mutable_layout()->set_element_size_in_bits(8);
  auto instruction = HloInstruction::CreateParameter(0, shape, "p0");

  EXPECT_THAT(
      VerifyInstruction(instruction.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "does not support custom element sizes on non-sub-byte-bit "
                   "types")));
}

TEST_F(CpuGpuShapeVerifierTest, NonSubByteTypeWithoutLayoutIsAccepted) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  shape.clear_layout();
  auto instruction = HloInstruction::CreateParameter(0, shape, "p0");

  EXPECT_THAT(VerifyInstruction(instruction.get()), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, TupleWithValidSubshapesIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = s32[4]{0} parameter(1)
  ROOT tuple = (f32[2,3]{1,0}, s32[4]{0}) tuple(p0, p1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, TupleWithInvalidCustomElementSizeSubshapeIsRejected) {
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 3});
  *f32_shape.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  f32_shape.mutable_layout()->set_element_size_in_bits(4);

  Shape s32_shape = ShapeUtil::MakeShape(S32, {4});
  *s32_shape.mutable_layout() = LayoutUtil::MakeLayout({0});

  Shape tuple_shape = ShapeUtil::MakeTupleShape({f32_shape, s32_shape});
  auto instruction = HloInstruction::CreateParameter(0, tuple_shape, "tuple_p0");

  EXPECT_THAT(
      VerifyInstruction(instruction.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "does not support custom element sizes on non-sub-byte-bit "
                   "types")));
}

TEST_F(CpuGpuShapeVerifierTest, S4ParameterIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = s4[2,3]{1,0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, U4ParameterIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = u4[2,3]{1,0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4ConstantIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT c = s4[] constant(1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4ConvertIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[2,3]{1,0} parameter(0)
  ROOT convert = s4[2,3]{1,0} convert(p0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4CopyIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT copy = s4[2,3]{1,0} copy(p0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4BitcastIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[6]{0} parameter(0)
  ROOT bitcast = s4[2,3]{1,0} bitcast(p0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4BroadcastIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[3]{0} parameter(0)
  ROOT broadcast = s4[2,3]{1,0} broadcast(p0), dimensions={1}
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4ConcatenateIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  p1 = s4[2,3]{1,0} parameter(1)
  ROOT concat = s4[4,3]{1,0} concatenate(p0, p1), dimensions={0}
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4SliceIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[4,3]{1,0} parameter(0)
  ROOT slice = s4[2,3]{1,0} slice(p0), slice={[0:2], [0:3]}
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4DynamicSliceIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[4,3]{1,0} parameter(0)
  c0 = s32[] constant(0)
  c1 = s32[] constant(0)
  ROOT ds = s4[2,3]{1,0} dynamic-slice(p0, c0, c1),
      dynamic_slice_sizes={2,3}
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4DynamicUpdateSliceIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[4,3]{1,0} parameter(0)
  update = s4[2,3]{1,0} parameter(1)
  c0 = s32[] constant(0)
  c1 = s32[] constant(0)
  ROOT dus = s4[4,3]{1,0} dynamic-update-slice(p0, update, c0, c1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4GetTupleElementIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  tuple = (s4[2,3]{1,0}) tuple(p0)
  ROOT gte = s4[2,3]{1,0} get-tuple-element(tuple), index=0
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4TupleIsAllowListedAndAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT tuple = (s4[2,3]{1,0}) tuple(p0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, S4AddIsRejectedBecauseOpcodeIsNotAllowListed) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  p1 = s4[2,3]{1,0} parameter(1)
  ROOT add = s4[2,3]{1,0} add(p0, p1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "s4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, U4MultiplyIsRejectedBecauseOpcodeIsNotAllowListed) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = u4[2,3]{1,0} parameter(0)
  p1 = u4[2,3]{1,0} parameter(1)
  ROOT multiply = u4[2,3]{1,0} multiply(p0, p1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "u4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, NonAllowListedOpcodeWithF32ShapeIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  ROOT add = f32[2,3]{1,0} add(p0, p1)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, CustomCallShardingWithS4IsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT cc = s4[2,3]{1,0} custom-call(p0), custom_call_target="Sharding"
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, CustomCallNonAllowListedTargetWithS4IsRejected) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT cc = s4[2,3]{1,0} custom-call(p0), custom_call_target="NotSharding"
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "s4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, CustomCallEmptyTargetWithS4IsRejected) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT cc = s4[2,3]{1,0} custom-call(p0), custom_call_target=""
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "s4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, CustomCallTargetIsCaseSensitive) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  ROOT cc = s4[2,3]{1,0} custom-call(p0), custom_call_target="sharding"
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "s4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, TupleRootWithS4SubshapeOnNonAllowListedOpcodeIsRejected) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s4[2,3]{1,0} parameter(0)
  p1 = s4[2,3]{1,0} parameter(1)
  add = s4[2,3]{1,0} add(p0, p1)
  ROOT tuple = (s4[2,3]{1,0}) tuple(add)
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* add = RootInstruction(module.get())->mutable_operand(0);

  EXPECT_THAT(VerifyInstruction(add),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "s4 is currently only supported in allow-listed "
                           "instructions")));
}

TEST_F(CpuGpuShapeVerifierTest, BoundaryRankZeroS4ParameterIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = s4[] parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, BoundaryRankOneS4ParameterIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = s4[1]{0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, BoundaryEmptyArrayShapeIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = f32[0]{0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

TEST_F(CpuGpuShapeVerifierTest, BoundaryEmptyS4ArrayOnAllowListedOpcodeIsAccepted) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  ROOT p0 = s4[0]{0} parameter(0)
}
)";

  auto module = ParseModuleOrDie(hlo);

  EXPECT_THAT(VerifyInstruction(RootInstruction(module.get())), IsOk());
}

}  // namespace
}  // namespace xla