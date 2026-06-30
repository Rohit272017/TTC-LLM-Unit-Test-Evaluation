#include "xla/service/result_caster.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ResultCasterTest : public HloHardwareIndependentTestBase {
 protected:
  HloInstruction* GetInstruction(HloModule* module, const std::string& name) {
    return module->entry_computation()->GetInstructionWithName(name);
  }
};

TEST_F(ResultCasterTest, NonDotAndNonConvolutionInstructionDoesNotMatch) {
  const char* hlo = R"(
HloModule add_module

ENTRY main {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  ROOT add = f32[2,2] add(p0, p1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* add = GetInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(add));
}

TEST_F(ResultCasterTest, DotWithSameInferredAndActualElementTypeDoesNotMatch) {
  const char* hlo = R"(
HloModule dot_same_type

ENTRY main {
  lhs = f32[2,3] parameter(0)
  rhs = f32[3,4] parameter(1)
  ROOT dot = f32[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, DotWithLowerPrecisionResultThanInferredShapeMatches) {
  const char* hlo = R"(
HloModule dot_lower_precision_result

ENTRY main {
  lhs = f16[2,3] parameter(0)
  rhs = f16[3,4] parameter(1)
  ROOT dot = f16[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_TRUE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, DotWithHigherPrecisionActualResultDoesNotMatch) {
  const char* hlo = R"(
HloModule dot_higher_precision_actual

ENTRY main {
  lhs = f16[2,3] parameter(0)
  rhs = f16[3,4] parameter(1)
  ROOT dot = f32[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, InvalidDotShapeDoesNotMatch) {
  const char* hlo = R"(
HloModule invalid_dot_shape

ENTRY main {
  lhs = f32[2,3] parameter(0)
  rhs = f32[5,4] parameter(1)
  ROOT dot = f32[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, ConvolutionWithSameInferredAndActualElementTypeDoesNotMatch) {
  const char* hlo = R"(
HloModule convolution_same_type

ENTRY main {
  input = f32[1,8,8,3] parameter(0)
  filter = f32[3,3,3,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(input, filter),
      window={size=3x3},
      dim_labels=b01f_01io->b01f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* conv = GetInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(conv));
}

TEST_F(ResultCasterTest, ConvolutionWithLowerPrecisionResultThanInferredShapeMatches) {
  const char* hlo = R"(
HloModule convolution_lower_precision_result

ENTRY main {
  input = f16[1,8,8,3] parameter(0)
  filter = f16[3,3,3,16] parameter(1)
  ROOT conv = f16[1,6,6,16] convolution(input, filter),
      window={size=3x3},
      dim_labels=b01f_01io->b01f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* conv = GetInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);

  ResultCaster caster;

  EXPECT_TRUE(caster.InstructionMatchesPattern(conv));
}

TEST_F(ResultCasterTest, InvalidConvolutionShapeDoesNotMatch) {
  const char* hlo = R"(
HloModule invalid_convolution_shape

ENTRY main {
  input = f32[1,8,8,3] parameter(0)
  filter = f32[3,3,4,16] parameter(1)
  ROOT conv = f32[1,6,6,16] convolution(input, filter),
      window={size=3x3},
      dim_labels=b01f_01io->b01f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));

  HloInstruction* conv = GetInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(conv));
}

TEST_F(ResultCasterTest, ExpandDotCreatesConvertWithOriginalShape) {
  const char* hlo = R"(
HloModule expand_dot

ENTRY main {
  lhs = f16[2,3] parameter(0)
  rhs = f16[3,4] parameter(1)
  ROOT dot = f16[2,4]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;
  ASSERT_TRUE(caster.InstructionMatchesPattern(dot));

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction* expanded,
                          caster.ExpandInstruction(dot));

  ASSERT_NE(expanded, nullptr);
  EXPECT_EQ(expanded->opcode(), HloOpcode::kConvert);
  EXPECT_TRUE(ShapeUtil::Equal(expanded->shape(), dot->shape()));
  ASSERT_EQ(expanded->operand_count(), 1);
  EXPECT_EQ(expanded->operand(0)->opcode(), HloOpcode::kDot);
  EXPECT_EQ(expanded->operand(0)->shape().element_type(), F32);
  EXPECT_TRUE(ShapeUtil::Equal(expanded->operand(0)->shape().layout(),
                               dot->shape().layout()));
}

TEST_F(ResultCasterTest, ExpandConvolutionCreatesConvertWithOriginalShape) {
  const char* hlo = R"(
HloModule expand_convolution

ENTRY main {
  input = f16[1,8,8,3]{3,2,1,0} parameter(0)
  filter = f16[3,3,3,16]{3,2,1,0} parameter(1)
  ROOT conv = f16[1,6,6,16]{3,2,1,0} convolution(input, filter),
      window={size=3x3},
      dim_labels=b01f_01io->b01f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* conv = GetInstruction(module.get(), "conv");
  ASSERT_NE(conv, nullptr);

  ResultCaster caster;
  ASSERT_TRUE(caster.InstructionMatchesPattern(conv));

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction* expanded,
                          caster.ExpandInstruction(conv));

  ASSERT_NE(expanded, nullptr);
  EXPECT_EQ(expanded->opcode(), HloOpcode::kConvert);
  EXPECT_TRUE(ShapeUtil::Equal(expanded->shape(), conv->shape()));
  ASSERT_EQ(expanded->operand_count(), 1);
  EXPECT_EQ(expanded->operand(0)->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(expanded->operand(0)->shape().element_type(), F32);
  EXPECT_TRUE(ShapeUtil::Equal(expanded->operand(0)->shape().layout(),
                               conv->shape().layout()));
}

TEST_F(ResultCasterTest, DotBoundaryRankOneOperandsMatchWhenLowerPrecision) {
  const char* hlo = R"(
HloModule dot_rank_one

ENTRY main {
  lhs = f16[3] parameter(0)
  rhs = f16[3] parameter(1)
  ROOT dot = f16[] dot(lhs, rhs),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_TRUE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, DotWithBf16LowerPrecisionResultMatches) {
  const char* hlo = R"(
HloModule dot_bf16

ENTRY main {
  lhs = bf16[2,3] parameter(0)
  rhs = bf16[3,4] parameter(1)
  ROOT dot = bf16[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_TRUE(caster.InstructionMatchesPattern(dot));
}

TEST_F(ResultCasterTest, DotWithMixedOperandTypesDoesNotMatchWhenActualIsHighestPrecision) {
  const char* hlo = R"(
HloModule dot_mixed_operands

ENTRY main {
  lhs = f16[2,3] parameter(0)
  rhs = f32[3,4] parameter(1)
  ROOT dot = f32[2,4] dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* dot = GetInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);

  ResultCaster caster;

  EXPECT_FALSE(caster.InstructionMatchesPattern(dot));
}

}  // namespace
}  // namespace xla