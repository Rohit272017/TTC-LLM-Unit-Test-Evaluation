#include "xla/service/convert_operand_folding.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class ConvertOperandFoldingTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<HloModule> ParseModuleOrDie(const std::string& hlo) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    CHECK_OK(module.status());
    return std::move(module).value();
  }

  HloInstruction* RootInstruction(HloModule* module) {
    return module->entry_computation()->root_instruction();
  }
};

TEST_F(ConvertOperandFoldingTest, DotWithUpcastConvertOperandMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[2,3] convert(p0)
  ROOT dot = f32[2,4] dot(convert, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, DotWithUpcastConvertOnRhsMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = s8[3,4] parameter(1)
  convert = f32[3,4] convert(p1)
  ROOT dot = f32[2,4] dot(p0, convert),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, DotWithoutConvertDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  ROOT dot = f32[2,4] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, NonDotAndNonConvolutionDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[2,3] parameter(0)
  ROOT convert = f32[2,3] convert(p0)
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, DowncastConvertDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = s8[3,4] parameter(1)
  convert = s8[2,3] convert(p0)
  ROOT dot = s8[2,4] dot(convert, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, BitcastConvertDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s32[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[2,3] convert(p0)
  ROOT dot = f32[2,4] dot(convert, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, UpcastConvertThroughReshapeMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[6] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[6] convert(p0)
  reshape = f32[2,3] reshape(convert)
  ROOT dot = f32[2,4] dot(reshape, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, UpcastConvertThroughTransposeMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[3,2] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[3,2] convert(p0)
  transpose = f32[2,3] transpose(convert), dimensions={1,0}
  ROOT dot = f32[2,4] dot(transpose, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, UpcastConvertThroughSliceMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[4,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[4,3] convert(p0)
  slice = f32[2,3] slice(convert), slice={[0:2], [0:3]}
  ROOT dot = f32[2,4] dot(slice, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, UpcastConvertThroughDynamicSliceMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[4,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  i0 = s32[] constant(0)
  i1 = s32[] constant(0)
  convert = f32[4,3] convert(p0)
  dynamic-slice = f32[2,3] dynamic-slice(convert, i0, i1),
      dynamic_slice_sizes={2,3}
  ROOT dot = f32[2,4] dot(dynamic-slice, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, UpcastConvertThroughBroadcastMatchesPatternIsFalse) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[3] convert(p0)
  broadcast = f32[2,3] broadcast(convert), dimensions={1}
  ROOT dot = f32[2,4] dot(broadcast, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ReduceWithoutElementCountChangeMatchesPattern) {
  const std::string hlo = R"(
HloModule test

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = s8[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  zero = f32[] constant(0)
  convert = f32[2,3] convert(p0)
  reduce = f32[2,3] reduce(convert, zero), dimensions={}, to_apply=add
  ROOT dot = f32[2,4] dot(reduce, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ReduceWithElementCountChangeDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = s8[2,3] parameter(0)
  p1 = f32[4,5] parameter(1)
  zero = f32[] constant(0)
  convert = f32[2,3] convert(p0)
  reduce = f32[2] reduce(convert, zero), dimensions={1}, to_apply=add
  broadcast = f32[4,2] broadcast(reduce), dimensions={1}
  ROOT dot = f32[4,5] dot(broadcast, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ScalarConvertOperandDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[] parameter(0)
  p1 = f32[1,1] parameter(1)
  convert = f32[] convert(p0)
  broadcast = f32[1,1] broadcast(convert), dimensions={}
  ROOT dot = f32[1,1] dot(broadcast, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ConvolutionWithUpcastConvertOperandMatchesPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  input = s8[1,4,4,1] parameter(0)
  filter = f32[2,2,1,1] parameter(1)
  convert = f32[1,4,4,1] convert(input)
  ROOT conv = f32[1,3,3,1] convolution(convert, filter),
      window={size=2x2},
      dim_labels=b01f_01io->b01f
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_TRUE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ConvolutionWithoutUpcastConvertDoesNotMatchPattern) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  input = f32[1,4,4,1] parameter(0)
  filter = f32[2,2,1,1] parameter(1)
  ROOT conv = f32[1,3,3,1] convolution(input, filter),
      window={size=2x2},
      dim_labels=b01f_01io->b01f
}
)";

  auto module = ParseModuleOrDie(hlo);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(RootInstruction(module.get())));
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsDirectConvertOperand) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[2,3] convert(p0)
  ROOT dot = f32[2,4] dot(convert, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(dot->operand(1)->shape().element_type(), F32);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsConvertThroughReshape) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[6] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[6] convert(p0)
  reshape = f32[2,3] reshape(convert)
  ROOT dot = f32[2,4] dot(reshape, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(dot->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(0)->operand(0)->shape().element_type(), S8);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsConvertThroughTranspose) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[3,2] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[3,2] convert(p0)
  transpose = f32[2,3] transpose(convert), dimensions={1,0}
  ROOT dot = f32[2,4] dot(transpose, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(dot->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsConvertThroughSlice) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[4,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  convert = f32[4,3] convert(p0)
  slice = f32[2,3] slice(convert), slice={[0:2], [0:3]}
  ROOT dot = f32[2,4] dot(slice, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kSlice);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(dot->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsBothDotOperands) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = s8[2,3] parameter(0)
  p1 = s8[3,4] parameter(1)
  convert0 = f32[2,3] convert(p0)
  convert1 = f32[3,4] convert(p1)
  ROOT dot = f32[2,4] dot(convert0, convert1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(dot->operand(1)->shape().element_type(), S8);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionLeavesNonMatchingDotUnchanged) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[3,4] parameter(1)
  ROOT dot = f32[2,4] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* dot = RootInstruction(module.get());
  HloInstruction* original_lhs = dot->mutable_operand(0);
  HloInstruction* original_rhs = dot->mutable_operand(1);
  ConvertOperandFolding pass;

  EXPECT_FALSE(pass.InstructionMatchesPattern(dot));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(dot));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(dot->operand(0), original_lhs);
  EXPECT_EQ(dot->operand(1), original_rhs);
}

TEST_F(ConvertOperandFoldingTest, ExpandInstructionFoldsConvolutionOperand) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  input = s8[1,4,4,1] parameter(0)
  filter = f32[2,2,1,1] parameter(1)
  convert = f32[1,4,4,1] convert(input)
  ROOT conv = f32[1,3,3,1] convolution(convert, filter),
      window={size=2x2},
      dim_labels=b01f_01io->b01f
}
)";

  auto module = ParseModuleOrDie(hlo);
  HloInstruction* conv = RootInstruction(module.get());
  ConvertOperandFolding pass;

  ASSERT_TRUE(pass.InstructionMatchesPattern(conv));
  ASSERT_OK_AND_ASSIGN(HloInstruction * replacement, pass.ExpandInstruction(conv));

  EXPECT_EQ(replacement, nullptr);
  EXPECT_EQ(conv->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(conv->operand(0)->shape().element_type(), S8);
  EXPECT_EQ(conv->operand(1)->shape().element_type(), F32);
}

}  // namespace
}  // namespace xla