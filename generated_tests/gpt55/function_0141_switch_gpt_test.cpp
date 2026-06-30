#include "xla/service/gpu/gpu_float_support.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

class GpuFloatSupportTest : public HloHardwareIndependentTestBase {
 protected:
  Shape ScalarShape(PrimitiveType type) { return ShapeUtil::MakeShape(type, {}); }

  Shape MatrixShape(PrimitiveType type) {
    return ShapeUtil::MakeShape(type, {2, 2});
  }

  HloInstruction* AddParameter(HloComputation::Builder* builder,
                               int64_t parameter_number,
                               PrimitiveType type) {
    return builder->AddInstruction(HloInstruction::CreateParameter(
        parameter_number, MatrixShape(type),
        absl::StrCat("p", parameter_number)));
  }

  HloInstruction* AddBinaryInstruction(HloComputation::Builder* builder,
                                       HloOpcode opcode,
                                       PrimitiveType type) {
    HloInstruction* lhs = AddParameter(builder, 0, type);
    HloInstruction* rhs = AddParameter(builder, 1, type);
    return builder->AddInstruction(
        HloInstruction::CreateBinary(MatrixShape(type), opcode, lhs, rhs));
  }

  HloInstruction* AddUnaryShapeInstruction(HloComputation::Builder* builder,
                                           HloOpcode opcode,
                                           PrimitiveType type) {
    HloInstruction* operand = AddParameter(builder, 0, type);
    switch (opcode) {
      case HloOpcode::kCopy:
        return builder->AddInstruction(
            HloInstruction::CreateUnary(MatrixShape(type), opcode, operand));
      case HloOpcode::kReshape:
        return builder->AddInstruction(
            HloInstruction::CreateReshape(MatrixShape(type), operand));
      case HloOpcode::kBitcast:
        return builder->AddInstruction(
            HloInstruction::CreateBitcast(MatrixShape(type), operand));
      case HloOpcode::kBroadcast:
        return builder->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShape(type, {2, 2, 1}), operand, {0, 1}));
      case HloOpcode::kTranspose:
        return builder->AddInstruction(
            HloInstruction::CreateTranspose(MatrixShape(type), operand, {1, 0}));
      case HloOpcode::kReverse:
        return builder->AddInstruction(
            HloInstruction::CreateReverse(MatrixShape(type), operand, {0}));
      default:
        return nullptr;
    }
  }

  HloInstruction* AddDotInstruction(HloComputation::Builder* builder,
                                    PrimitiveType lhs_type,
                                    PrimitiveType rhs_type,
                                    PrimitiveType result_type) {
    HloInstruction* lhs = AddParameter(builder, 0, lhs_type);
    HloInstruction* rhs = AddParameter(builder, 1, rhs_type);

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);

    PrecisionConfig precision_config;
    return builder->AddInstruction(HloInstruction::CreateDot(
        MatrixShape(result_type), lhs, rhs, dot_dnums, precision_config));
  }

  std::unique_ptr<HloModule> CreateModuleWithRoot(HloInstruction* root,
                                                  HloComputation::Builder* b) {
    auto module = CreateNewVerifiedModule();
    HloComputation* computation = module->AddEntryComputation(b->Build(root));
    CHECK_EQ(computation->root_instruction(), root);
    return module;
  }

  se::CudaComputeCapability Ampere() {
    return se::CudaComputeCapability{
        se::CudaComputeCapability::CudaComputeCapabilities::AMPERE};
  }

  se::CudaComputeCapability Hopper() {
    return se::CudaComputeCapability{
        se::CudaComputeCapability::CudaComputeCapabilities::HOPPER};
  }

  se::CudaComputeCapability Volta() {
    return se::CudaComputeCapability{
        se::CudaComputeCapability::CudaComputeCapabilities::VOLTA};
  }
};

TEST_F(GpuFloatSupportTest, SupportsMixedPrecisionsForF16DotProducingF32) {
  HloComputation::Builder builder("dot_f16_f32");
  HloInstruction* dot = AddDotInstruction(&builder, F16, F16, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_TRUE(support.SupportsMixedPrecisions(*dot));
}

TEST_F(GpuFloatSupportTest, SupportsMixedPrecisionsForBf16DotProducingF32) {
  HloComputation::Builder builder("dot_bf16_f32");
  HloInstruction* dot = AddDotInstruction(&builder, BF16, BF16, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_TRUE(support.SupportsMixedPrecisions(*dot));
}

TEST_F(GpuFloatSupportTest, DoesNotSupportMixedPrecisionsForF16DotProducingF16) {
  HloComputation::Builder builder("dot_f16_f16");
  HloInstruction* dot = AddDotInstruction(&builder, F16, F16, F16);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_FALSE(support.SupportsMixedPrecisions(*dot));
}

TEST_F(GpuFloatSupportTest, DoesNotSupportMixedPrecisionsForBf16DotProducingBf16) {
  HloComputation::Builder builder("dot_bf16_bf16");
  HloInstruction* dot = AddDotInstruction(&builder, BF16, BF16, BF16);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_FALSE(support.SupportsMixedPrecisions(*dot));
}

TEST_F(GpuFloatSupportTest, DoesNotSupportMixedPrecisionsForMixedOperandDot) {
  HloComputation::Builder builder("dot_mixed_operands");
  HloInstruction* dot = AddDotInstruction(&builder, F16, BF16, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_FALSE(support.SupportsMixedPrecisions(*dot));
}

TEST_F(GpuFloatSupportTest, DoesNotSupportMixedPrecisionsForNonDotOpcode) {
  HloComputation::Builder builder("add");
  HloInstruction* add = AddBinaryInstruction(&builder, HloOpcode::kAdd, F16);
  auto module = CreateModuleWithRoot(add, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_FALSE(support.SupportsMixedPrecisions(*add));
}

TEST_F(GpuFloatSupportTest, Bf16SupportsDot) {
  HloComputation::Builder builder("bf16_dot");
  HloInstruction* dot = AddDotInstruction(&builder, BF16, BF16, BF16);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_TRUE(support.IsSupported(*dot));
}

TEST_F(GpuFloatSupportTest, Bf16SupportsAllReduceFamilyAndReduceScatter) {
  HloComputation::Builder builder("bf16_collectives");
  HloInstruction* p0 = AddParameter(&builder, 0, BF16);

  auto* all_reduce = builder.AddInstruction(HloInstruction::CreateUnary(
      MatrixShape(BF16), HloOpcode::kAllReduce, p0));
  auto module = CreateModuleWithRoot(all_reduce, &builder);

  GpuFloatSupport support(BF16);

  EXPECT_TRUE(support.IsSupported(*all_reduce));
}

TEST_F(GpuFloatSupportTest, F16DoesNotSupportDotThroughGpuFloatSupportRule) {
  HloComputation::Builder builder("f16_dot");
  HloInstruction* dot = AddDotInstruction(&builder, F16, F16, F16);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(F16);

  EXPECT_FALSE(support.IsSupported(*dot));
}

TEST_F(GpuFloatSupportTest, ShapeChangingOpcodesAreSupportedForBf16) {
  const HloOpcode supported_opcodes[] = {
      HloOpcode::kCopy,      HloOpcode::kReshape, HloOpcode::kReverse,
      HloOpcode::kTranspose, HloOpcode::kBitcast,
  };

  for (HloOpcode opcode : supported_opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddUnaryShapeInstruction(&builder, opcode, BF16);
    ASSERT_NE(root, nullptr);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(BF16);

    EXPECT_TRUE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, ShapeChangingOpcodesAreSupportedForF16) {
  const HloOpcode supported_opcodes[] = {
      HloOpcode::kCopy,      HloOpcode::kReshape, HloOpcode::kReverse,
      HloOpcode::kTranspose, HloOpcode::kBitcast,
  };

  for (HloOpcode opcode : supported_opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddUnaryShapeInstruction(&builder, opcode, F16);
    ASSERT_NE(root, nullptr);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(F16);

    EXPECT_TRUE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, Bf16AddSubtractMultiplySupportedOnHopper) {
  const HloOpcode opcodes[] = {
      HloOpcode::kAdd,
      HloOpcode::kSubtract,
      HloOpcode::kMultiply,
  };

  for (HloOpcode opcode : opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddBinaryInstruction(&builder, opcode, BF16);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(BF16, Hopper());

    EXPECT_TRUE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, Bf16AddSubtractMultiplyNotSupportedBelowHopper) {
  const HloOpcode opcodes[] = {
      HloOpcode::kAdd,
      HloOpcode::kSubtract,
      HloOpcode::kMultiply,
  };

  for (HloOpcode opcode : opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddBinaryInstruction(&builder, opcode, BF16);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(BF16, Ampere());

    EXPECT_FALSE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, Bf16AddSubtractMultiplyNotSupportedWithoutCudaCapability) {
  const HloOpcode opcodes[] = {
      HloOpcode::kAdd,
      HloOpcode::kSubtract,
      HloOpcode::kMultiply,
  };

  for (HloOpcode opcode : opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddBinaryInstruction(&builder, opcode, BF16);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(BF16);

    EXPECT_FALSE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, F16AddSubtractMultiplyAreNotSupported) {
  const HloOpcode opcodes[] = {
      HloOpcode::kAdd,
      HloOpcode::kSubtract,
      HloOpcode::kMultiply,
  };

  for (HloOpcode opcode : opcodes) {
    HloComputation::Builder builder(HloOpcodeString(opcode));
    HloInstruction* root = AddBinaryInstruction(&builder, opcode, F16);
    auto module = CreateModuleWithRoot(root, &builder);

    GpuFloatSupport support(F16, Hopper());

    EXPECT_FALSE(support.IsSupported(*root)) << HloOpcodeString(opcode);
  }
}

TEST_F(GpuFloatSupportTest, UnsupportedOpcodeReturnsFalse) {
  HloComputation::Builder builder("unsupported_exp");
  HloInstruction* operand = AddParameter(&builder, 0, BF16);
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(MatrixShape(BF16), HloOpcode::kExp, operand));
  auto module = CreateModuleWithRoot(exp, &builder);

  GpuFloatSupport support(BF16, Hopper());

  EXPECT_FALSE(support.IsSupported(*exp));
}

TEST_F(GpuFloatSupportTest, ConstantOpcodeReturnsFalse) {
  HloComputation::Builder builder("constant");
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto module = CreateModuleWithRoot(constant, &builder);

  GpuFloatSupport support(BF16, Hopper());

  EXPECT_FALSE(support.IsSupported(*constant));
}

TEST_F(GpuFloatSupportTest, F8E4M3FnDotWithoutCudaCapabilityReturnsFalse) {
  HloComputation::Builder builder("f8_dot_no_cuda");
  HloInstruction* dot = AddDotInstruction(&builder, F8E4M3FN, F8E4M3FN, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(F8E4M3FN);

  EXPECT_FALSE(support.IsSupported(*dot));
}

TEST_F(GpuFloatSupportTest, F8E4M3FnDotBelowAmpereReturnsFalse) {
  HloComputation::Builder builder("f8_dot_volta");
  HloInstruction* dot = AddDotInstruction(&builder, F8E4M3FN, F8E4M3FN, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(F8E4M3FN, Volta());

  EXPECT_FALSE(support.IsSupported(*dot));
}

TEST_F(GpuFloatSupportTest, F8E5M2DotBelowHopperReturnsFalse) {
  HloComputation::Builder builder("f8e5m2_dot_ampere");
  HloInstruction* dot = AddDotInstruction(&builder, F8E5M2, F8E5M2, F32);
  auto module = CreateModuleWithRoot(dot, &builder);

  GpuFloatSupport support(F8E5M2, Ampere());

  EXPECT_FALSE(support.IsSupported(*dot));
}

TEST_F(GpuFloatSupportTest, F8E5M2AddReturnsFalseEvenOnHopper) {
  HloComputation::Builder builder("f8_add");
  HloInstruction* add = AddBinaryInstruction(&builder, HloOpcode::kAdd, F8E5M2);
  auto module = CreateModuleWithRoot(add, &builder);

  GpuFloatSupport support(F8E5M2, Hopper());

  EXPECT_FALSE(support.IsSupported(*add));
}

}  // namespace
}  // namespace gpu
}  // namespace xla