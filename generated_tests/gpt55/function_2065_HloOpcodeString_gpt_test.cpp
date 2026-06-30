#include "xla/hlo/ir/hlo_opcode.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace xla {
namespace {

TEST(HloOpcodeTest, ECP_HloOpcodeStringReturnsExpectedKnownOpcodeNames) {
  EXPECT_EQ(HloOpcodeString(HloOpcode::kAdd), "add");
  EXPECT_EQ(HloOpcodeString(HloOpcode::kMultiply), "multiply");
  EXPECT_EQ(HloOpcodeString(HloOpcode::kCompare), "compare");
  EXPECT_EQ(HloOpcodeString(HloOpcode::kParameter), "parameter");
  EXPECT_EQ(HloOpcodeString(HloOpcode::kTuple), "tuple");
}

TEST(HloOpcodeTest, ECP_StringToHloOpcodeReturnsExpectedKnownOpcodes) {
  EXPECT_EQ(StringToHloOpcode("add").value(), HloOpcode::kAdd);
  EXPECT_EQ(StringToHloOpcode("multiply").value(), HloOpcode::kMultiply);
  EXPECT_EQ(StringToHloOpcode("compare").value(), HloOpcode::kCompare);
  EXPECT_EQ(StringToHloOpcode("parameter").value(), HloOpcode::kParameter);
  EXPECT_EQ(StringToHloOpcode("tuple").value(), HloOpcode::kTuple);
}

TEST(HloOpcodeTest, ECP_RoundTripOpcodeStringConversionForRepresentativeOpcodes) {
  const std::vector<HloOpcode> opcodes = {
      HloOpcode::kAdd,
      HloOpcode::kSubtract,
      HloOpcode::kMultiply,
      HloOpcode::kDivide,
      HloOpcode::kCompare,
      HloOpcode::kConstant,
      HloOpcode::kParameter,
      HloOpcode::kTuple,
      HloOpcode::kCall,
      HloOpcode::kFusion,
      HloOpcode::kWhile,
      HloOpcode::kAsyncStart,
      HloOpcode::kAsyncUpdate,
      HloOpcode::kAsyncDone,
  };

  for (HloOpcode opcode : opcodes) {
    absl::StatusOr<HloOpcode> parsed =
        StringToHloOpcode(HloOpcodeString(opcode));

    ASSERT_TRUE(parsed.ok()) << HloOpcodeString(opcode);
    EXPECT_EQ(parsed.value(), opcode);
  }
}

TEST(HloOpcodeTest, Invalid_StringToHloOpcodeUnknownNameReturnsError) {
  absl::StatusOr<HloOpcode> result = StringToHloOpcode("unknown_opcode");

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HloOpcodeTest, BVA_StringToHloOpcodeEmptyStringReturnsError) {
  absl::StatusOr<HloOpcode> result = StringToHloOpcode("");

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HloOpcodeTest, Edge_StringToHloOpcodeIsCaseSensitive) {
  EXPECT_FALSE(StringToHloOpcode("ADD").ok());
  EXPECT_FALSE(StringToHloOpcode("Add").ok());
  EXPECT_TRUE(StringToHloOpcode("add").ok());
}

TEST(HloOpcodeTest, Edge_StringToHloOpcodeDoesNotTrimWhitespace) {
  EXPECT_FALSE(StringToHloOpcode(" add").ok());
  EXPECT_FALSE(StringToHloOpcode("add ").ok());
  EXPECT_FALSE(StringToHloOpcode("\tadd").ok());
  EXPECT_TRUE(StringToHloOpcode("add").ok());
}

TEST(HloOpcodeTest, Edge_StringToHloOpcodeRejectsSimilarPrefixOrSuffix) {
  EXPECT_FALSE(StringToHloOpcode("add-extra").ok());
  EXPECT_FALSE(StringToHloOpcode("extra-add").ok());
  EXPECT_FALSE(StringToHloOpcode("addition").ok());
}

TEST(HloOpcodeTest, ECP_HloOpcodeIsComparisonOnlyForCompare) {
  EXPECT_TRUE(HloOpcodeIsComparison(HloOpcode::kCompare));

  EXPECT_FALSE(HloOpcodeIsComparison(HloOpcode::kAdd));
  EXPECT_FALSE(HloOpcodeIsComparison(HloOpcode::kSubtract));
  EXPECT_FALSE(HloOpcodeIsComparison(HloOpcode::kParameter));
  EXPECT_FALSE(HloOpcodeIsComparison(HloOpcode::kTuple));
}

TEST(HloOpcodeTest, ECP_HloOpcodeIsAsyncOnlyForAsyncOpcodes) {
  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncStart));
  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncUpdate));
  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncDone));

  EXPECT_FALSE(HloOpcodeIsAsync(HloOpcode::kAdd));
  EXPECT_FALSE(HloOpcodeIsAsync(HloOpcode::kCall));
  EXPECT_FALSE(HloOpcodeIsAsync(HloOpcode::kCopy));
  EXPECT_FALSE(HloOpcodeIsAsync(HloOpcode::kParameter));
}

TEST(HloOpcodeTest, ECP_NullaryOpcodesHaveArityZero) {
  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kConstant).has_value());
  EXPECT_EQ(HloOpcodeArity(HloOpcode::kConstant).value(), 0);

  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kIota).has_value());
  EXPECT_EQ(HloOpcodeArity(HloOpcode::kIota).value(), 0);
}

TEST(HloOpcodeTest, ECP_UnaryOpcodesHaveArityOne) {
  const std::vector<HloOpcode> unary_opcodes = {
      HloOpcode::kAbs,
      HloOpcode::kCeil,
      HloOpcode::kConvert,
      HloOpcode::kCopy,
      HloOpcode::kExp,
      HloOpcode::kLog,
      HloOpcode::kNegate,
      HloOpcode::kSqrt,
      HloOpcode::kTanh,
  };

  for (HloOpcode opcode : unary_opcodes) {
    std::optional<int> arity = HloOpcodeArity(opcode);

    ASSERT_TRUE(arity.has_value()) << HloOpcodeString(opcode);
    EXPECT_EQ(arity.value(), 1) << HloOpcodeString(opcode);
    EXPECT_FALSE(HloOpcodeIsVariadic(opcode)) << HloOpcodeString(opcode);
  }
}

TEST(HloOpcodeTest, ECP_BinaryOpcodesHaveArityTwo) {
  const std::vector<HloOpcode> binary_opcodes = {
      HloOpcode::kAdd,
      HloOpcode::kAtan2,
      HloOpcode::kCompare,
      HloOpcode::kDivide,
      HloOpcode::kMaximum,
      HloOpcode::kMinimum,
      HloOpcode::kMultiply,
      HloOpcode::kPower,
      HloOpcode::kRemainder,
      HloOpcode::kSubtract,
  };

  for (HloOpcode opcode : binary_opcodes) {
    std::optional<int> arity = HloOpcodeArity(opcode);

    ASSERT_TRUE(arity.has_value()) << HloOpcodeString(opcode);
    EXPECT_EQ(arity.value(), 2) << HloOpcodeString(opcode);
    EXPECT_FALSE(HloOpcodeIsVariadic(opcode)) << HloOpcodeString(opcode);
  }
}

TEST(HloOpcodeTest, ECP_VariadicOpcodesReturnNulloptArity) {
  const std::vector<HloOpcode> variadic_opcodes = {
      HloOpcode::kTuple,
      HloOpcode::kConcatenate,
      HloOpcode::kCall,
      HloOpcode::kCustomCall,
  };

  for (HloOpcode opcode : variadic_opcodes) {
    EXPECT_TRUE(HloOpcodeIsVariadic(opcode)) << HloOpcodeString(opcode);
    EXPECT_FALSE(HloOpcodeArity(opcode).has_value()) << HloOpcodeString(opcode);
  }
}

TEST(HloOpcodeTest, ECP_NonVariadicOpcodesReturnFalseForIsVariadic) {
  const std::vector<HloOpcode> non_variadic_opcodes = {
      HloOpcode::kAdd,
      HloOpcode::kAbs,
      HloOpcode::kCompare,
      HloOpcode::kConstant,
      HloOpcode::kParameter,
      HloOpcode::kCopy,
      HloOpcode::kWhile,
      HloOpcode::kAsyncStart,
      HloOpcode::kAsyncUpdate,
      HloOpcode::kAsyncDone,
  };

  for (HloOpcode opcode : non_variadic_opcodes) {
    EXPECT_FALSE(HloOpcodeIsVariadic(opcode)) << HloOpcodeString(opcode);
  }
}

TEST(HloOpcodeTest, BVA_ParameterHasArityZero) {
  std::optional<int> arity = HloOpcodeArity(HloOpcode::kParameter);

  ASSERT_TRUE(arity.has_value());
  EXPECT_EQ(arity.value(), 0);
  EXPECT_FALSE(HloOpcodeIsVariadic(HloOpcode::kParameter));
}

TEST(HloOpcodeTest, Edge_AsyncStartUpdateDoneHaveExpectedArityAndAsyncClassification) {
  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kAsyncStart).has_value());
  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kAsyncUpdate).has_value());
  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kAsyncDone).has_value());

  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncStart));
  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncUpdate));
  EXPECT_TRUE(HloOpcodeIsAsync(HloOpcode::kAsyncDone));

  EXPECT_FALSE(HloOpcodeIsVariadic(HloOpcode::kAsyncStart));
  EXPECT_FALSE(HloOpcodeIsVariadic(HloOpcode::kAsyncUpdate));
  EXPECT_FALSE(HloOpcodeIsVariadic(HloOpcode::kAsyncDone));
}

TEST(HloOpcodeTest, Edge_ComparisonOpcodeIsBinaryAndNotAsyncOrVariadic) {
  EXPECT_TRUE(HloOpcodeIsComparison(HloOpcode::kCompare));
  EXPECT_FALSE(HloOpcodeIsAsync(HloOpcode::kCompare));
  EXPECT_FALSE(HloOpcodeIsVariadic(HloOpcode::kCompare));

  ASSERT_TRUE(HloOpcodeArity(HloOpcode::kCompare).has_value());
  EXPECT_EQ(HloOpcodeArity(HloOpcode::kCompare).value(), 2);
}

TEST(HloOpcodeTest, Edge_StringToHloOpcodeRepeatedCallsReturnSameOpcode) {
  for (int i = 0; i < 10; ++i) {
    absl::StatusOr<HloOpcode> result = StringToHloOpcode("add");

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value(), HloOpcode::kAdd);
  }
}

}  // namespace
}  // namespace xla