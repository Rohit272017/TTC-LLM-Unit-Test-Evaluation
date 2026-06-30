#include "tensorflow/core/profiler/utils/xprof_gpu_cost_analysis.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_parser.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::xla::HloInstruction;
using ::xla::HloModule;
using ::xla::ShapeUtil;

class XProfGpuCostAnalysisTest : public ::testing::Test {
 protected:
  std::unique_ptr<HloModule> ParseModule(const std::string& hlo_text) {
    auto module_or = xla::ParseAndReturnUnverifiedModule(hlo_text);
    EXPECT_TRUE(module_or.ok()) << module_or.status();
    return std::move(module_or).value();
  }

  HloInstruction* FindInstruction(HloModule* module, const std::string& name) {
    for (auto* computation : module->computations()) {
      for (auto* instruction : computation->instructions()) {
        if (instruction->name() == name) {
          return instruction;
        }
      }
    }
    return nullptr;
  }

  XProfGpuCostAnalysis CreateAnalysis() {
    xla::HloCostAnalysis::Options options;
    return XProfGpuCostAnalysis(options);
  }
};

TEST_F(XProfGpuCostAnalysisTest, ECP_NullInstructionReturnsOkStatus) {
  XProfGpuCostAnalysis analysis = CreateAnalysis();

  absl::Status status = analysis.Postprocess(nullptr);

  EXPECT_TRUE(status.ok());
}

TEST_F(XProfGpuCostAnalysisTest, ECP_CreateNestedCostAnalysisReturnsXProfGpuCostAnalysis) {
  XProfGpuCostAnalysis analysis = CreateAnalysis();

  std::unique_ptr<xla::HloCostAnalysis> nested =
      analysis.CreateNestedCostAnalysis();

  ASSERT_NE(nested, nullptr);
  EXPECT_NE(dynamic_cast<XProfGpuCostAnalysis*>(nested.get()), nullptr);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_Float32InputDoesNotApplyFlopRateAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  add = f32[8] add(p0, p1)
  ROOT root = f32[8] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), 0);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_Float16InputDoesNotApplyFlopRateAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = f16[8] parameter(0)
  p1 = f16[8] parameter(1)
  add = f16[8] add(p0, p1)
  ROOT root = f16[8] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), 0);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_Int32InputDoesNotApplyFlopRateAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s32[8] parameter(0)
  p1 = s32[8] parameter(1)
  add = s32[8] add(p0, p1)
  ROOT root = s32[8] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), 0);
}

TEST_F(XProfGpuCostAnalysisTest, BVA_Int8InputAppliesHalfModelFlopsAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s8[8] parameter(0)
  p1 = s8[8] parameter(1)
  add = s8[8] add(p0, p1)
  ROOT root = s8[8] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), model_flops / 2);
}

TEST_F(XProfGpuCostAnalysisTest, BVA_UInt8InputAppliesHalfModelFlopsAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = u8[16] parameter(0)
  p1 = u8[16] parameter(1)
  add = u8[16] add(p0, p1)
  ROOT root = u8[16] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), model_flops / 2);
}

TEST_F(XProfGpuCostAnalysisTest, BVA_Int4InputAppliesThreeQuarterModelFlopsAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s4[16] parameter(0)
  p1 = s4[16] parameter(1)
  add = s4[16] add(p0, p1)
  ROOT root = s4[16] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add),
            model_flops - model_flops / 4);
}

TEST_F(XProfGpuCostAnalysisTest, BVA_UInt4InputAppliesThreeQuarterModelFlopsAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = u4[16] parameter(0)
  p1 = u4[16] parameter(1)
  add = u4[16] add(p0, p1)
  ROOT root = u4[16] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add),
            model_flops - model_flops / 4);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_MixedInputBitwidthUsesMaximumBitwidth) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s8[16] parameter(0)
  c0 = s32[] constant(1)
  broadcast = s32[16] broadcast(c0), dimensions={}
  convert = s8[16] convert(broadcast)
  add = s8[16] add(p0, convert)
  ROOT root = s8[16] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), model_flops / 2);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_InstructionWithNoOperandsHasNoAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  ROOT constant = f32[] constant(1)
}
)");

  HloInstruction* constant = FindInstruction(module.get(), "constant");
  ASSERT_NE(constant, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(constant->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*constant), 0);
}

TEST_F(XProfGpuCostAnalysisTest, Edge_TupleOperandIsIgnoredForInputBitwidth) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  tuple = (f32[4], f32[4]) tuple(p0, p1)
  gte = f32[4] get-tuple-element(tuple), index=0
  ROOT root = f32[4] copy(gte)
}
)");

  HloInstruction* gte = FindInstruction(module.get(), "gte");
  ASSERT_NE(gte, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(gte->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*gte), 0);
}

TEST_F(XProfGpuCostAnalysisTest, Edge_TokenOperandIsIgnoredForInputBitwidth) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  token0 = token[] after-all()
  ROOT root = token[] after-all(token0)
}
)");

  HloInstruction* root = FindInstruction(module.get(), "root");
  ASSERT_NE(root, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(root->Accept(&analysis).ok());

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*root), 0);
}

TEST_F(XProfGpuCostAnalysisTest, Edge_ZeroElementArrayProducesZeroAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s8[0] parameter(0)
  p1 = s8[0] parameter(1)
  add = s8[0] add(p0, p1)
  ROOT root = s8[0] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  EXPECT_EQ(analysis.flop_count(*add), 0);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), 0);
}

TEST_F(XProfGpuCostAnalysisTest, BVA_SingleElementInt8InputAdjustment) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s8[1] parameter(0)
  p1 = s8[1] parameter(1)
  add = s8[1] add(p0, p1)
  ROOT root = s8[1] copy(add)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  ASSERT_TRUE(add->Accept(&analysis).ok());

  const int64_t model_flops = analysis.flop_count(*add);
  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), model_flops / 2);
}

TEST_F(XProfGpuCostAnalysisTest, ECP_GetDeviceFlopsAdjustmentForUnvisitedInstructionReturnsZero) {
  auto module = ParseModule(R"(
HloModule test

ENTRY main {
  p0 = s8[8] parameter(0)
  p1 = s8[8] parameter(1)
  ROOT add = s8[8] add(p0, p1)
}
)");

  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);

  XProfGpuCostAnalysis analysis = CreateAnalysis();

  EXPECT_EQ(analysis.GetDeviceFlopsAdjustment(*add), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow