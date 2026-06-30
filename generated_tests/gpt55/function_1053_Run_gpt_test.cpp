#include "xla/service/dynamic_index_splitter.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class DynamicIndexSplitterTest : public HloHardwareIndependentTestBase {
 protected:
  HloInstruction* FindInstruction(HloModule* module, const std::string& name) {
    return module->entry_computation()->GetInstructionWithName(name);
  }
};

TEST_F(DynamicIndexSplitterTest, ModuleWithoutDynamicOpsReturnsUnchanged) {
  const char* hlo = R"(
HloModule no_dynamic_ops

ENTRY main {
  p0 = f32[4,4] parameter(0)
  ROOT copy = f32[4,4] copy(p0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(DynamicIndexSplitterTest, DynamicSliceWithScalarIndicesReturnsUnchanged) {
  const char* hlo = R"(
HloModule scalar_indices_dynamic_slice

ENTRY main {
  operand = f32[8,8] parameter(0)
  i0 = s32[] parameter(1)
  i1 = s32[] parameter(2)
  ROOT ds = f32[2,2] dynamic-slice(operand, i0, i1),
      dynamic_slice_sizes={2,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(DynamicIndexSplitterTest, DynamicUpdateSliceWithScalarIndicesReturnsUnchanged) {
  const char* hlo = R"(
HloModule scalar_indices_dynamic_update_slice

ENTRY main {
  operand = f32[8,8] parameter(0)
  update = f32[2,2] parameter(1)
  i0 = s32[] parameter(2)
  i1 = s32[] parameter(3)
  ROOT dus = f32[8,8] dynamic-update-slice(operand, update, i0, i1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(DynamicIndexSplitterTest, DynamicSliceWithVectorIndexIsSplitIntoScalarIndices) {
  const char* hlo = R"(
HloModule vector_index_dynamic_slice

ENTRY main {
  operand = f32[8,8] parameter(0)
  indices = s32[2] parameter(1)
  ROOT ds = f32[2,2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(2)->shape()));
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(1)->operand(0)->opcode(), HloOpcode::kSlice);
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kSlice);
}

TEST_F(DynamicIndexSplitterTest, DynamicUpdateSliceWithVectorIndexIsSplitIntoScalarIndices) {
  const char* hlo = R"(
HloModule vector_index_dynamic_update_slice

ENTRY main {
  operand = f32[8,8] parameter(0)
  update = f32[2,2] parameter(1)
  indices = s32[2] parameter(2)
  ROOT dus = f32[8,8] dynamic-update-slice(operand, update, indices)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicUpdateSlice);
  EXPECT_EQ(root->operand_count(), 4);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(2)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(3)->shape()));
  EXPECT_EQ(root->operand(2)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(3)->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kSlice);
  EXPECT_EQ(root->operand(3)->operand(0)->opcode(), HloOpcode::kSlice);
}

TEST_F(DynamicIndexSplitterTest, RankOneOperandDynamicSliceIsReplacedByOperand) {
  const char* hlo = R"(
HloModule rank_zero_operand_dynamic_slice

ENTRY main {
  operand = f32[] parameter(0)
  ROOT ds = f32[] dynamic-slice(operand),
      dynamic_slice_sizes={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* operand = FindInstruction(module.get(), "operand");
  ASSERT_NE(operand, nullptr);

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction(), operand);
}

TEST_F(DynamicIndexSplitterTest, RankZeroOperandDynamicUpdateSliceIsReplacedByUpdate) {
  const char* hlo = R"(
HloModule rank_zero_operand_dynamic_update_slice

ENTRY main {
  operand = f32[] parameter(0)
  update = f32[] parameter(1)
  ROOT dus = f32[] dynamic-update-slice(operand, update)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_EQ(module->entry_computation()->root_instruction(), update);
}

TEST_F(DynamicIndexSplitterTest, VectorIndexLengthOneForRankOneSliceIsSplit) {
  const char* hlo = R"(
HloModule rank_one_vector_index_dynamic_slice

ENTRY main {
  operand = f32[8] parameter(0)
  indices = s32[1] parameter(1)
  ROOT ds = f32[2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(1)->shape()));
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kReshape);
}

TEST_F(DynamicIndexSplitterTest, VectorIndexForRankThreeSliceCreatesThreeScalarIndices) {
  const char* hlo = R"(
HloModule rank_three_vector_index_dynamic_slice

ENTRY main {
  operand = f32[8,8,8] parameter(0)
  indices = s32[3] parameter(1)
  ROOT ds = f32[2,2,2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2,2,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand_count(), 4);
  for (int i = 1; i <= 3; ++i) {
    EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(i)->shape()));
    EXPECT_EQ(root->operand(i)->opcode(), HloOpcode::kReshape);
    ASSERT_EQ(root->operand(i)->operand_count(), 1);
    EXPECT_EQ(root->operand(i)->operand(0)->opcode(), HloOpcode::kSlice);
  }
}

TEST_F(DynamicIndexSplitterTest, VectorIndexElementTypeIsPreservedAsS64) {
  const char* hlo = R"(
HloModule s64_vector_index_dynamic_slice

ENTRY main {
  operand = f32[8,8] parameter(0)
  indices = s64[2] parameter(1)
  ROOT ds = f32[2,2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2,2}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand(1)->shape().element_type(), S64);
  EXPECT_EQ(root->operand(2)->shape().element_type(), S64);
}

TEST_F(DynamicIndexSplitterTest, MultipleDynamicOpsAreAllSplit) {
  const char* hlo = R"(
HloModule multiple_dynamic_ops

ENTRY main {
  operand = f32[8,8] parameter(0)
  update = f32[2,2] parameter(1)
  indices0 = s32[2] parameter(2)
  indices1 = s32[2] parameter(3)
  ds = f32[2,2] dynamic-slice(operand, indices0),
      dynamic_slice_sizes={2,2}
  dus = f32[8,8] dynamic-update-slice(operand, update, indices1)
  ROOT tuple = (f32[2,2], f32[8,8]) tuple(ds, dus)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* tuple = module->entry_computation()->root_instruction();
  ASSERT_EQ(tuple->opcode(), HloOpcode::kTuple);

  HloInstruction* ds = tuple->operand(0);
  HloInstruction* dus = tuple->operand(1);

  ASSERT_EQ(ds->opcode(), HloOpcode::kDynamicSlice);
  ASSERT_EQ(dus->opcode(), HloOpcode::kDynamicUpdateSlice);
  EXPECT_EQ(ds->operand_count(), 3);
  EXPECT_EQ(dus->operand_count(), 4);
  EXPECT_TRUE(ShapeUtil::IsScalar(ds->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(ds->operand(2)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(dus->operand(2)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(dus->operand(3)->shape()));
}

TEST_F(DynamicIndexSplitterTest, NonEntryNonFusionComputationIsProcessed) {
  const char* hlo = R"(
HloModule non_entry_computation

called {
  operand = f32[8,8] parameter(0)
  indices = s32[2] parameter(1)
  ROOT ds = f32[2,2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2,2}
}

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT call = f32[2,2] call(p0, p1), to_apply=called
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_TRUE(changed);

  HloComputation* called = module->GetComputationWithName("called");
  ASSERT_NE(called, nullptr);

  HloInstruction* root = called->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand_count(), 3);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(2)->shape()));
}

TEST_F(DynamicIndexSplitterTest, FusionComputationIsNotProcessed) {
  const char* hlo = R"(
HloModule fusion_not_processed

fused_computation {
  operand = f32[8,8] parameter(0)
  indices = s32[2] parameter(1)
  ROOT ds = f32[2,2] dynamic-slice(operand, indices),
      dynamic_slice_sizes={2,2}
}

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = s32[2] parameter(1)
  ROOT fusion = f32[2,2] fusion(p0, p1), kind=kLoop,
      calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  DynamicIndexSplitter splitter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, splitter.Run(module.get()));

  EXPECT_FALSE(changed);

  HloComputation* fused = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fused, nullptr);

  HloInstruction* root = fused->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kDynamicSlice);
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_FALSE(ShapeUtil::IsScalar(root->operand(1)->shape()));
}

}  // namespace
}  // namespace xla