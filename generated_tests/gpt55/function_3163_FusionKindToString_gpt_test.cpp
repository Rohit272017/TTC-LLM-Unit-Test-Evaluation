#include "xla/service/gpu/transforms/rename_fusions.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class RenameFusionsTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<HloModule> ParseModuleOrDie(const std::string& hlo) {
    auto module = ParseAndReturnVerifiedModule(hlo);
    CHECK_OK(module.status());
    return std::move(module).value();
  }

  HloInstruction* FindInstruction(HloModule* module, absl::string_view name) {
    return FindInstruction(module, name);
  }

  HloInstruction* FindInstr(HloModule* module, absl::string_view name) {
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->name() == name) {
          return instruction;
        }
      }
    }
    return nullptr;
  }
};

TEST_F(RenameFusionsTest, LoopFusionWithAddHeroIsRenamed) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->name(), "loop_add_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_add");
}

TEST_F(RenameFusionsTest, InputFusionWithMultiplyHeroIsRenamed) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT multiply = f32[4] multiply(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = f32[4] fusion(a, b), kind=kInput, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "input_multiply_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_multiply");
}

TEST_F(RenameFusionsTest, OutputFusionWithSubtractHeroIsRenamed) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT subtract = f32[4] subtract(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = f32[4] fusion(a, b), kind=kOutput, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "output_subtract_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_subtract");
}

TEST_F(RenameFusionsTest, CustomFusionIsSkipped) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT custom_fusion = f32[4] fusion(a, b), kind=kCustom, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "custom_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_computation");
}

TEST_F(RenameFusionsTest, ModuleWithoutFusionStillReturnsTrue) {
  const std::string hlo = R"(
HloModule test

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT add = f32[4] add(a, b)
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "add");
}

TEST_F(RenameFusionsTest, MultipleFusionsWithSameHeroAreUniquified) {
  const std::string hlo = R"(
HloModule test

fused_add_0 {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

fused_add_1 {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  fusion0 = f32[4] fusion(a, b), kind=kLoop, calls=fused_add_0
  fusion1 = f32[4] fusion(a, b), kind=kLoop, calls=fused_add_1
  ROOT tuple = (f32[4], f32[4]) tuple(fusion0, fusion1)
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion0 = FindInstr(module.get(), "loop_add_fusion");
  HloInstruction* fusion1 = FindInstr(module.get(), "loop_add_fusion.1");

  ASSERT_NE(fusion0, nullptr);
  ASSERT_NE(fusion1, nullptr);
  EXPECT_EQ(fusion0->fused_instructions_computation()->name(), "fused_add");
  EXPECT_EQ(fusion1->fused_instructions_computation()->name(), "fused_add.1");
}

TEST_F(RenameFusionsTest, ExistingInstructionNameCollisionIsUniquified) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  loop_add_fusion = f32[4] add(a, b)
  fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation
  ROOT tuple = (f32[4], f32[4]) tuple(loop_add_fusion, fusion)
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  EXPECT_NE(FindInstr(module.get(), "loop_add_fusion"), nullptr);
  EXPECT_NE(FindInstr(module.get(), "loop_add_fusion.1"), nullptr);
}

TEST_F(RenameFusionsTest, ExistingComputationNameCollisionIsUniquified) {
  const std::string hlo = R"(
HloModule test

fused_add {
  p0 = f32[4] parameter(0)
  ROOT negate = f32[4] negate(p0)
}

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation
  ROOT tuple = (f32[4]) tuple(fusion)
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = FindInstr(module.get(), "loop_add_fusion");
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_add.1");
}

TEST_F(RenameFusionsTest, FusionWithTupleRootUsesAllHeroNamesSorted) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  add = f32[4] add(p0, p1)
  multiply = f32[4] multiply(p0, p1)
  ROOT tuple = (f32[4], f32[4]) tuple(multiply, add)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = (f32[4], f32[4]) fusion(a, b), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "loop_add_multiply_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(),
            "fused_add_multiply");
}

TEST_F(RenameFusionsTest, DuplicateHeroNamesAreDeduplicated) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  add0 = f32[4] add(p0, p1)
  add1 = f32[4] add(p1, p0)
  ROOT tuple = (f32[4], f32[4]) tuple(add0, add1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = (f32[4], f32[4]) fusion(a, b), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "loop_add_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_add");
}

TEST_F(RenameFusionsTest, HyphenInOpcodeNameIsReplacedWithUnderscore) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  ROOT abs = f32[4] abs(p0)
}

ENTRY main {
  a = f32[4] parameter(0)
  ROOT fusion = f32[4] fusion(a), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "loop_abs_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_abs");
}

TEST_F(RenameFusionsTest, NonTrivialHeroSkipsBitcastRoot) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[2,2] parameter(0)
  p1 = f32[2,2] parameter(1)
  add = f32[2,2] add(p0, p1)
  ROOT bitcast = f32[4] bitcast(add)
}

ENTRY main {
  a = f32[2,2] parameter(0)
  b = f32[2,2] parameter(1)
  ROOT fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "loop_add_fusion");
  EXPECT_EQ(fusion->fused_instructions_computation()->name(), "fused_add");
}

TEST_F(RenameFusionsTest, ExecutionThreadsArgumentDoesNotRestrictRenaming) {
  const std::string hlo = R"(
HloModule test

fused_computation {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT add = f32[4] add(p0, p1)
}

ENTRY main {
  a = f32[4] parameter(0)
  b = f32[4] parameter(1)
  ROOT fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation
}
)";

  auto module = ParseModuleOrDie(hlo);
  RenameFusions pass;

  EXPECT_THAT(pass.Run(module.get(), {"non_main_thread"}), IsOkAndHolds(true));

  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_EQ(fusion->name(), "loop_add_fusion");
}

}  // namespace
}  // namespace gpu
}  // namespace xla