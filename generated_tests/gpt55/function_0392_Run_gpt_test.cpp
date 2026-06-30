#include "xla/service/collectives_schedule_linearizer.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class CollectivesScheduleLinearizerTest
    : public HloHardwareIndependentTestBase {};

TEST_F(CollectivesScheduleLinearizerTest, EmptyModuleReturnsUnchanged) {
  const char* hlo = R"(
HloModule empty

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT copy = f32[4] copy(p0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(CollectivesScheduleLinearizerTest, SingleCollectiveReturnsUnchanged) {
  const char* hlo = R"(
HloModule single_collective

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT ar = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_FALSE(changed);
}

TEST_F(CollectivesScheduleLinearizerTest,
       IndependentCollectivesGetControlDependency) {
  const char* hlo = R"(
HloModule independent_collectives

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4]) tuple(ar0, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar0 =
      module->entry_computation()->GetInstructionWithName("ar0");
  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar0, nullptr);
  ASSERT_NE(ar1, nullptr);
  EXPECT_TRUE(ar1->control_predecessors().empty());

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar0));
}

TEST_F(CollectivesScheduleLinearizerTest,
       AlreadyConnectedCollectivesDoNotAddControlDependency) {
  const char* hlo = R"(
HloModule connected_collectives

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(ar0), replica_groups={}, to_apply=sum
  ROOT copy = f32[4] copy(ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_FALSE(changed);
  EXPECT_TRUE(ar1->control_predecessors().empty());
}

TEST_F(CollectivesScheduleLinearizerTest,
       ThreeIndependentCollectivesAreLinearizedInPostOrder) {
  const char* hlo = R"(
HloModule three_independent_collectives

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  p2 = f32[4] parameter(2)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ar2 = f32[4] all-reduce(p2), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4], f32[4]) tuple(ar0, ar1, ar2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar0 =
      module->entry_computation()->GetInstructionWithName("ar0");
  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  HloInstruction* ar2 =
      module->entry_computation()->GetInstructionWithName("ar2");
  ASSERT_NE(ar0, nullptr);
  ASSERT_NE(ar1, nullptr);
  ASSERT_NE(ar2, nullptr);

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar0));
  EXPECT_THAT(ar2->control_predecessors(), ::testing::Contains(ar1));
}

TEST_F(CollectivesScheduleLinearizerTest,
       ExistingControlDependencyPreventsAdditionalChange) {
  const char* hlo = R"(
HloModule control_connected_collectives

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum, control-predecessors={ar0}
  ROOT tuple = (f32[4], f32[4]) tuple(ar0, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar0 =
      module->entry_computation()->GetInstructionWithName("ar0");
  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar0, nullptr);
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_FALSE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar0));
}

TEST_F(CollectivesScheduleLinearizerTest,
       DisabledByPredicateReturnsFalseAndDoesNotModifyModule) {
  const char* hlo = R"(
HloModule disabled_by_predicate

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4]) tuple(ar0, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer(
      [](const HloModule*) { return false; });

  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_FALSE(changed);
  EXPECT_TRUE(ar1->control_predecessors().empty());
}

TEST_F(CollectivesScheduleLinearizerTest,
       EnabledByPredicateLinearizesIndependentCollectives) {
  const char* hlo = R"(
HloModule enabled_by_predicate

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4]) tuple(ar0, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar0 =
      module->entry_computation()->GetInstructionWithName("ar0");
  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar0, nullptr);
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer(
      [](const HloModule*) { return true; });

  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar0));
}

TEST_F(CollectivesScheduleLinearizerTest,
       LinearizesAllReduceStartDoneUsingDoneAsPrevious) {
  const char* hlo = R"(
HloModule async_collectives

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar_start = (f32[4], f32[4], u32[], u32[]) all-reduce-start(p0), replica_groups={}, to_apply=sum
  ar_done = f32[4] all-reduce-done(ar_start)
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4]) tuple(ar_done, ar1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* ar_done =
      module->entry_computation()->GetInstructionWithName("ar_done");
  HloInstruction* ar1 =
      module->entry_computation()->GetInstructionWithName("ar1");
  ASSERT_NE(ar_done, nullptr);
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar_done));
}

TEST_F(CollectivesScheduleLinearizerTest,
       NonEntryNonFusionComputationIsAlsoProcessed) {
  const char* hlo = R"(
HloModule non_entry_computation

sum {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

called {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ar0 = f32[4] all-reduce(p0), replica_groups={}, to_apply=sum
  ar1 = f32[4] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT tuple = (f32[4], f32[4]) tuple(ar0, ar1)
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT call = (f32[4], f32[4]) call(p0, p0), to_apply=called
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloComputation* called =
      module->GetComputationWithName("called");
  ASSERT_NE(called, nullptr);

  HloInstruction* ar0 = called->GetInstructionWithName("ar0");
  HloInstruction* ar1 = called->GetInstructionWithName("ar1");
  ASSERT_NE(ar0, nullptr);
  ASSERT_NE(ar1, nullptr);

  CollectivesScheduleLinearizer linearizer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, linearizer.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_THAT(ar1->control_predecessors(), ::testing::Contains(ar0));
}

}  // namespace
}  // namespace xla