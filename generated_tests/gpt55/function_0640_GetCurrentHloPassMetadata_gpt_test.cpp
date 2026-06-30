#include "xla/hlo/ir/hlo_module_metadata.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "xla/hlo/ir/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class FakeEnv : public tsl::EnvWrapper {
 public:
  FakeEnv() : tsl::EnvWrapper(tsl::Env::Default()) {}

  uint64_t NowMicros() override { return now_micros_++; }

 private:
  uint64_t now_micros_ = 1000;
};

TEST(HloModuleMetadataTest, GetCurrentHloPassMetadataWithoutRunningPassReturnsNotFound) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  auto result = metadata.GetCurrentHloPassMetadata();

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
}

TEST(HloModuleMetadataTest, MutateCurrentHloPassMetadataWithoutRunningPassReturnsNotFound) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  absl::Status status = metadata.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("should_not_run");
      });

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(metadata.proto().pass_metadata_size(), 0);
}

TEST(HloModuleMetadataTest, RecordPassStartCreatesRunningPassMetadata) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();

  ASSERT_EQ(metadata.proto().pass_metadata_size(), 1);
  EXPECT_EQ(metadata.proto().pass_metadata(0).pass_id(), 0);
  EXPECT_EQ(metadata.proto().pass_metadata(0).start_timestamp_usec(), 1000);
  EXPECT_EQ(metadata.proto().pass_metadata(0).end_timestamp_usec(), 0);

  TF_ASSERT_OK_AND_ASSIGN(HloPassMetadata* current,
                          metadata.GetCurrentHloPassMetadata());
  EXPECT_EQ(current->pass_id(), 0);
}

TEST(HloModuleMetadataTest, RecordPassStartAssignsSequentialPassIds) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();
  metadata.RecordPassStart();

  ASSERT_EQ(metadata.proto().pass_metadata_size(), 3);
  EXPECT_EQ(metadata.proto().pass_metadata(0).pass_id(), 0);
  EXPECT_EQ(metadata.proto().pass_metadata(1).pass_id(), 1);
  EXPECT_EQ(metadata.proto().pass_metadata(2).pass_id(), 2);
}

TEST(HloModuleMetadataTest, GetCurrentHloPassMetadataReturnsMostRecentRunningPass) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();

  TF_ASSERT_OK_AND_ASSIGN(HloPassMetadata* current,
                          metadata.GetCurrentHloPassMetadata());

  EXPECT_EQ(current->pass_id(), 1);
}

TEST(HloModuleMetadataTest, MutateCurrentHloPassMetadataMutatesMostRecentPass) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();

  TF_EXPECT_OK(metadata.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("current_pass");
      }));

  ASSERT_EQ(metadata.proto().pass_metadata_size(), 2);
  EXPECT_EQ(metadata.proto().pass_metadata(0).pass_name(), "");
  EXPECT_EQ(metadata.proto().pass_metadata(1).pass_name(), "current_pass");
}

TEST(HloModuleMetadataTest, RecordPassEndSetsEndTimestampAndPopsCurrentPass) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();

  TF_EXPECT_OK(metadata.RecordPassEnd());

  ASSERT_EQ(metadata.proto().pass_metadata_size(), 2);
  EXPECT_EQ(metadata.proto().pass_metadata(1).end_timestamp_usec(), 1002);

  TF_ASSERT_OK_AND_ASSIGN(HloPassMetadata* current,
                          metadata.GetCurrentHloPassMetadata());
  EXPECT_EQ(current->pass_id(), 0);
}

TEST(HloModuleMetadataTest, RecordPassEndWithoutRunningPassReturnsNotFound) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  absl::Status status = metadata.RecordPassEnd();

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(HloModuleMetadataTest, RecordPassEndAllRunningPassesThenCurrentIsNotFound) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();

  TF_EXPECT_OK(metadata.RecordPassEnd());
  TF_EXPECT_OK(metadata.RecordPassEnd());

  auto result = metadata.GetCurrentHloPassMetadata();

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
  ASSERT_EQ(metadata.proto().pass_metadata_size(), 2);
  EXPECT_EQ(metadata.proto().pass_metadata(0).end_timestamp_usec(), 1003);
  EXPECT_EQ(metadata.proto().pass_metadata(1).end_timestamp_usec(), 1002);
}

TEST(HloModuleMetadataTest, NestedPassesEndInLifoOrder) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();
  metadata.RecordPassStart();

  TF_EXPECT_OK(metadata.RecordPassEnd());
  TF_EXPECT_OK(metadata.RecordPassEnd());

  TF_ASSERT_OK_AND_ASSIGN(HloPassMetadata* current,
                          metadata.GetCurrentHloPassMetadata());

  EXPECT_EQ(current->pass_id(), 0);
  ASSERT_EQ(metadata.proto().pass_metadata_size(), 3);
  EXPECT_EQ(metadata.proto().pass_metadata(2).end_timestamp_usec(), 1003);
  EXPECT_EQ(metadata.proto().pass_metadata(1).end_timestamp_usec(), 1004);
  EXPECT_EQ(metadata.proto().pass_metadata(0).end_timestamp_usec(), 0);
}

TEST(HloModuleMetadataTest, SetCustomMetadataWithoutRunningPassReturnsNotFound) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  HloModuleProto custom_proto;
  custom_proto.set_name("custom");

  absl::Status status = metadata.set_custom_metadata(custom_proto);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(HloModuleMetadataTest, SetCustomMetadataOnCurrentPassStoresPackedMessage) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();

  HloModuleProto custom_proto;
  custom_proto.set_name("custom_module");

  TF_EXPECT_OK(metadata.set_custom_metadata(custom_proto));

  ASSERT_EQ(metadata.proto().pass_metadata_size(), 1);
  ASSERT_TRUE(metadata.proto().pass_metadata(0).has_custom_metadata());

  HloModuleProto unpacked;
  EXPECT_TRUE(metadata.proto()
                  .pass_metadata(0)
                  .custom_metadata()
                  .UnpackTo(&unpacked));
  EXPECT_EQ(unpacked.name(), "custom_module");
}

TEST(HloModuleMetadataTest, SetCustomMetadataOnlyMutatesCurrentPass) {
  FakeEnv env;
  HloModuleMetadata metadata(&env);

  metadata.RecordPassStart();
  metadata.RecordPassStart();

  HloModuleProto custom_proto;
  custom_proto.set_name("current_only");

  TF_EXPECT_OK(metadata.set_custom_metadata(custom_proto));

  EXPECT_FALSE(metadata.proto().pass_metadata(0).has_custom_metadata());
  EXPECT_TRUE(metadata.proto().pass_metadata(1).has_custom_metadata());
}

TEST(HloModuleMetadataTest, SetPrepartitioningMetadataWithNoRunningPassesCopiesCompletedPassesToPrepartitioningOnly) {
  FakeEnv env;
  HloModuleMetadata source(&env);

  source.mutable_proto()->set_canonical_module_id(123);
  source.RecordPassStart();
  TF_EXPECT_OK(source.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("completed");
      }));
  TF_EXPECT_OK(source.RecordPassEnd());

  HloModuleMetadata destination(&env);
  destination.set_prepartitioning_metadata(source);

  EXPECT_EQ(destination.proto().original_module_id(), 123);
  EXPECT_EQ(destination.proto().pass_metadata_size(), 0);
  ASSERT_TRUE(destination.prepartitioning_metadata().has_value());
  EXPECT_EQ(destination.prepartitioning_metadata()->pass_metadata_size(), 1);
  EXPECT_EQ(destination.prepartitioning_metadata()->pass_metadata(0).pass_name(),
            "completed");
}

TEST(HloModuleMetadataTest, SetPrepartitioningMetadataCopiesRunningPassesIntoCurrentMetadata) {
  FakeEnv env;
  HloModuleMetadata source(&env);

  source.mutable_proto()->set_canonical_module_id(456);

  source.RecordPassStart();
  TF_EXPECT_OK(source.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("running_outer");
      }));

  source.RecordPassStart();
  TF_EXPECT_OK(source.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("running_inner");
      }));

  HloModuleMetadata destination(&env);
  destination.set_prepartitioning_metadata(source);

  EXPECT_EQ(destination.proto().original_module_id(), 456);
  ASSERT_EQ(destination.proto().pass_metadata_size(), 2);
  EXPECT_EQ(destination.proto().pass_metadata(0).pass_name(), "running_outer");
  EXPECT_EQ(destination.proto().pass_metadata(1).pass_name(), "running_inner");

  ASSERT_TRUE(destination.prepartitioning_metadata().has_value());
  EXPECT_EQ(destination.prepartitioning_metadata()->pass_metadata_size(), 0);

  TF_ASSERT_OK_AND_ASSIGN(HloPassMetadata* current,
                          destination.GetCurrentHloPassMetadata());
  EXPECT_EQ(current->pass_name(), "running_inner");
}

TEST(HloModuleMetadataTest, SetPrepartitioningMetadataSplitsCompletedAndRunningPasses) {
  FakeEnv env;
  HloModuleMetadata source(&env);

  source.mutable_proto()->set_canonical_module_id(789);

  source.RecordPassStart();
  TF_EXPECT_OK(source.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("completed");
      }));
  TF_EXPECT_OK(source.RecordPassEnd());

  source.RecordPassStart();
  TF_EXPECT_OK(source.MutateCurrentHloPassMetadata(
      [](HloPassMetadata* pass_metadata) {
        pass_metadata->set_pass_name("running");
      }));

  HloModuleMetadata destination(&env);
  destination.set_prepartitioning_metadata(source);

  EXPECT_EQ(destination.proto().original_module_id(), 789);

  ASSERT_EQ(destination.proto().pass_metadata_size(), 1);
  EXPECT_EQ(destination.proto().pass_metadata(0).pass_name(), "running");

  ASSERT_TRUE(destination.prepartitioning_metadata().has_value());
  ASSERT_EQ(destination.prepartitioning_metadata()->pass_metadata_size(), 1);
  EXPECT_EQ(destination.prepartitioning_metadata()->pass_metadata(0).pass_name(),
            "completed");
}

TEST(HloModuleMetadataTest, SetPrepartitioningMetadataUpdatesNextPassIdAfterRunningPasses) {
  FakeEnv env;
  HloModuleMetadata source(&env);

  source.RecordPassStart();
  source.RecordPassStart();

  HloModuleMetadata destination(&env);
  destination.set_prepartitioning_metadata(source);

  metadata.RecordPassStart();
}

TEST(HloModuleMetadataTest, SetPrepartitioningMetadataThenRecordPassStartUsesNextAvailablePassId) {
  FakeEnv env;
  HloModuleMetadata source(&env);

  source.RecordPassStart();
  source.RecordPassStart();

  HloModuleMetadata destination(&env);
  destination.set_prepartitioning_metadata(source);

  destination.RecordPassStart();

  ASSERT_EQ(destination.proto().pass_metadata_size(), 3);
  EXPECT_EQ(destination.proto().pass_metadata(0).pass_id(), 0);
  EXPECT_EQ(destination.proto().pass_metadata(1).pass_id(), 1);
  EXPECT_EQ(destination.proto().pass_metadata(2).pass_id(), 2);
}

}  // namespace
}  // namespace xla