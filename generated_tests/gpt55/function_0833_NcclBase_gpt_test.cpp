#include "tensorflow/core/kernels/collective_nccl.h"

#include <gtest/gtest.h>

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {

TEST(NcclBaseTest, ReductionCollectiveWithExpectedNameReturnsOk) {
  NcclBase nccl_base(REDUCTION_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCTION_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclReduce";

  TF_EXPECT_OK(nccl_base.InitializeCollectiveParams(&params));
}

TEST(NcclBaseTest, BroadcastCollectiveWithExpectedNameReturnsOk) {
  NcclBase nccl_base(BROADCAST_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = BROADCAST_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclBroadcast";

  TF_EXPECT_OK(nccl_base.InitializeCollectiveParams(&params));
}

TEST(NcclBaseTest, GatherCollectiveWithExpectedNameReturnsOk) {
  NcclBase nccl_base(GATHER_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = GATHER_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclGather";

  TF_EXPECT_OK(nccl_base.InitializeCollectiveParams(&params));
}

TEST(NcclBaseTest, ReduceScatterCollectiveWithExpectedNameReturnsOk) {
  NcclBase nccl_base(REDUCE_SCATTER_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCE_SCATTER_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclReduceScatter";

  TF_EXPECT_OK(nccl_base.InitializeCollectiveParams(&params));
}

TEST(NcclBaseTest, AllToAllCollectiveWithExpectedNameReturnsOk) {
  NcclBase nccl_base(ALL_TO_ALL_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = ALL_TO_ALL_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclAllToAll";

  TF_EXPECT_OK(nccl_base.InitializeCollectiveParams(&params));
}

TEST(NcclBaseTest, MismatchedTypeReturnsInternalError) {
  NcclBase nccl_base(REDUCTION_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = BROADCAST_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclBroadcast";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("Expected initialized type"),
            std::string::npos);
}

TEST(NcclBaseTest, ReductionWithWrongNameReturnsInternalError) {
  NcclBase nccl_base(REDUCTION_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCTION_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclBroadcast";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("Unexpected combination"),
            std::string::npos);
  EXPECT_NE(status.error_message().find("NcclReduce"), std::string::npos);
}

TEST(NcclBaseTest, BroadcastWithWrongNameReturnsInternalError) {
  NcclBase nccl_base(BROADCAST_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = BROADCAST_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclReduce";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("NcclBroadcast"), std::string::npos);
}

TEST(NcclBaseTest, GatherWithWrongNameReturnsInternalError) {
  NcclBase nccl_base(GATHER_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = GATHER_COLLECTIVE;
  params.instance.impl_details.collective_name = "WrongGather";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("NcclGather"), std::string::npos);
}

TEST(NcclBaseTest, ReduceScatterWithWrongNameReturnsInternalError) {
  NcclBase nccl_base(REDUCE_SCATTER_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCE_SCATTER_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclReduce";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("NcclReduceScatter"),
            std::string::npos);
}

TEST(NcclBaseTest, AllToAllWithWrongNameReturnsInternalError) {
  NcclBase nccl_base(ALL_TO_ALL_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = ALL_TO_ALL_COLLECTIVE;
  params.instance.impl_details.collective_name = "NcclGather";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("NcclAllToAll"), std::string::npos);
}

TEST(NcclBaseTest, EmptyCollectiveNameReturnsInternalError) {
  NcclBase nccl_base(REDUCTION_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCTION_COLLECTIVE;
  params.instance.impl_details.collective_name = "";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("Unexpected combination"),
            std::string::npos);
}

TEST(NcclBaseTest, CollectiveNameIsCaseSensitive) {
  NcclBase nccl_base(REDUCTION_COLLECTIVE, "test");
  CollectiveParams params;
  params.instance.type = REDUCTION_COLLECTIVE;
  params.instance.impl_details.collective_name = "ncclreduce";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
}

TEST(NcclBaseTest, UnexpectedCollectiveTypeReturnsInternalError) {
  const CollectiveType unexpected_type =
      static_cast<CollectiveType>(999);

  NcclBase nccl_base(unexpected_type, "test");
  CollectiveParams params;
  params.instance.type = unexpected_type;
  params.instance.impl_details.collective_name = "Unknown";

  Status status = nccl_base.InitializeCollectiveParams(&params);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INTERNAL);
  EXPECT_NE(status.error_message().find("Unexpected CollectiveType"),
            std::string::npos);
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM