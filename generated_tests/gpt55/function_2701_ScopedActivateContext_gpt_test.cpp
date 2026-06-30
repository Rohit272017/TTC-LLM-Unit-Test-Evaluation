#include "xla/stream_executor/gpu/scoped_activate_context.h"

#include <gtest/gtest.h>

#include "xla/stream_executor/gpu/context.h"

namespace stream_executor::gpu {
namespace {

class FakeContext : public Context {
 public:
  explicit FakeContext(int device_ordinal) : device_ordinal_(device_ordinal) {}

  void SetActive() override {
    is_active_ = true;
    ++set_active_count_;
    active_context_ = this;
  }

  bool IsActive() const override { return active_context_ == this; }

  int device_ordinal() const override { return device_ordinal_; }

  int set_active_count() const { return set_active_count_; }

 private:
  int device_ordinal_;
  int set_active_count_ = 0;
  inline static thread_local FakeContext* active_context_ = nullptr;
  bool is_active_ = false;
};

TEST(ScopedActivateContextTest, ECP_FirstScopeActivatesContext) {
  FakeContext context0(0);

  {
    ScopedActivateContext scoped(&context0);

    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);
  }

  EXPECT_TRUE(context0.IsActive());
  EXPECT_EQ(context0.set_active_count(), 1);
}

TEST(ScopedActivateContextTest, ECP_NestedSameDeviceDoesNotReactivateContext) {
  FakeContext context0(0);

  {
    ScopedActivateContext outer(&context0);
    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);

    {
      ScopedActivateContext inner(&context0);
      EXPECT_TRUE(context0.IsActive());
      EXPECT_EQ(context0.set_active_count(), 1);
    }

    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);
  }

  EXPECT_TRUE(context0.IsActive());
  EXPECT_EQ(context0.set_active_count(), 1);
}

TEST(ScopedActivateContextTest, ECP_NestedDifferentDeviceSwitchesAndRestores) {
  FakeContext context0(0);
  FakeContext context1(1);

  {
    ScopedActivateContext outer(&context0);
    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);

    {
      ScopedActivateContext inner(&context1);
      EXPECT_TRUE(context1.IsActive());
      EXPECT_FALSE(context0.IsActive());
      EXPECT_EQ(context1.set_active_count(), 1);
    }

    EXPECT_TRUE(context0.IsActive());
    EXPECT_FALSE(context1.IsActive());
    EXPECT_EQ(context0.set_active_count(), 2);
  }
}

TEST(ScopedActivateContextTest, ECP_MultipleNestedDifferentDevicesRestoreInLifoOrder) {
  FakeContext context0(0);
  FakeContext context1(1);
  FakeContext context2(2);

  {
    ScopedActivateContext scope0(&context0);
    EXPECT_TRUE(context0.IsActive());

    {
      ScopedActivateContext scope1(&context1);
      EXPECT_TRUE(context1.IsActive());

      {
        ScopedActivateContext scope2(&context2);
        EXPECT_TRUE(context2.IsActive());
      }

      EXPECT_TRUE(context1.IsActive());
      EXPECT_FALSE(context2.IsActive());
    }

    EXPECT_TRUE(context0.IsActive());
    EXPECT_FALSE(context1.IsActive());
  }

  EXPECT_EQ(context0.set_active_count(), 2);
  EXPECT_EQ(context1.set_active_count(), 2);
  EXPECT_EQ(context2.set_active_count(), 1);
}

TEST(ScopedActivateContextTest, BVA_DeviceOrdinalZeroIsSupported) {
  FakeContext context0(0);

  {
    ScopedActivateContext scoped(&context0);

    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.device_ordinal(), 0);
    EXPECT_EQ(context0.set_active_count(), 1);
  }
}

TEST(ScopedActivateContextTest, BVA_NegativeDeviceOrdinalIsSupported) {
  FakeContext context_negative(-1);

  {
    ScopedActivateContext scoped(&context_negative);

    EXPECT_TRUE(context_negative.IsActive());
    EXPECT_EQ(context_negative.device_ordinal(), -1);
    EXPECT_EQ(context_negative.set_active_count(), 1);
  }
}

TEST(ScopedActivateContextTest, BVA_LargeDeviceOrdinalIsSupported) {
  FakeContext context_large(1024);

  {
    ScopedActivateContext scoped(&context_large);

    EXPECT_TRUE(context_large.IsActive());
    EXPECT_EQ(context_large.device_ordinal(), 1024);
    EXPECT_EQ(context_large.set_active_count(), 1);
  }
}

TEST(ScopedActivateContextTest, Edge_SequentialScopesSameContextEachActivateAtDepthZero) {
  FakeContext context0(0);

  {
    ScopedActivateContext scoped(&context0);
    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);
  }

  {
    ScopedActivateContext scoped(&context0);
    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 2);
  }
}

TEST(ScopedActivateContextTest, Edge_SequentialScopesDifferentContextsActivateEachOne) {
  FakeContext context0(0);
  FakeContext context1(1);

  {
    ScopedActivateContext scoped(&context0);
    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);
  }

  {
    ScopedActivateContext scoped(&context1);
    EXPECT_TRUE(context1.IsActive());
    EXPECT_FALSE(context0.IsActive());
    EXPECT_EQ(context1.set_active_count(), 1);
  }
}

TEST(ScopedActivateContextTest, Edge_OuterSameInnerDifferentThenSameAgain) {
  FakeContext context0(0);
  FakeContext context1(1);

  {
    ScopedActivateContext outer(&context0);
    EXPECT_TRUE(context0.IsActive());

    {
      ScopedActivateContext inner_different(&context1);
      EXPECT_TRUE(context1.IsActive());
    }

    EXPECT_TRUE(context0.IsActive());

    {
      ScopedActivateContext inner_same(&context0);
      EXPECT_TRUE(context0.IsActive());
    }

    EXPECT_TRUE(context0.IsActive());
  }

  EXPECT_EQ(context0.set_active_count(), 2);
  EXPECT_EQ(context1.set_active_count(), 1);
}

TEST(ScopedActivateContextTest, Edge_DeepNestedSameContextOnlyFirstActivation) {
  FakeContext context0(0);

  {
    ScopedActivateContext scope1(&context0);
    ScopedActivateContext scope2(&context0);
    ScopedActivateContext scope3(&context0);
    ScopedActivateContext scope4(&context0);

    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 1);
  }

  EXPECT_TRUE(context0.IsActive());
  EXPECT_EQ(context0.set_active_count(), 1);
}

TEST(ScopedActivateContextTest, Edge_RestoreOnlyOccursForDifferentNestedContext) {
  FakeContext context0(0);
  FakeContext context1(1);

  {
    ScopedActivateContext outer(&context0);

    {
      ScopedActivateContext same(&context0);
      EXPECT_TRUE(context0.IsActive());
    }

    EXPECT_EQ(context0.set_active_count(), 1);

    {
      ScopedActivateContext different(&context1);
      EXPECT_TRUE(context1.IsActive());
    }

    EXPECT_TRUE(context0.IsActive());
    EXPECT_EQ(context0.set_active_count(), 2);
  }
}

}  // namespace
}  // namespace stream_executor::gpu