#include "quiche/quic/core/quic_alarm.h"

#include <gtest/gtest.h>

#include <memory>

#include "quiche/quic/core/quic_connection_context.h"
#include "quiche/quic/core/quic_time.h"
#include "quiche/quic/platform/api/quic_test.h"

namespace quic {
namespace test {
namespace {

class TestDelegate : public QuicAlarm::Delegate {
 public:
  explicit TestDelegate(int* fire_count) : fire_count_(fire_count) {}

  void OnAlarm() override { ++(*fire_count_); }

  QuicConnectionContext* GetConnectionContext() override { return nullptr; }

 private:
  int* fire_count_;
};

class TestAlarm : public QuicAlarm {
 public:
  explicit TestAlarm(int* fire_count)
      : QuicAlarm(QuicArenaScopedPtr<Delegate>(
            new TestDelegate(fire_count))) {}

  int set_impl_count() const { return set_impl_count_; }
  int cancel_impl_count() const { return cancel_impl_count_; }
  int update_impl_count() const { return update_impl_count_; }

 protected:
  void SetImpl() override { ++set_impl_count_; }

  void CancelImpl() override { ++cancel_impl_count_; }

  void UpdateImpl() override {
    ++update_impl_count_;
    QuicAlarm::UpdateImpl();
  }

 private:
  int set_impl_count_ = 0;
  int cancel_impl_count_ = 0;
  int update_impl_count_ = 0;
};

class QuicAlarmTest : public QuicTest {
 protected:
  QuicTime TimeFromUs(int64_t us) {
    return QuicTime::Zero() + QuicTime::Delta::FromMicroseconds(us);
  }

  QuicTime::Delta DeltaFromUs(int64_t us) {
    return QuicTime::Delta::FromMicroseconds(us);
  }
};

TEST_F(QuicAlarmTest, NewAlarmIsInitiallyUnset) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_FALSE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(fire_count, 0);
  EXPECT_EQ(alarm.set_impl_count(), 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
}

TEST_F(QuicAlarmTest, SetValidDeadlineMarksAlarmSetAndCallsSetImpl) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.Set(TimeFromUs(1000));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1000));
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(fire_count, 0);
}

TEST_F(QuicAlarmTest, CancelUnsetAlarmDoesNotCallCancelImpl) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.Cancel();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_FALSE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.set_impl_count(), 0);
}

TEST_F(QuicAlarmTest, CancelSetAlarmClearsDeadlineAndCallsCancelImpl) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Cancel();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), QuicTime::Zero());
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_FALSE(alarm.IsPermanentlyCancelled());
}

TEST_F(QuicAlarmTest, PermanentCancelUnsetAlarmMarksAlarmPermanentlyCancelled) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.PermanentCancel();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_TRUE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(fire_count, 0);
}

TEST_F(QuicAlarmTest, PermanentCancelSetAlarmClearsDeadlineAndCallsCancelImpl) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.PermanentCancel();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_TRUE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.set_impl_count(), 1);
}

TEST_F(QuicAlarmTest, SetAfterPermanentCancelDoesNothing) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.PermanentCancel();

  alarm.Set(TimeFromUs(1000));

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_TRUE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(alarm.set_impl_count(), 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateAfterPermanentCancelDoesNothing) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.PermanentCancel();

  alarm.Update(TimeFromUs(1000), DeltaFromUs(10));

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_TRUE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(alarm.set_impl_count(), 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateUnsetAlarmWithInitializedDeadlineCallsSetImpl) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.Update(TimeFromUs(1000), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1000));
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateUnsetAlarmWithZeroDeadlineKeepsAlarmUnset) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.Update(QuicTime::Zero(), DeltaFromUs(10));

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(alarm.set_impl_count(), 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateSetAlarmWithZeroDeadlineCancelsAlarm) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(QuicTime::Zero(), DeltaFromUs(10));

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateWithinGranularityDoesNotChangeDeadline) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(1009), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1000));
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateAtGranularityBoundaryUpdatesDeadline) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(1010), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1010));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.update_impl_count(), 1);
}

TEST_F(QuicAlarmTest, UpdateBeyondGranularityUpdatesDeadline) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(1100), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1100));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.update_impl_count(), 1);
}

TEST_F(QuicAlarmTest, UpdateBackwardWithinGranularityDoesNotChangeDeadline) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(991), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1000));
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateBackwardAtGranularityBoundaryUpdatesDeadline) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(990), DeltaFromUs(10));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(990));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.update_impl_count(), 1);
}

TEST_F(QuicAlarmTest, UpdateWithZeroGranularitySameDeadlineDoesNotUpdate) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(1000), DeltaFromUs(0));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1000));
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
  EXPECT_EQ(alarm.update_impl_count(), 0);
}

TEST_F(QuicAlarmTest, UpdateWithZeroGranularityDifferentDeadlineUpdates) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Update(TimeFromUs(1001), DeltaFromUs(0));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(1001));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
  EXPECT_EQ(alarm.update_impl_count(), 1);
}

TEST_F(QuicAlarmTest, FireUnsetAlarmDoesNothing) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);

  alarm.Fire();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(fire_count, 0);
  EXPECT_EQ(alarm.set_impl_count(), 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
}

TEST_F(QuicAlarmTest, FireSetAlarmClearsDeadlineAndInvokesDelegateOnce) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Fire();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), QuicTime::Zero());
  EXPECT_EQ(fire_count, 1);
  EXPECT_EQ(alarm.set_impl_count(), 1);
  EXPECT_EQ(alarm.cancel_impl_count(), 0);
}

TEST_F(QuicAlarmTest, FireTwiceInvokesDelegateOnlyOnce) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));

  alarm.Fire();
  alarm.Fire();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(fire_count, 1);
}

TEST_F(QuicAlarmTest, FireAfterCancelDoesNotInvokeDelegate) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));
  alarm.Cancel();

  alarm.Fire();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_EQ(fire_count, 0);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
}

TEST_F(QuicAlarmTest, FireAfterPermanentCancelDoesNotInvokeDelegate) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));
  alarm.PermanentCancel();

  alarm.Fire();

  EXPECT_FALSE(alarm.IsSet());
  EXPECT_TRUE(alarm.IsPermanentlyCancelled());
  EXPECT_EQ(fire_count, 0);
}

TEST_F(QuicAlarmTest, CanSetAgainAfterFire) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));
  alarm.Fire();

  alarm.Set(TimeFromUs(2000));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(2000));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(fire_count, 1);
}

TEST_F(QuicAlarmTest, CanSetAgainAfterCancel) {
  int fire_count = 0;
  TestAlarm alarm(&fire_count);
  alarm.Set(TimeFromUs(1000));
  alarm.Cancel();

  alarm.Set(TimeFromUs(2000));

  EXPECT_TRUE(alarm.IsSet());
  EXPECT_EQ(alarm.deadline(), TimeFromUs(2000));
  EXPECT_EQ(alarm.set_impl_count(), 2);
  EXPECT_EQ(alarm.cancel_impl_count(), 1);
}

}  // namespace
}  // namespace test
}  // namespace quic