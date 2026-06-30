#include "quiche/quic/moqt/moqt_subscribe_windows.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "quiche/quic/moqt/moqt_messages.h"
#include "quiche/web_transport/web_transport.h"

namespace moqt {
namespace {

using ::webtransport::StreamId;

TEST(SubscribeWindowTest, ECP_InWindowReturnsTrueForStartBoundary) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_TRUE(window.InWindow(FullSequence(10, 20)));
}

TEST(SubscribeWindowTest, ECP_InWindowReturnsTrueForEndBoundary) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_TRUE(window.InWindow(FullSequence(30, 40)));
}

TEST(SubscribeWindowTest, ECP_InWindowReturnsTrueForMiddleSequence) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_TRUE(window.InWindow(FullSequence(20, 30)));
}

TEST(SubscribeWindowTest, BVA_InWindowReturnsFalseJustBeforeStartObject) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_FALSE(window.InWindow(FullSequence(10, 19)));
}

TEST(SubscribeWindowTest, BVA_InWindowReturnsFalseJustBeforeStartGroup) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(30, 40));

  EXPECT_FALSE(window.InWindow(FullSequence(9, 999)));
}

TEST(SubscribeWindowTest, BVA_InWindowReturnsFalseJustAfterEndObject) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_FALSE(window.InWindow(FullSequence(30, 41)));
}

TEST(SubscribeWindowTest, BVA_InWindowReturnsFalseJustAfterEndGroup) {
  SubscribeWindow window(FullSequence(10, 20), FullSequence(30, 40));

  EXPECT_FALSE(window.InWindow(FullSequence(31, 0)));
}

TEST(SubscribeWindowTest, ECP_InWindowWithOpenEndedWindowAcceptsStart) {
  SubscribeWindow window(FullSequence(10, 20), std::nullopt);

  EXPECT_TRUE(window.InWindow(FullSequence(10, 20)));
}

TEST(SubscribeWindowTest, ECP_InWindowWithOpenEndedWindowAcceptsLargeFutureSequence) {
  SubscribeWindow window(FullSequence(10, 20), std::nullopt);

  EXPECT_TRUE(window.InWindow(FullSequence(999, 999)));
}

TEST(SubscribeWindowTest, Invalid_InWindowWithOpenEndedWindowRejectsBeforeStart) {
  SubscribeWindow window(FullSequence(10, 20), std::nullopt);

  EXPECT_FALSE(window.InWindow(FullSequence(10, 19)));
}

TEST(SubscribeWindowTest, BVA_InWindowSingleElementWindow) {
  SubscribeWindow window(FullSequence(5, 7), FullSequence(5, 7));

  EXPECT_TRUE(window.InWindow(FullSequence(5, 7)));
  EXPECT_FALSE(window.InWindow(FullSequence(5, 6)));
  EXPECT_FALSE(window.InWindow(FullSequence(5, 8)));
}

TEST(SubscribeWindowTest, BVA_InWindowZeroSequence) {
  SubscribeWindow window(FullSequence(0, 0), FullSequence(0, 0));

  EXPECT_TRUE(window.InWindow(FullSequence(0, 0)));
  EXPECT_FALSE(window.InWindow(FullSequence(0, 1)));
}

TEST(SubscribeWindowTest, ECP_UpdateStartEndNarrowsClosedWindow) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_TRUE(window.UpdateStartEnd(FullSequence(12, 0), FullSequence(18, 0)));

  EXPECT_FALSE(window.InWindow(FullSequence(11, 999)));
  EXPECT_TRUE(window.InWindow(FullSequence(12, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(18, 0)));
  EXPECT_FALSE(window.InWindow(FullSequence(18, 1)));
}

TEST(SubscribeWindowTest, ECP_UpdateStartEndCanKeepSameWindow) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_TRUE(window.UpdateStartEnd(FullSequence(10, 0), FullSequence(20, 0)));

  EXPECT_TRUE(window.InWindow(FullSequence(10, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(20, 0)));
}

TEST(SubscribeWindowTest, BVA_UpdateStartEndToSingleElementWindow) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_TRUE(window.UpdateStartEnd(FullSequence(15, 5), FullSequence(15, 5)));

  EXPECT_TRUE(window.InWindow(FullSequence(15, 5)));
  EXPECT_FALSE(window.InWindow(FullSequence(15, 4)));
  EXPECT_FALSE(window.InWindow(FullSequence(15, 6)));
}

TEST(SubscribeWindowTest, Invalid_UpdateStartEndRejectsStartBeforeCurrentStart) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_FALSE(window.UpdateStartEnd(FullSequence(9, 999), FullSequence(18, 0)));

  EXPECT_TRUE(window.InWindow(FullSequence(10, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(20, 0)));
}

TEST(SubscribeWindowTest, Invalid_UpdateStartEndRejectsStartAfterCurrentEnd) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_FALSE(window.UpdateStartEnd(FullSequence(20, 1), FullSequence(21, 0)));

  EXPECT_TRUE(window.InWindow(FullSequence(10, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(20, 0)));
}

TEST(SubscribeWindowTest, Invalid_UpdateStartEndRejectsRemovingEndFromClosedWindow) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_FALSE(window.UpdateStartEnd(FullSequence(12, 0), std::nullopt));

  EXPECT_TRUE(window.InWindow(FullSequence(20, 0)));
  EXPECT_FALSE(window.InWindow(FullSequence(20, 1)));
}

TEST(SubscribeWindowTest, Invalid_UpdateStartEndRejectsExpandingClosedEnd) {
  SubscribeWindow window(FullSequence(10, 0), FullSequence(20, 0));

  EXPECT_FALSE(window.UpdateStartEnd(FullSequence(12, 0), FullSequence(21, 0)));

  EXPECT_TRUE(window.InWindow(FullSequence(20, 0)));
  EXPECT_FALSE(window.InWindow(FullSequence(21, 0)));
}

TEST(SubscribeWindowTest, ECP_UpdateStartEndOpenWindowCanRemainOpen) {
  SubscribeWindow window(FullSequence(10, 0), std::nullopt);

  EXPECT_TRUE(window.UpdateStartEnd(FullSequence(15, 0), std::nullopt));

  EXPECT_FALSE(window.InWindow(FullSequence(14, 999)));
  EXPECT_TRUE(window.InWindow(FullSequence(15, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(999, 999)));
}

TEST(SubscribeWindowTest, ECP_UpdateStartEndOpenWindowCanBecomeClosed) {
  SubscribeWindow window(FullSequence(10, 0), std::nullopt);

  EXPECT_TRUE(window.UpdateStartEnd(FullSequence(15, 0), FullSequence(30, 0)));

  EXPECT_TRUE(window.InWindow(FullSequence(15, 0)));
  EXPECT_TRUE(window.InWindow(FullSequence(30, 0)));
  EXPECT_FALSE(window.InWindow(FullSequence(30, 1)));
}

TEST(SendStreamMapTest, ECP_TrackPreferenceMapsAllSequencesToSameStream) {
  SendStreamMap map(MoqtForwardingPreference::kTrack);

  map.AddStream(FullSequence(1, 1), StreamId(100));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(1, 1)), StreamId(100));
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(99, 99)), StreamId(100));
}

TEST(SendStreamMapTest, ECP_SubgroupPreferenceMapsSameGroupToSameStream) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(5, 1), StreamId(200));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(5, 1)), StreamId(200));
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(5, 999)), StreamId(200));
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(4, 999)), std::nullopt);
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(6, 0)), std::nullopt);
}

TEST(SendStreamMapTest, ECP_DatagramPreferenceDoesNotAddStream) {
  SendStreamMap map(MoqtForwardingPreference::kDatagram);

  map.AddStream(FullSequence(1, 1), StreamId(300));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(1, 1)), std::nullopt);
  EXPECT_TRUE(map.GetAllStreams().empty());
}

TEST(SendStreamMapTest, ECP_GetStreamForMissingSequenceReturnsNullopt) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(1, 1)), std::nullopt);
}

TEST(SendStreamMapTest, ECP_RemoveStreamRemovesExistingTrackStream) {
  SendStreamMap map(MoqtForwardingPreference::kTrack);

  map.AddStream(FullSequence(1, 1), StreamId(100));
  ASSERT_EQ(map.GetStreamForSequence(FullSequence(99, 99)), StreamId(100));

  map.RemoveStream(FullSequence(2, 2), StreamId(100));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(1, 1)), std::nullopt);
  EXPECT_TRUE(map.GetAllStreams().empty());
}

TEST(SendStreamMapTest, ECP_RemoveStreamRemovesExistingSubgroupStream) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(10, 1), StreamId(400));
  ASSERT_EQ(map.GetStreamForSequence(FullSequence(10, 99)), StreamId(400));

  map.RemoveStream(FullSequence(10, 50), StreamId(400));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(10, 1)), std::nullopt);
  EXPECT_TRUE(map.GetAllStreams().empty());
}

TEST(SendStreamMapTest, ECP_GetAllStreamsReturnsAllStoredStreams) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(1, 0), StreamId(10));
  map.AddStream(FullSequence(2, 0), StreamId(20));
  map.AddStream(FullSequence(3, 0), StreamId(30));

  std::vector<StreamId> streams = map.GetAllStreams();
  std::sort(streams.begin(), streams.end());

  ASSERT_EQ(streams.size(), 3u);
  EXPECT_EQ(streams[0], StreamId(10));
  EXPECT_EQ(streams[1], StreamId(20));
  EXPECT_EQ(streams[2], StreamId(30));
}

TEST(SendStreamMapTest, BVA_GetAllStreamsOnEmptyMapReturnsEmptyVector) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  EXPECT_TRUE(map.GetAllStreams().empty());
}

TEST(SendStreamMapTest, BVA_ZeroSequenceAndZeroStreamIdAreSupported) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(0, 0), StreamId(0));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(0, 0)), StreamId(0));
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(0, 999)), StreamId(0));
}

TEST(SendStreamMapTest, BVA_LargeSequenceAndStreamIdAreSupported) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(UINT64_MAX, UINT64_MAX), StreamId(UINT64_MAX));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(UINT64_MAX, 0)),
            StreamId(UINT64_MAX));
}

TEST(SendStreamMapTest, Edge_SubgroupPreferenceDifferentObjectsDoNotCreateDifferentEntries) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(7, 1), StreamId(70));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(7, 2)), StreamId(70));
  EXPECT_EQ(map.GetAllStreams().size(), 1u);
}

TEST(SendStreamMapTest, Edge_TrackPreferenceDifferentGroupsDoNotCreateDifferentEntries) {
  SendStreamMap map(MoqtForwardingPreference::kTrack);

  map.AddStream(FullSequence(7, 1), StreamId(70));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(8, 2)), StreamId(70));
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(0, 0)), StreamId(70));
  EXPECT_EQ(map.GetAllStreams().size(), 1u);
}

TEST(SendStreamMapTest, Edge_RemoveOneSubgroupDoesNotRemoveOtherSubgroups) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(1, 0), StreamId(10));
  map.AddStream(FullSequence(2, 0), StreamId(20));

  map.RemoveStream(FullSequence(1, 999), StreamId(10));

  EXPECT_EQ(map.GetStreamForSequence(FullSequence(1, 0)), std::nullopt);
  EXPECT_EQ(map.GetStreamForSequence(FullSequence(2, 999)), StreamId(20));
}

#ifndef NDEBUG

TEST(SendStreamMapDeathTest, Invalid_RemoveMissingStreamDiesInDebug) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  EXPECT_DEATH(map.RemoveStream(FullSequence(1, 0), StreamId(10)), ".*");
}

TEST(SendStreamMapDeathTest, Invalid_RemoveStreamWithWrongStreamIdDiesInDebug) {
  SendStreamMap map(MoqtForwardingPreference::kSubgroup);

  map.AddStream(FullSequence(1, 0), StreamId(10));

  EXPECT_DEATH(map.RemoveStream(FullSequence(1, 0), StreamId(20)), ".*");
}

#endif

}  // namespace
}  // namespace moqt