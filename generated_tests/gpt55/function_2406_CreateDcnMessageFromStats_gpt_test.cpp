#include "tensorflow/core/profiler/convert/dcn_utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xla/tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::CreateTfXPlaneVisitor;

class DcnUtilsTest : public ::testing::Test {
 protected:
  XEventVisitor CreateEventVisitor(
      const std::string& event_name,
      int64_t timestamp_ns = 1000000,
      int64_t duration_us = 10,
      bool add_required_stats = true) {
    plane_ = XPlane();
    XPlaneBuilder plane_builder(&plane_);
    XLineBuilder line = plane_builder.GetOrCreateLine(0);

    XEventMetadata* event_metadata =
        plane_builder.GetOrCreateEventMetadata(event_name);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetTimestampNs(timestamp_ns);
    event.SetDurationNs(duration_us * 1000);

    if (add_required_stats) {
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kDcnLabel)),
          "collective_a");
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kDcnSourceSliceId)),
          static_cast<int64_t>(1));
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kDcnSourcePerSliceDeviceId)),
          static_cast<int64_t>(2));
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kDcnDestinationSliceId)),
          static_cast<int64_t>(3));
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kDcnDestinationPerSliceDeviceId)),
          static_cast<int64_t>(4));
      event.AddStatValue(
          *plane_builder.GetOrCreateStatMetadata(
              static_cast<int64_t>(StatType::kPayloadSizeBytes)),
          static_cast<int64_t>(1024));
    }

    event.AddStatValue(
        *plane_builder.GetOrCreateStatMetadata(
            static_cast<int64_t>(StatType::kDuration)),
        duration_us);

    visitor_ = CreateTfXPlaneVisitor(&plane_);
    line_visitor_ = visitor_->GetLine(0);
    event_visitor_ = line_visitor_->GetEvent(0);
    return *event_visitor_;
  }

  void AddStat(XPlaneBuilder& plane_builder,
               XEventBuilder& event,
               StatType stat_type,
               int64_t value) {
    event.AddStatValue(
        *plane_builder.GetOrCreateStatMetadata(static_cast<int64_t>(stat_type)),
        value);
  }

  void AddStat(XPlaneBuilder& plane_builder,
               XEventBuilder& event,
               StatType stat_type,
               const std::string& value) {
    event.AddStatValue(
        *plane_builder.GetOrCreateStatMetadata(static_cast<int64_t>(stat_type)),
        value);
  }

  XPlane plane_;
  std::unique_ptr<tsl::profiler::XPlaneVisitor> visitor_;
  std::optional<tsl::profiler::XLineVisitor> line_visitor_;
  std::optional<XEventVisitor> event_visitor_;
};

TEST_F(DcnUtilsTest, ECP_IsDcnEventReturnsTrueForMegaScalePrefix) {
  XEventVisitor event = CreateEventVisitor("MegaScale:Send");

  EXPECT_TRUE(IsDcnEvent(event));
}

TEST_F(DcnUtilsTest, BVA_IsDcnEventReturnsTrueForExactPrefixOnly) {
  XEventVisitor event = CreateEventVisitor("MegaScale:");

  EXPECT_TRUE(IsDcnEvent(event));
}

TEST_F(DcnUtilsTest, Invalid_IsDcnEventReturnsFalseForMissingPrefix) {
  XEventVisitor event = CreateEventVisitor("Send");

  EXPECT_FALSE(IsDcnEvent(event));
}

TEST_F(DcnUtilsTest, Edge_IsDcnEventIsCaseSensitive) {
  XEventVisitor event = CreateEventVisitor("megascale:Send");

  EXPECT_FALSE(IsDcnEvent(event));
}

TEST_F(DcnUtilsTest, Edge_IsDcnEventDoesNotTrimWhitespace) {
  XEventVisitor event = CreateEventVisitor(" MegaScale:Send");

  EXPECT_FALSE(IsDcnEvent(event));
}

TEST_F(DcnUtilsTest, ECP_ValidDcnMessageParsesAllRequiredFields) {
  XEventVisitor event = CreateEventVisitor("MegaScale:Send", 1000000, 10);

  DcnMessage message = GetDcnMessageFromXEvent(event);

  EXPECT_EQ(message.collective_name, "collective_a");
  EXPECT_EQ(message.slice_src, 1);
  EXPECT_EQ(message.tpu_src, 2);
  EXPECT_EQ(message.slice_dst, 3);
  EXPECT_EQ(message.tpu_dst, 4);
  EXPECT_EQ(message.size_bytes, 1024);
  EXPECT_EQ(message.duration_us, 10);
  EXPECT_EQ(message.start_timestamp_ns, 990000);
  EXPECT_EQ(message.end_timestamp_ns, 1000000);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
}

TEST_F(DcnUtilsTest, BVA_DurationOneMicrosecondComputesTimestampWindow) {
  XEventVisitor event = CreateEventVisitor("MegaScale:Send", 5000, 1);

  DcnMessage message = GetDcnMessageFromXEvent(event);

  EXPECT_EQ(message.duration_us, 1);
  EXPECT_EQ(message.start_timestamp_ns, 4000);
  EXPECT_EQ(message.end_timestamp_ns, 5000);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
}

TEST_F(DcnUtilsTest, BVA_ZeroDurationMarksClockSkew) {
  XEventVisitor event = CreateEventVisitor("MegaScale:Send", 5000, 0);

  DcnMessage message = GetDcnMessageFromXEvent(event);

  EXPECT_EQ(message.duration_us, 0);
  EXPECT_EQ(message.start_timestamp_ns, 5000);
  EXPECT_EQ(message.end_timestamp_ns, 5000);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_CLOCK_SKEW);
}

TEST_F(DcnUtilsTest, ECP_SameSourceAndDestinationSliceMarksLoopback) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);
  event.SetDurationNs(10000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "loopback_collective");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 7);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 1);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 7);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 128);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.slice_src, 7);
  EXPECT_EQ(message.slice_dst, 7);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID_LOOPBACK);
}

TEST_F(DcnUtilsTest, Invalid_MissingCollectiveNameMarksBadKey) {
  XEventVisitor event = CreateEventVisitor("MegaScale:Send", 1000000, 10,
                                           false);

  DcnMessage message = GetDcnMessageFromXEvent(event);

  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, Invalid_MissingSourceSliceMarksBadKey) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.slice_src, -1);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, Invalid_MissingSourceDeviceMarksBadKey) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.tpu_src, -1);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, Invalid_MissingDestinationSliceMarksBadKey) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.slice_dst, -1);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, Invalid_MissingDestinationDeviceMarksBadKey) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.tpu_dst, -1);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, Invalid_MissingPayloadSizeMarksBadKey) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.size_bytes, -1);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_INVALID_BAD_KEY);
}

TEST_F(DcnUtilsTest, BVA_ZeroPayloadSizeIsValid) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 0);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.size_bytes, 0);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
}

TEST_F(DcnUtilsTest, ECP_OptionalChunkAndLoopIndexAreParsed) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDcnChunk, 9);
  AddStat(plane_builder, event, StatType::kDcnLoopIndex, 11);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.chunk_id, 9);
  EXPECT_EQ(message.loop_index_id, 11);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
}

TEST_F(DcnUtilsTest, Edge_UnknownStatsAreIgnored) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kKernelDetails, "ignored");
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
  EXPECT_EQ(message.collective_name, "collective_a");
  EXPECT_EQ(message.size_bytes, 1024);
}

TEST_F(DcnUtilsTest, Edge_StatsWithoutTypeAreIgnored) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  event.AddStatValue(*plane_builder.GetOrCreateStatMetadata("no_type_stat"),
                     static_cast<int64_t>(999));

  AddStat(plane_builder, event, StatType::kDcnLabel, "collective_a");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
  EXPECT_EQ(message.collective_name, "collective_a");
}

TEST_F(DcnUtilsTest, Edge_DuplicateStatsLastValueWins) {
  plane_ = XPlane();
  XPlaneBuilder plane_builder(&plane_);
  XLineBuilder line = plane_builder.GetOrCreateLine(0);

  XEventBuilder event =
      line.AddEvent(*plane_builder.GetOrCreateEventMetadata("MegaScale:Send"));
  event.SetTimestampNs(1000000);

  AddStat(plane_builder, event, StatType::kDcnLabel, "first");
  AddStat(plane_builder, event, StatType::kDcnLabel, "second");
  AddStat(plane_builder, event, StatType::kDcnSourceSliceId, 1);
  AddStat(plane_builder, event, StatType::kDcnSourcePerSliceDeviceId, 2);
  AddStat(plane_builder, event, StatType::kDcnDestinationSliceId, 3);
  AddStat(plane_builder, event, StatType::kDcnDestinationPerSliceDeviceId, 4);
  AddStat(plane_builder, event, StatType::kPayloadSizeBytes, 1024);
  AddStat(plane_builder, event, StatType::kDuration, 5);
  AddStat(plane_builder, event, StatType::kDuration, 10);

  visitor_ = CreateTfXPlaneVisitor(&plane_);
  line_visitor_ = visitor_->GetLine(0);
  event_visitor_ = line_visitor_->GetEvent(0);

  DcnMessage message = GetDcnMessageFromXEvent(*event_visitor_);

  EXPECT_EQ(message.collective_name, "second");
  EXPECT_EQ(message.duration_us, 10);
  EXPECT_EQ(message.start_timestamp_ns, 990000);
  EXPECT_EQ(message.validity_info, DCN_MESSAGE_VALID);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow