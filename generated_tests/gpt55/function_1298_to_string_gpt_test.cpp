#include "tensorstore/open_mode.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "absl/status/status.h"

namespace tensorstore {
namespace {

TEST(ReadWriteModeToStringTest, KnownModesReturnExpectedStrings) {
  EXPECT_EQ(to_string(ReadWriteMode::dynamic), "dynamic");
  EXPECT_EQ(to_string(ReadWriteMode::read), "read");
  EXPECT_EQ(to_string(ReadWriteMode::write), "write");
  EXPECT_EQ(to_string(ReadWriteMode::read_write), "read_write");
}

TEST(ReadWriteModeToStringTest, UnknownModeReturnsUnknownString) {
  auto mode = static_cast<ReadWriteMode>(12345);

  EXPECT_EQ(to_string(mode), "<unknown>");
}

TEST(ReadWriteModeStreamTest, StreamsKnownReadWriteModes) {
  std::ostringstream os;

  os << ReadWriteMode::dynamic << " "
     << ReadWriteMode::read << " "
     << ReadWriteMode::write << " "
     << ReadWriteMode::read_write;

  EXPECT_EQ(os.str(), "dynamic read write read_write");
}

TEST(ReadWriteModeStreamTest, StreamsUnknownReadWriteMode) {
  std::ostringstream os;

  os << static_cast<ReadWriteMode>(12345);

  EXPECT_EQ(os.str(), "<unknown>");
}

TEST(OpenModeStreamTest, EmptyOpenModeStreamsEmptyString) {
  std::ostringstream os;

  os << OpenMode{};

  EXPECT_EQ(os.str(), "");
}

TEST(OpenModeStreamTest, StreamsOpenOnly) {
  std::ostringstream os;

  os << OpenMode::open;

  EXPECT_EQ(os.str(), "open");
}

TEST(OpenModeStreamTest, StreamsCreateOnly) {
  std::ostringstream os;

  os << OpenMode::create;

  EXPECT_EQ(os.str(), "create");
}

TEST(OpenModeStreamTest, StreamsDeleteExistingOnly) {
  std::ostringstream os;

  os << OpenMode::delete_existing;

  EXPECT_EQ(os.str(), "delete_existing");
}

TEST(OpenModeStreamTest, StreamsAssumeMetadataOnly) {
  std::ostringstream os;

  os << OpenMode::assume_metadata;

  EXPECT_EQ(os.str(), "assume_metadata");
}

TEST(OpenModeStreamTest, StreamsOpenAndCreateInDefinedOrder) {
  std::ostringstream os;

  os << (OpenMode::open | OpenMode::create);

  EXPECT_EQ(os.str(), "open|create");
}

TEST(OpenModeStreamTest, StreamsAllFlagsInDefinedOrder) {
  std::ostringstream os;

  os << (OpenMode::open | OpenMode::create |
         OpenMode::delete_existing | OpenMode::assume_metadata);

  EXPECT_EQ(os.str(), "open|create|delete_existing|assume_metadata");
}

TEST(OpenModeStreamTest, StreamsNonAdjacentFlagsWithSeparator) {
  std::ostringstream os;

  os << (OpenMode::open | OpenMode::assume_metadata);

  EXPECT_EQ(os.str(), "open|assume_metadata");
}

TEST(OpenModeStreamTest, IgnoresUnknownOpenModeBitsWhenKnownBitsAbsent) {
  std::ostringstream os;

  os << static_cast<OpenMode>(0x1000);

  EXPECT_EQ(os.str(), "");
}

TEST(OpenModeStreamTest, IgnoresUnknownOpenModeBitsWhenKnownBitsPresent) {
  std::ostringstream os;

  os << (OpenMode::open | static_cast<OpenMode>(0x1000));

  EXPECT_EQ(os.str(), "open");
}

TEST(ValidateSupportsReadTest, ReadModeIsSupported) {
  absl::Status status =
      internal::ValidateSupportsRead(ReadWriteMode::read);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsReadTest, ReadWriteModeSupportsRead) {
  absl::Status status =
      internal::ValidateSupportsRead(ReadWriteMode::read_write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsReadTest, DynamicModeDoesNotSupportRead) {
  absl::Status status =
      internal::ValidateSupportsRead(ReadWriteMode::dynamic);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Source does not support reading.");
}

TEST(ValidateSupportsReadTest, WriteOnlyModeDoesNotSupportRead) {
  absl::Status status =
      internal::ValidateSupportsRead(ReadWriteMode::write);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Source does not support reading.");
}

TEST(ValidateSupportsWriteTest, WriteModeIsSupported) {
  absl::Status status =
      internal::ValidateSupportsWrite(ReadWriteMode::write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsWriteTest, ReadWriteModeSupportsWrite) {
  absl::Status status =
      internal::ValidateSupportsWrite(ReadWriteMode::read_write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsWriteTest, DynamicModeDoesNotSupportWrite) {
  absl::Status status =
      internal::ValidateSupportsWrite(ReadWriteMode::dynamic);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Destination does not support writing.");
}

TEST(ValidateSupportsWriteTest, ReadOnlyModeDoesNotSupportWrite) {
  absl::Status status =
      internal::ValidateSupportsWrite(ReadWriteMode::read);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Destination does not support writing.");
}

TEST(ValidateSupportsModesTest, NoRequiredModesAlwaysSucceeds) {
  EXPECT_TRUE(internal::ValidateSupportsModes(ReadWriteMode::dynamic,
                                             ReadWriteMode::dynamic)
                  .ok());
  EXPECT_TRUE(internal::ValidateSupportsModes(ReadWriteMode::read,
                                             ReadWriteMode::dynamic)
                  .ok());
}

TEST(ValidateSupportsModesTest, ReadRequirementSatisfiedByReadMode) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::read,
                                      ReadWriteMode::read);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsModesTest, WriteRequirementSatisfiedByWriteMode) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::write,
                                      ReadWriteMode::write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsModesTest, ReadWriteRequirementSatisfiedByReadWriteMode) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::read_write,
                                      ReadWriteMode::read_write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsModesTest, ReadWriteModeSatisfiesReadOnlyRequirement) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::read_write,
                                      ReadWriteMode::read);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsModesTest, ReadWriteModeSatisfiesWriteOnlyRequirement) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::read_write,
                                      ReadWriteMode::write);

  EXPECT_TRUE(status.ok());
}

TEST(ValidateSupportsModesTest, MissingReadReturnsReadModeNotSupported) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::write,
                                      ReadWriteMode::read);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Read mode not supported");
}

TEST(ValidateSupportsModesTest, MissingWriteReturnsWriteModeNotSupported) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::read,
                                      ReadWriteMode::write);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Write mode not supported");
}

TEST(ValidateSupportsModesTest, DynamicModeMissingReadWriteReportsReadFirst) {
  absl::Status status =
      internal::ValidateSupportsModes(ReadWriteMode::dynamic,
                                      ReadWriteMode::read_write);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Read mode not supported");
}

}  // namespace
}  // namespace tensorstore