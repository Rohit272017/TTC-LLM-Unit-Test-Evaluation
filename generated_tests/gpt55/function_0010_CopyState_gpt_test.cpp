#include "leveldb/status.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

namespace leveldb {
namespace {

TEST(StatusTest, OkStatusReturnsOkString) {
  Status status = Status::OK();

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.ToString(), "OK");
}

TEST(StatusTest, NotFoundWithOnlyPrimaryMessage) {
  Status status = Status::NotFound("key");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsNotFound());
  EXPECT_EQ(status.ToString(), "NotFound: key");
}

TEST(StatusTest, NotFoundWithPrimaryAndSecondaryMessage) {
  Status status = Status::NotFound("key", "missing");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsNotFound());
  EXPECT_EQ(status.ToString(), "NotFound: key: missing");
}

TEST(StatusTest, CorruptionWithPrimaryAndSecondaryMessage) {
  Status status = Status::Corruption("block", "bad checksum");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsCorruption());
  EXPECT_EQ(status.ToString(), "Corruption: block: bad checksum");
}

TEST(StatusTest, NotSupportedWithMessage) {
  Status status = Status::NotSupported("compression");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsNotSupportedError());
  EXPECT_EQ(status.ToString(), "Not implemented: compression");
}

TEST(StatusTest, InvalidArgumentWithMessage) {
  Status status = Status::InvalidArgument("bad option");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_EQ(status.ToString(), "Invalid argument: bad option");
}

TEST(StatusTest, IOErrorWithMessage) {
  Status status = Status::IOError("disk failure");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsIOError());
  EXPECT_EQ(status.ToString(), "IO error: disk failure");
}

TEST(StatusTest, EmptyPrimaryMessage) {
  Status status = Status::InvalidArgument("");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_EQ(status.ToString(), "Invalid argument: ");
}

TEST(StatusTest, EmptySecondaryMessageDoesNotAppendSeparator) {
  Status status = Status::InvalidArgument("bad option", "");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_EQ(status.ToString(), "Invalid argument: bad option");
}

TEST(StatusTest, EmptyPrimaryAndNonEmptySecondaryMessageAddsSeparator) {
  Status status = Status::Corruption("", "metadata error");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsCorruption());
  EXPECT_EQ(status.ToString(), "Corruption: : metadata error");
}

TEST(StatusTest, OneCharacterMessages) {
  Status status = Status::IOError("a", "b");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsIOError());
  EXPECT_EQ(status.ToString(), "IO error: a: b");
}

TEST(StatusTest, LongPrimaryAndSecondaryMessages) {
  const std::string msg1(1024, 'x');
  const std::string msg2(2048, 'y');

  Status status = Status::NotFound(Slice(msg1), Slice(msg2));

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsNotFound());
  EXPECT_EQ(status.ToString(), "NotFound: " + msg1 + ": " + msg2);
}

TEST(StatusTest, EmbeddedNullCharactersArePreserved) {
  const std::string msg1("abc\0def", 7);
  const std::string msg2("xy\0z", 4);

  Status status = Status::InvalidArgument(Slice(msg1), Slice(msg2));

  const std::string expected =
      std::string("Invalid argument: ") + msg1 + ": " + msg2;

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_EQ(status.ToString().size(), expected.size());
  EXPECT_EQ(status.ToString(), expected);
}

TEST(StatusTest, CopyConstructorCopiesState) {
  Status original = Status::Corruption("block", "bad crc");
  Status copied(original);

  EXPECT_FALSE(copied.ok());
  EXPECT_TRUE(copied.IsCorruption());
  EXPECT_EQ(copied.ToString(), original.ToString());
  EXPECT_EQ(copied.ToString(), "Corruption: block: bad crc");
}

TEST(StatusTest, AssignmentOperatorCopiesState) {
  Status original = Status::NotFound("file", "missing");
  Status assigned = Status::OK();

  assigned = original;

  EXPECT_FALSE(assigned.ok());
  EXPECT_TRUE(assigned.IsNotFound());
  EXPECT_EQ(assigned.ToString(), original.ToString());
  EXPECT_EQ(assigned.ToString(), "NotFound: file: missing");
}

TEST(StatusTest, AssignmentFromOkClearsState) {
  Status status = Status::IOError("disk");

  status = Status::OK();

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.ToString(), "OK");
}

TEST(StatusTest, AssignmentFromNonOkToOkObjectCopiesState) {
  Status status = Status::OK();

  status = Status::InvalidArgument("bad input");

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_EQ(status.ToString(), "Invalid argument: bad input");
}

TEST(StatusTest, SelfAssignmentKeepsStateValid) {
  Status status = Status::NotSupported("feature");

  status = status;

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsNotSupportedError());
  EXPECT_EQ(status.ToString(), "Not implemented: feature");
}

}  // namespace
}  // namespace leveldb