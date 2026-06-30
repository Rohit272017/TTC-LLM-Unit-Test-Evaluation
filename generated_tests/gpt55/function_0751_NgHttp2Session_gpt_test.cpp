#include "quiche/http2/adapter/nghttp2_session.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "nghttp2/nghttp2.h"
#include "quiche/http2/adapter/http2_protocol.h"

namespace http2 {
namespace adapter {
namespace {

class NgHttp2SessionTest : public ::testing::Test {
 protected:
  nghttp2_session_callbacks_unique_ptr MakeCallbacks() {
    nghttp2_session_callbacks* callbacks = nullptr;
    EXPECT_EQ(nghttp2_session_callbacks_new(&callbacks), 0);
    return nghttp2_session_callbacks_unique_ptr(callbacks);
  }
};

TEST_F(NgHttp2SessionTest, ClientSessionCanBeConstructed) {
  auto callbacks = MakeCallbacks();

  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_TRUE(session.want_read());
  EXPECT_FALSE(session.want_write());
  EXPECT_GT(session.GetRemoteWindowSize(), 0);
}

TEST_F(NgHttp2SessionTest, ServerSessionCanBeConstructed) {
  auto callbacks = MakeCallbacks();

  NgHttp2Session session(Perspective::kServer, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_TRUE(session.want_read());
  EXPECT_FALSE(session.want_write());
  EXPECT_GT(session.GetRemoteWindowSize(), 0);
}

TEST_F(NgHttp2SessionTest, ClientSessionCanBeConstructedWithOptions) {
  auto callbacks = MakeCallbacks();

  nghttp2_option* raw_options = nullptr;
  ASSERT_EQ(nghttp2_option_new(&raw_options), 0);
  nghttp2_option_set_no_auto_window_update(raw_options, 1);

  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         raw_options,
                         /*userdata=*/nullptr);

  nghttp2_option_del(raw_options);

  EXPECT_TRUE(session.want_read());
  EXPECT_FALSE(session.want_write());
}

TEST_F(NgHttp2SessionTest, ServerSessionCanBeConstructedWithUserData) {
  auto callbacks = MakeCallbacks();
  int userdata = 123;

  NgHttp2Session session(Perspective::kServer, std::move(callbacks),
                         /*options=*/nullptr,
                         &userdata);

  EXPECT_TRUE(session.want_read());
  EXPECT_FALSE(session.want_write());
}

TEST_F(NgHttp2SessionTest, ProcessEmptyBytesReturnsZero) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kServer, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_EQ(session.ProcessBytes(""), 0);
}

TEST_F(NgHttp2SessionTest, ClientProcessServerConnectionPrefaceReturnsInputSize) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  const std::string settings_frame("\x00\x00\x00\x04\x00\x00\x00\x00\x00", 9);

  EXPECT_EQ(session.ProcessBytes(settings_frame), settings_frame.size());
}

TEST_F(NgHttp2SessionTest, ServerProcessClientConnectionPrefaceReturnsInputSize) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kServer, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  const std::string client_preface =
      "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
  const std::string settings_frame("\x00\x00\x00\x04\x00\x00\x00\x00\x00", 9);
  const std::string input = client_preface + settings_frame;

  EXPECT_EQ(session.ProcessBytes(input), input.size());
}

TEST_F(NgHttp2SessionTest, ProcessInvalidBytesReturnsNegativeError) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kServer, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  const std::string invalid_bytes("invalid data");

  EXPECT_LT(session.ProcessBytes(invalid_bytes), 0);
}

TEST_F(NgHttp2SessionTest, ProcessPartialFrameHeaderReturnsInputSize) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  const std::string partial_header("\x00\x00", 2);

  EXPECT_EQ(session.ProcessBytes(partial_header), partial_header.size());
}

TEST_F(NgHttp2SessionTest, ProcessFrameInMultipleChunks) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  const std::string chunk1("\x00\x00", 2);
  const std::string chunk2("\x00\x04\x00\x00\x00\x00\x00", 7);

  EXPECT_EQ(session.ProcessBytes(chunk1), chunk1.size());
  EXPECT_EQ(session.ProcessBytes(chunk2), chunk2.size());
}

TEST_F(NgHttp2SessionTest, WantReadInitiallyTrue) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_TRUE(session.want_read());
}

TEST_F(NgHttp2SessionTest, WantWriteInitiallyFalse) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_FALSE(session.want_write());
}

TEST_F(NgHttp2SessionTest, RemoteWindowSizeInitiallyPositive) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_GT(session.GetRemoteWindowSize(), 0);
}

TEST_F(NgHttp2SessionTest, ConsumeZeroBytesReturnsSuccessOrNoOp) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_LE(session.Consume(/*stream_id=*/0, /*num_bytes=*/0), 0);
}

TEST_F(NgHttp2SessionTest, ConsumeOnInvalidStreamReturnsNegativeError) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_LT(session.Consume(/*stream_id=*/1, /*num_bytes=*/1), 0);
}

TEST_F(NgHttp2SessionTest, ConsumeLargeNumberOfBytesOnInvalidStreamReturnsNegativeError) {
  auto callbacks = MakeCallbacks();
  NgHttp2Session session(Perspective::kClient, std::move(callbacks),
                         /*options=*/nullptr,
                         /*userdata=*/nullptr);

  EXPECT_LT(session.Consume(/*stream_id=*/1,
                            /*num_bytes=*/static_cast<size_t>(UINT32_MAX)),
            0);
}

}  // namespace
}  // namespace adapter
}  // namespace http2