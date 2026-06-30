#include "quiche/quic/tools/quic_simple_server_stream.h"
#include <algorithm>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "quiche/http2/core/spdy_protocol.h"
#include "quiche/quic/core/http/quic_spdy_stream.h"
#include "quiche/quic/core/http/spdy_utils.h"
#include "quiche/quic/core/http/web_transport_http3.h"
#include "quiche/quic/core/quic_error_codes.h"
#include "quiche/quic/core/quic_utils.h"
#include "quiche/quic/platform/api/quic_bug_tracker.h"
#include "quiche/quic/platform/api/quic_flags.h"
#include "quiche/quic/platform/api/quic_logging.h"
#include "quiche/quic/tools/quic_simple_server_session.h"
using quiche::HttpHeaderBlock;
namespace quic {
QuicSimpleServerStream::QuicSimpleServerStream(
    QuicStreamId id, QuicSpdySession* session, StreamType type,
    QuicSimpleServerBackend* quic_simple_server_backend)
    : QuicSpdyServerStreamBase(id, session, type),
      content_length_(-1),
      generate_bytes_length_(0),
      quic_simple_server_backend_(quic_simple_server_backend) {
  QUICHE_DCHECK(quic_simple_server_backend_);
}
QuicSimpleServerStream::QuicSimpleServerStream(
    PendingStream* pending, QuicSpdySession* session,
    QuicSimpleServerBackend* quic_simple_server_backend)
    : QuicSpdyServerStreamBase(pending, session),
      content_length_(-1),
      generate_bytes_length_(0),
      quic_simple_server_backend_(quic_simple_server_backend) {
  QUICHE_DCHECK(quic_simple_server_backend_);
}
QuicSimpleServerStream::~QuicSimpleServerStream() {
  quic_simple_server_backend_->CloseBackendResponseStream(this);
}
void QuicSimpleServerStream::OnInitialHeadersComplete(
    bool fin, size_t frame_len, const QuicHeaderList& header_list) {
  QuicSpdyStream::OnInitialHeadersComplete(fin, frame_len, header_list);
  if (!response_sent_ &&
      !SpdyUtils::CopyAndValidateHeaders(header_list, &content_length_,
                                         &request_headers_)) {
    QUIC_DVLOG(1) << "Invalid headers";
    SendErrorResponse();
  }
  ConsumeHeaderList();
  if (!fin && !response_sent_ && IsConnectRequest()) {
    if (quic_simple_server_backend_ == nullptr) {
      QUIC_DVLOG(1) << "Backend is missing on CONNECT headers.";
      SendErrorResponse();
      return;
    }
    if (web_transport() != nullptr) {
      QuicSimpleServerBackend::WebTransportResponse response =
          quic_simple_server_backend_->ProcessWebTransportRequest(
              request_headers_, web_transport());
      if (response.response_headers[":status"] == "200") {
        WriteHeaders(std::move(response.response_headers), false, nullptr);
        if (response.visitor != nullptr) {
          web_transport()->SetVisitor(std::move(response.visitor));
        }
        web_transport()->HeadersReceived(request_headers_);
      } else {
        WriteHeaders(std::move(response.response_headers), true, nullptr);
      }
      return;
    }
    quic_simple_server_backend_->HandleConnectHeaders(request_headers_,
                                                      this);
  }
}
void QuicSimpleServerStream::OnBodyAvailable() {
  while (HasBytesToRead()) {
    struct iovec iov;
    if (GetReadableRegions(&iov, 1) == 0) {
      break;
    }
    QUIC_DVLOG(1) << "Stream " << id() << " processed " << iov.iov_len
                  << " bytes.";
    body_.append(static_cast<char*>(iov.iov_base), iov.iov_len);
    if (content_length_ >= 0 &&
        body_.size() > static_cast<uint64_t>(content_length_)) {
      QUIC_DVLOG(1) << "Body size (" << body_.size() << ") > content length ("
                    << content_length_ << ").";
      SendErrorResponse();
      return;
    }
    MarkConsumed(iov.iov_len);
  }
  if (!sequencer()->IsClosed()) {
    if (IsConnectRequest()) {
      HandleRequestConnectData(false);
    }
    sequencer()->SetUnblocked();
    return;
  }
  OnFinRead();
  if (write_side_closed() || fin_buffered()) {
    return;
  }
  if (IsConnectRequest()) {
    HandleRequestConnectData(true);
  } else {
    SendResponse();
  }
}
void QuicSimpleServerStream::HandleRequestConnectData(bool fin_received) {
  QUICHE_DCHECK(IsConnectRequest());
  if (quic_simple_server_backend_ == nullptr) {
    QUIC_DVLOG(1) << "Backend is missing on CONNECT data.";
    ResetWriteSide(
        QuicResetStreamError::FromInternal(QUIC_STREAM_CONNECT_ERROR));
    return;
  }
  std::string data = std::move(body_);
  body_.clear();
  quic_simple_server_backend_->HandleConnectData(data,
                                                 fin_received,
                                                 this);
}
void QuicSimpleServerStream::SendResponse() {
  QUICHE_DCHECK(!IsConnectRequest());
  if (request_headers_.empty()) {
    QUIC_DVLOG(1) << "Request headers empty.";
    SendErrorResponse();
    return;
  }
  if (content_length_ > 0 &&
      static_cast<uint64_t>(content_length_) != body_.size()) {
    QUIC_DVLOG(1) << "Content length (" << content_length_ << ") != body size ("
                  << body_.size() << ").";
    SendErrorResponse();
    return;
  }
  if (!request_headers_.contains(":authority")) {
    QUIC_DVLOG(1) << "Request headers do not contain :authority.";
    SendErrorResponse();
    return;
  }
  if (!request_headers_.contains(":path")) {
    QUIC_DVLOG(1) << "Request headers do not contain :path.";
    SendErrorResponse();
    return;
  }
  if (quic_simple_server_backend_ == nullptr) {
    QUIC_DVLOG(1) << "Backend is missing in SendResponse().";
    SendErrorResponse();
    return;
  }
  if (web_transport() != nullptr) {
    QuicSimpleServerBackend::WebTransportResponse response =
        quic_simple_server_backend_->ProcessWebTransportRequest(
            request_headers_, web_transport());
    if (response.response_headers[":status"] == "200") {
      WriteHeaders(std::move(response.response_headers), false, nullptr);
      if (response.visitor != nullptr) {
        web_transport()->SetVisitor(std::move(response.visitor));
      }
      web_transport()->HeadersReceived(request_headers_);
    } else {
      WriteHeaders(std::move(response.response_headers), true, nullptr);
    }
    return;
  }
  quic_simple_server_backend_->FetchResponseFromBackend(request_headers_, body_,
                                                        this);
}
QuicConnectionId QuicSimpleServerStream::connection_id() const {
  return spdy_session()->connection_id();
}
QuicStreamId QuicSimpleServerStream::stream_id() const { return id(); }
std::string QuicSimpleServerStream::peer_host() const {
  return spdy_session()->peer_address().host().ToString();
}
QuicSpdyStream* QuicSimpleServerStream::GetStream() { return this; }
namespace {
class DelayedResponseAlarm : public QuicAlarm::DelegateWithContext {
 public:
  DelayedResponseAlarm(QuicSimpleServerStream* stream,
                       const QuicBackendResponse* response)
      : QuicAlarm::DelegateWithContext(
            stream->spdy_session()->connection()->context()),
        stream_(stream),
        response_(response) {
    stream_ = stream;
    response_ = response;
  }
  ~DelayedResponseAlarm() override = default;
  void OnAlarm() override { stream_->Respond(response_); }
 private:
  QuicSimpleServerStream* stream_;
  const QuicBackendResponse* response_;
};
}  
void QuicSimpleServerStream::OnResponseBackendComplete(
    const QuicBackendResponse* response) {
  if (response == nullptr) {
    QUIC_DVLOG(1) << "Response not found in cache.";
    SendNotFoundResponse();
    return;
  }
  auto delay = response->delay();
  if (delay.IsZero()) {
    Respond(response);
    return;
  }
  auto* connection = session()->connection();
  delayed_response_alarm_.reset(connection->alarm_factory()->CreateAlarm(
      new DelayedResponseAlarm(this, response)));
  delayed_response_alarm_->Set(connection->clock()->Now() + delay);
}
void QuicSimpleServerStream::Respond(const QuicBackendResponse* response) {
  for (const auto& headers : response->early_hints()) {
    QUIC_DVLOG(1) << "Stream " << id() << " sending an Early Hints response: "
                  << headers.DebugString();
    WriteHeaders(headers.Clone(), false, nullptr);
  }
  if (response->response_type() == QuicBackendResponse::CLOSE_CONNECTION) {
    QUIC_DVLOG(1) << "Special response: closing connection.";
    OnUnrecoverableError(QUIC_NO_ERROR, "Toy server forcing close");
    return;
  }
  if (response->response_type() == QuicBackendResponse::IGNORE_REQUEST) {
    QUIC_DVLOG(1) << "Special response: ignoring request.";
    return;
  }
  if (response->response_type() == QuicBackendResponse::BACKEND_ERR_RESPONSE) {
    QUIC_DVLOG(1) << "Quic Proxy: Backend connection error.";
    SendErrorResponse(502);
    return;
  }
  std::string request_url = request_headers_[":authority"].as_string() +
                            request_headers_[":path"].as_string();
  int response_code;
  const HttpHeaderBlock& response_headers = response->headers();
  if (!ParseHeaderStatusCode(response_headers, &response_code)) {
    auto status = response_headers.find(":status");
    if (status == response_headers.end()) {
      QUIC_LOG(WARNING)
          << ":status not present in response from cache for request "
          << request_url;
    } else {
      QUIC_LOG(WARNING) << "Illegal (non-integer) response :status from cache: "
                        << status->second << " for request " << request_url;
    }
    SendErrorResponse();
    return;
  }
  if (response->response_type() == QuicBackendResponse::INCOMPLETE_RESPONSE) {
    QUIC_DVLOG(1)
        << "Stream " << id()
        << " sending an incomplete response, i.e. no trailer, no fin.";
    SendIncompleteResponse(response->headers().Clone(), response->body());
    return;
  }
  if (response->response_type() == QuicBackendResponse::GENERATE_BYTES) {
    QUIC_DVLOG(1) << "Stream " << id() << " sending a generate bytes response.";
    std::string path = request_headers_[":path"].as_string().substr(1);
    if (!absl::SimpleAtoi(path, &generate_bytes_length_)) {
      QUIC_LOG(ERROR) << "Path is not a number.";
      SendNotFoundResponse();
      return;
    }
    HttpHeaderBlock headers = response->headers().Clone();
    headers["content-length"] = absl::StrCat(generate_bytes_length_);
    WriteHeaders(std::move(headers), false, nullptr);
    QUICHE_DCHECK(!response_sent_);
    response_sent_ = true;
    WriteGeneratedBytes();
    return;
  }
  QUIC_DVLOG(1) << "Stream " << id() << " sending response.";
  SendHeadersAndBodyAndTrailers(response->headers().Clone(), response->body(),
                                response->trailers().Clone());
}
void QuicSimpleServerStream::SendStreamData(absl::string_view data,
                                            bool close_stream) {
  QUICHE_DCHECK(!data.empty() || close_stream);
  if (close_stream) {
    SendHeadersAndBodyAndTrailers(
        std::nullopt, data,
        quiche::HttpHeaderBlock());
  } else {
    SendIncompleteResponse(std::nullopt, data);
  }
}
void QuicSimpleServerStream::TerminateStreamWithError(
    QuicResetStreamError error) {
  QUIC_DVLOG(1) << "Stream " << id() << " abruptly terminating with error "
                << error.internal_code();
  ResetWriteSide(error);
}
void QuicSimpleServerStream::OnCanWrite() {
  QuicSpdyStream::OnCanWrite();
  WriteGeneratedBytes();
}
void QuicSimpleServerStream::WriteGeneratedBytes() {
  static size_t kChunkSize = 1024;
  while (!HasBufferedData() && generate_bytes_length_ > 0) {
    size_t len = std::min<size_t>(kChunkSize, generate_bytes_length_);
    std::string data(len, 'a');
    generate_bytes_length_ -= len;
    bool fin = generate_bytes_length_ == 0;
    WriteOrBufferBody(data, fin);
  }
}
void QuicSimpleServerStream::SendNotFoundResponse() {
  QUIC_DVLOG(1) << "Stream " << id() << " sending not found response.";
  HttpHeaderBlock headers;
  headers[":status"] = "404";
  headers["content-length"] = absl::StrCat(strlen(kNotFoundResponseBody));
  SendHeadersAndBody(std::move(headers), kNotFoundResponseBody);
}
void QuicSimpleServerStream::SendErrorResponse() { SendErrorResponse(0); }
void QuicSimpleServerStream::SendErrorResponse(int resp_code) {
  QUIC_DVLOG(1) << "Stream " << id() << " sending error response.";
  if (!reading_stopped()) {
    StopReading();
  }
  HttpHeaderBlock headers;
  if (resp_code <= 0) {
    headers[":status"] = "500";
  } else {
    headers[":status"] = absl::StrCat(resp_code);
  }
  headers["content-length"] = absl::StrCat(strlen(kErrorResponseBody));
  SendHeadersAndBody(std::move(headers), kErrorResponseBody);
}
void QuicSimpleServerStream::SendIncompleteResponse(
    std::optional<HttpHeaderBlock> response_headers, absl::string_view body) {
  QUICHE_DCHECK_NE(response_headers.has_value(), response_sent_);
  if (response_headers.has_value()) {
    QUIC_DLOG(INFO) << "Stream " << id() << " writing headers (fin = false) : "
                    << response_headers.value().DebugString();
    int response_code;
    if (!ParseHeaderStatusCode(*response_headers, &response_code) ||
        response_code != 100) {
      response_sent_ = true;
    }
    WriteHeaders(std::move(response_headers).value(), false, nullptr);
  }
  QUIC_DLOG(INFO) << "Stream " << id()
                  << " writing body (fin = false) with size: " << body.size();
  if (!body.empty()) {
    WriteOrBufferBody(body, false);
  }
}
void QuicSimpleServerStream::SendHeadersAndBody(
    HttpHeaderBlock response_headers, absl::string_view body) {
  SendHeadersAndBodyAndTrailers(std::move(response_headers), body,
                                HttpHeaderBlock());
}
void QuicSimpleServerStream::SendHeadersAndBodyAndTrailers(
    std::optional<HttpHeaderBlock> response_headers, absl::string_view body,
    HttpHeaderBlock response_trailers) {
  QUICHE_DCHECK_NE(response_headers.has_value(), response_sent_);
  if (response_headers.has_value()) {
    bool send_fin = (body.empty() && response_trailers.empty());
    QUIC_DLOG(INFO) << "Stream " << id()
                    << " writing headers (fin = " << send_fin
                    << ") : " << response_headers.value().DebugString();
    WriteHeaders(std::move(response_headers).value(), send_fin, nullptr);
    response_sent_ = true;
    if (send_fin) {
      return;
    }
  }
  bool send_fin = response_trailers.empty();
  QUIC_DLOG(INFO) << "Stream " << id() << " writing body (fin = " << send_fin
                  << ") with size: " << body.size();
  if (!body.empty() || send_fin) {
    WriteOrBufferBody(body, send_fin);
  }
  if (send_fin) {
    return;
  }
  QUIC_DLOG(INFO) << "Stream " << id() << " writing trailers (fin = true): "
                  << response_trailers.DebugString();
  WriteTrailers(std::move(response_trailers), nullptr);
}
bool QuicSimpleServerStream::IsConnectRequest() const {
  auto method_it = request_headers_.find(":method");
  return method_it != request_headers_.end() && method_it->second == "CONNECT";
}
void QuicSimpleServerStream::OnInvalidHeaders() {
  QUIC_DVLOG(1) << "Invalid headers";
  SendErrorResponse(400);
}
const char* const QuicSimpleServerStream::kErrorResponseBody = "bad";
const char* const QuicSimpleServerStream::kNotFoundResponseBody =
    "file not found";
}  