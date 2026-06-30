#include "quiche/balsa/balsa_frame.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "quiche/balsa/balsa_enums.h"
#include "quiche/balsa/balsa_headers.h"
#include "quiche/balsa/balsa_visitor_interface.h"
#include "quiche/balsa/header_properties.h"
#include "quiche/common/platform/api/quiche_logging.h"
#define CHAR_LT(a, b) \
  (static_cast<unsigned char>(a) < static_cast<unsigned char>(b))
#define CHAR_LE(a, b) \
  (static_cast<unsigned char>(a) <= static_cast<unsigned char>(b))
#define CHAR_GT(a, b) \
  (static_cast<unsigned char>(a) > static_cast<unsigned char>(b))
#define CHAR_GE(a, b) \
  (static_cast<unsigned char>(a) >= static_cast<unsigned char>(b))
#define QUICHE_DCHECK_CHAR_GE(a, b) \
  QUICHE_DCHECK_GE(static_cast<unsigned char>(a), static_cast<unsigned char>(b))
namespace quiche {
namespace {
using FirstLineValidationOption =
    HttpValidationPolicy::FirstLineValidationOption;
constexpr size_t kContinueStatusCode = 100;
constexpr size_t kSwitchingProtocolsStatusCode = 101;
constexpr absl::string_view kChunked = "chunked";
constexpr absl::string_view kContentLength = "content-length";
constexpr absl::string_view kIdentity = "identity";
constexpr absl::string_view kTransferEncoding = "transfer-encoding";
bool IsInterimResponse(size_t response_code) {
  return response_code >= 100 && response_code < 200;
}
bool IsObsTextChar(char c) { return static_cast<uint8_t>(c) >= 0x80; }
}  
void BalsaFrame::Reset() {
  last_char_was_slash_r_ = false;
  saw_non_newline_char_ = false;
  start_was_space_ = true;
  chunk_length_character_extracted_ = false;
  allow_reading_until_close_for_request_ = false;
  chunk_length_remaining_ = 0;
  content_length_remaining_ = 0;
  last_slash_n_idx_ = 0;
  term_chars_ = 0;
  parse_state_ = BalsaFrameEnums::READING_HEADER_AND_FIRSTLINE;
  last_error_ = BalsaFrameEnums::BALSA_NO_ERROR;
  lines_.clear();
  if (continue_headers_ != nullptr) {
    continue_headers_->Clear();
  }
  if (headers_ != nullptr) {
    headers_->Clear();
  }
  trailer_lines_.clear();
  start_of_trailer_line_ = 0;
  trailer_length_ = 0;
  if (trailers_ != nullptr) {
    trailers_->Clear();
  }
  is_valid_target_uri_ = true;
}
namespace {
inline char* ParseOneIsland(char* current, char* begin, char* end,
                            size_t* first_whitespace, size_t* first_nonwhite) {
  *first_whitespace = current - begin;
  while (current < end && CHAR_LE(*current, ' ')) {
    ++current;
  }
  *first_nonwhite = current - begin;
  while (current < end && CHAR_GT(*current, ' ')) {
    ++current;
  }
  return current;
}
}  
bool ParseHTTPFirstLine(char* begin, char* end, bool is_request,
                        BalsaHeaders* headers,
                        BalsaFrameEnums::ErrorCode* error_code,
                        FirstLineValidationOption whitespace_option) {
  while (begin < end && (end[-1] == '\n' || end[-1] == '\r')) {
    --end;
  }
  if (whitespace_option != FirstLineValidationOption::NONE) {
    constexpr absl::string_view kBadWhitespace = "\r\t";
    char* pos = std::find_first_of(begin, end, kBadWhitespace.begin(),
                                   kBadWhitespace.end());
    if (pos != end) {
      if (whitespace_option == FirstLineValidationOption::REJECT) {
        *error_code = static_cast<BalsaFrameEnums::ErrorCode>(
            BalsaFrameEnums::INVALID_WS_IN_STATUS_LINE +
            static_cast<int>(is_request));
        return false;
      }
      QUICHE_DCHECK(whitespace_option == FirstLineValidationOption::SANITIZE);
      std::replace_if(
          pos, end, [](char c) { return c == '\r' || c == '\t'; }, ' ');
    }
  }
  char* current = ParseOneIsland(begin, begin, end, &headers->whitespace_1_idx_,
                                 &headers->non_whitespace_1_idx_);
  current = ParseOneIsland(current, begin, end, &headers->whitespace_2_idx_,
                           &headers->non_whitespace_2_idx_);
  current = ParseOneIsland(current, begin, end, &headers->whitespace_3_idx_,
                           &headers->non_whitespace_3_idx_);
  const char* last = end;
  while (current <= last && CHAR_LE(*last, ' ')) {
    --last;
  }
  headers->whitespace_4_idx_ = last - begin + 1;
  QUICHE_DCHECK(begin == end || static_cast<unsigned char>(*begin) > ' ');
  QUICHE_DCHECK_EQ(0u, headers->whitespace_1_idx_);
  QUICHE_DCHECK_EQ(0u, headers->non_whitespace_1_idx_);
  QUICHE_DCHECK(begin == end ||
                headers->non_whitespace_1_idx_ < headers->whitespace_2_idx_);
  if (headers->non_whitespace_2_idx_ == headers->whitespace_3_idx_) {
    *error_code = static_cast<BalsaFrameEnums::ErrorCode>(
        BalsaFrameEnums::FAILED_TO_FIND_WS_AFTER_RESPONSE_VERSION +
        static_cast<int>(is_request));
    if (!is_request) {  
      return false;
    }
  }
  if (headers->whitespace_3_idx_ == headers->non_whitespace_3_idx_) {
    if (*error_code == BalsaFrameEnums::BALSA_NO_ERROR) {
      *error_code = static_cast<BalsaFrameEnums::ErrorCode>(
          BalsaFrameEnums::FAILED_TO_FIND_WS_AFTER_RESPONSE_STATUSCODE +
          static_cast<int>(is_request));
    }
  }
  if (!is_request) {
    headers->parsed_response_code_ = 0;
    if (headers->non_whitespace_2_idx_ < headers->whitespace_3_idx_) {
      if (!absl::SimpleAtoi(
              absl::string_view(begin + headers->non_whitespace_2_idx_,
                                headers->non_whitespace_3_idx_ -
                                    headers->non_whitespace_2_idx_),
              &headers->parsed_response_code_)) {
        *error_code = BalsaFrameEnums::FAILED_CONVERTING_STATUS_CODE_TO_INT;
        return false;
      }
    }
  }
  return true;
}
namespace {
bool IsValidTargetUri(absl::string_view method, absl::string_view target_uri) {
  if (target_uri.empty()) {
    QUICHE_CODE_COUNT(invalid_target_uri_empty);
    return false;
  }
  if (target_uri == "*") {
    if (method == "OPTIONS") {
      return true;
    }
    QUICHE_CODE_COUNT(invalid_target_uri_asterisk_not_options);
    return false;
  }
  if (method == "CONNECT") {
    size_t index = target_uri.find_last_of(':');
    if (index == absl::string_view::npos || index == 0) {
      QUICHE_CODE_COUNT(invalid_target_uri_connect_missing_port);
      return false;
    }
    if (target_uri[0] == '[' && target_uri[index - 1] != ']') {
      QUICHE_CODE_COUNT(invalid_target_uri_connect_bad_v6_literal);
      return false;
    }
    int port;
    if (!absl::SimpleAtoi(target_uri.substr(index + 1), &port) || port < 0 ||
        port > 65535) {
      QUICHE_CODE_COUNT(invalid_target_uri_connect_bad_port);
      return false;
    }
    return true;
  }
  if (target_uri[0] == '/' || absl::StrContains(target_uri, ":
    return true;
  }
  QUICHE_CODE_COUNT(invalid_target_uri_bad_path);
  return false;
}
}  
void BalsaFrame::ProcessFirstLine(char* begin, char* end) {
  BalsaFrameEnums::ErrorCode previous_error = last_error_;
  if (!ParseHTTPFirstLine(
          begin, end, is_request_, headers_, &last_error_,
          http_validation_policy().sanitize_cr_tab_in_first_line)) {
    parse_state_ = BalsaFrameEnums::ERROR;
    HandleError(last_error_);
    return;
  }
  if (previous_error != last_error_) {
    HandleWarning(last_error_);
  }
  const absl::string_view line_input(
      begin + headers_->non_whitespace_1_idx_,
      headers_->whitespace_4_idx_ - headers_->non_whitespace_1_idx_);
  const absl::string_view part1(
      begin + headers_->non_whitespace_1_idx_,
      headers_->whitespace_2_idx_ - headers_->non_whitespace_1_idx_);
  const absl::string_view part2(
      begin + headers_->non_whitespace_2_idx_,
      headers_->whitespace_3_idx_ - headers_->non_whitespace_2_idx_);
  const absl::string_view part3(
      begin + headers_->non_whitespace_3_idx_,
      headers_->whitespace_4_idx_ - headers_->non_whitespace_3_idx_);
  if (is_request_) {
    is_valid_target_uri_ = IsValidTargetUri(part1, part2);
    if (http_validation_policy().disallow_invalid_target_uris &&
        !is_valid_target_uri_) {
      parse_state_ = BalsaFrameEnums::ERROR;
      last_error_ = BalsaFrameEnums::INVALID_TARGET_URI;
      HandleError(last_error_);
      return;
    }
    visitor_->OnRequestFirstLineInput(line_input, part1, part2, part3);
    if (part3.empty()) {
      parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
    }
    return;
  }
  visitor_->OnResponseFirstLineInput(line_input, part1, part2, part3);
}
void BalsaFrame::CleanUpKeyValueWhitespace(
    const char* stream_begin, const char* line_begin, const char* current,
    const char* line_end, HeaderLineDescription* current_header_line) {
  const char* colon_loc = current;
  QUICHE_DCHECK_LT(colon_loc, line_end);
  QUICHE_DCHECK_EQ(':', *colon_loc);
  QUICHE_DCHECK_EQ(':', *current);
  QUICHE_DCHECK_CHAR_GE(' ', *line_end)
      << "\"" << std::string(line_begin, line_end) << "\"";
  --current;
  while (current > line_begin && CHAR_LE(*current, ' ')) {
    --current;
  }
  current += static_cast<int>(current != colon_loc);
  current_header_line->key_end_idx = current - stream_begin;
  current = colon_loc;
  QUICHE_DCHECK_EQ(':', *current);
  ++current;
  while (current < line_end && CHAR_LE(*current, ' ')) {
    ++current;
  }
  current_header_line->value_begin_idx = current - stream_begin;
  QUICHE_DCHECK_GE(current_header_line->key_end_idx,
                   current_header_line->first_char_idx);
  QUICHE_DCHECK_GE(current_header_line->value_begin_idx,
                   current_header_line->key_end_idx);
  QUICHE_DCHECK_GE(current_header_line->last_char_idx,
                   current_header_line->value_begin_idx);
}
bool BalsaFrame::FindColonsAndParseIntoKeyValue(const Lines& lines,
                                                bool is_trailer,
                                                BalsaHeaders* headers) {
  QUICHE_DCHECK(!lines.empty());
  const char* stream_begin = headers->OriginalHeaderStreamBegin();
  const Lines::size_type lines_size_m1 = lines.size() - 1;
  int first_header_idx = (is_trailer ? 0 : 1);
  const char* current = stream_begin + lines[first_header_idx].first;
  for (Lines::size_type i = first_header_idx; i < lines_size_m1;) {
    const char* line_begin = stream_begin + lines[i].first;
    for (++i; i < lines_size_m1; ++i) {
      const char c = *(stream_begin + lines[i].first);
      if (CHAR_GT(c, ' ')) {
        break;
      }
      if ((c != ' ' && c != '\t') ||
          http_validation_policy().disallow_header_continuation_lines) {
        HandleError(is_trailer ? BalsaFrameEnums::INVALID_TRAILER_FORMAT
                               : BalsaFrameEnums::INVALID_HEADER_FORMAT);
        return false;
      }
    }
    const char* line_end = stream_begin + lines[i - 1].second;
    QUICHE_DCHECK_LT(line_begin - stream_begin, line_end - stream_begin);
    --line_end;
    QUICHE_DCHECK_EQ('\n', *line_end)
        << "\"" << std::string(line_begin, line_end) << "\"";
    while (CHAR_LE(*line_end, ' ') && line_end > line_begin) {
      --line_end;
    }
    ++line_end;
    QUICHE_DCHECK_CHAR_GE(' ', *line_end);
    QUICHE_DCHECK_LT(line_begin, line_end);
    headers->header_lines_.push_back(HeaderLineDescription(
        line_begin - stream_begin, line_end - stream_begin,
        line_end - stream_begin, line_end - stream_begin, 0));
    if (current >= line_end) {
      if (http_validation_policy().require_header_colon) {
        HandleError(is_trailer ? BalsaFrameEnums::TRAILER_MISSING_COLON
                               : BalsaFrameEnums::HEADER_MISSING_COLON);
        return false;
      }
      HandleWarning(is_trailer ? BalsaFrameEnums::TRAILER_MISSING_COLON
                               : BalsaFrameEnums::HEADER_MISSING_COLON);
      continue;
    }
    if (current < line_begin) {
      current = line_begin;
    }
    for (; current < line_end; ++current) {
      const char c = *current;
      if (c == ':') {
        break;
      }
      if (http_validation_policy().disallow_double_quote_in_header_name) {
        if (header_properties::IsInvalidHeaderKeyChar(c)) {
          HandleError(is_trailer
                          ? BalsaFrameEnums::INVALID_TRAILER_NAME_CHARACTER
                          : BalsaFrameEnums::INVALID_HEADER_NAME_CHARACTER);
          return false;
        }
      } else if (header_properties::IsInvalidHeaderKeyCharAllowDoubleQuote(c)) {
        HandleError(is_trailer
                        ? BalsaFrameEnums::INVALID_TRAILER_NAME_CHARACTER
                        : BalsaFrameEnums::INVALID_HEADER_NAME_CHARACTER);
        return false;
      }
      if (http_validation_policy().disallow_obs_text_in_field_names &&
          IsObsTextChar(c)) {
        HandleError(is_trailer
                        ? BalsaFrameEnums::INVALID_TRAILER_NAME_CHARACTER
                        : BalsaFrameEnums::INVALID_HEADER_NAME_CHARACTER);
        return false;
      }
    }
    if (current == line_end) {
      if (http_validation_policy().require_header_colon) {
        HandleError(is_trailer ? BalsaFrameEnums::TRAILER_MISSING_COLON
                               : BalsaFrameEnums::HEADER_MISSING_COLON);
        return false;
      }
      HandleWarning(is_trailer ? BalsaFrameEnums::TRAILER_MISSING_COLON
                               : BalsaFrameEnums::HEADER_MISSING_COLON);
      continue;
    }
    QUICHE_DCHECK_EQ(*current, ':');
    QUICHE_DCHECK_LE(current - stream_begin, line_end - stream_begin);
    QUICHE_DCHECK_LE(stream_begin - stream_begin, current - stream_begin);
    HeaderLineDescription& current_header_line = headers->header_lines_.back();
    current_header_line.key_end_idx = current - stream_begin;
    current_header_line.value_begin_idx = current_header_line.key_end_idx;
    if (current < line_end) {
      ++current_header_line.key_end_idx;
      CleanUpKeyValueWhitespace(stream_begin, line_begin, current, line_end,
                                &current_header_line);
    }
  }
  return true;
}
void BalsaFrame::HandleWarning(BalsaFrameEnums::ErrorCode error_code) {
  last_error_ = error_code;
  visitor_->HandleWarning(last_error_);
}
void BalsaFrame::HandleError(BalsaFrameEnums::ErrorCode error_code) {
  last_error_ = error_code;
  parse_state_ = BalsaFrameEnums::ERROR;
  visitor_->HandleError(last_error_);
}
BalsaHeadersEnums::ContentLengthStatus BalsaFrame::ProcessContentLengthLine(
    HeaderLines::size_type line_idx, size_t* length) {
  const HeaderLineDescription& header_line = headers_->header_lines_[line_idx];
  const char* stream_begin = headers_->OriginalHeaderStreamBegin();
  const char* line_end = stream_begin + header_line.last_char_idx;
  const char* value_begin = (stream_begin + header_line.value_begin_idx);
  if (value_begin >= line_end) {
    QUICHE_DVLOG(1) << "invalid content-length -- no non-whitespace value data";
    return BalsaHeadersEnums::INVALID_CONTENT_LENGTH;
  }
  *length = 0;
  while (value_begin < line_end) {
    if (*value_begin < '0' || *value_begin > '9') {
      QUICHE_DVLOG(1)
          << "invalid content-length - non numeric character detected";
      return BalsaHeadersEnums::INVALID_CONTENT_LENGTH;
    }
    const size_t kMaxDiv10 = std::numeric_limits<size_t>::max() / 10;
    size_t length_x_10 = *length * 10;
    const size_t c = *value_begin - '0';
    if (*length > kMaxDiv10 ||
        (std::numeric_limits<size_t>::max() - length_x_10) < c) {
      QUICHE_DVLOG(1) << "content-length overflow";
      return BalsaHeadersEnums::CONTENT_LENGTH_OVERFLOW;
    }
    *length = length_x_10 + c;
    ++value_begin;
  }
  QUICHE_DVLOG(1) << "content_length parsed: " << *length;
  return BalsaHeadersEnums::VALID_CONTENT_LENGTH;
}
void BalsaFrame::ProcessTransferEncodingLine(HeaderLines::size_type line_idx) {
  const HeaderLineDescription& header_line = headers_->header_lines_[line_idx];
  const char* stream_begin = headers_->OriginalHeaderStreamBegin();
  const absl::string_view transfer_encoding(
      stream_begin + header_line.value_begin_idx,
      header_line.last_char_idx - header_line.value_begin_idx);
  if (absl::EqualsIgnoreCase(transfer_encoding, kChunked)) {
    headers_->transfer_encoding_is_chunked_ = true;
    return;
  }
  if (absl::EqualsIgnoreCase(transfer_encoding, kIdentity)) {
    headers_->transfer_encoding_is_chunked_ = false;
    return;
  }
  if (http_validation_policy().validate_transfer_encoding) {
    HandleError(BalsaFrameEnums::UNKNOWN_TRANSFER_ENCODING);
  }
}
bool BalsaFrame::CheckHeaderLinesForInvalidChars(const Lines& lines,
                                                 const BalsaHeaders* headers) {
  const char* stream_begin =
      headers->OriginalHeaderStreamBegin() + lines.front().first;
  const char* stream_end =
      headers->OriginalHeaderStreamBegin() + lines.back().second;
  bool found_invalid = false;
  for (const char* c = stream_begin; c < stream_end; c++) {
    if (header_properties::IsInvalidHeaderChar(*c)) {
      found_invalid = true;
    }
    if (*c == '\r' &&
        http_validation_policy().disallow_lone_cr_in_request_headers &&
        c + 1 < stream_end && *(c + 1) != '\n') {
      found_invalid = true;
    }
  }
  return found_invalid;
}
void BalsaFrame::ProcessHeaderLines(const Lines& lines, bool is_trailer,
                                    BalsaHeaders* headers) {
  QUICHE_DCHECK(!lines.empty());
  QUICHE_DVLOG(1) << "******@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@**********\n";
  if (invalid_chars_error_enabled() &&
      CheckHeaderLinesForInvalidChars(lines, headers)) {
    HandleError(BalsaFrameEnums::INVALID_HEADER_CHARACTER);
    return;
  }
  if (lines.size() <= (is_trailer ? 1 : 2)) {
    return;
  }
  HeaderLines::size_type content_length_idx = 0;
  HeaderLines::size_type transfer_encoding_idx = 0;
  const char* stream_begin = headers->OriginalHeaderStreamBegin();
  if (!FindColonsAndParseIntoKeyValue(lines, is_trailer, headers)) {
    return;
  }
  const HeaderLines::size_type lines_size = headers->header_lines_.size();
  for (HeaderLines::size_type i = 0; i < lines_size; ++i) {
    const HeaderLineDescription& line = headers->header_lines_[i];
    const absl::string_view key(stream_begin + line.first_char_idx,
                                line.key_end_idx - line.first_char_idx);
    QUICHE_DVLOG(2) << "[" << i << "]: " << key << " key_len: " << key.length();
    if (key.empty() || key[0] == ' ') {
      parse_state_ = BalsaFrameEnums::ERROR;
      HandleError(is_trailer ? BalsaFrameEnums::INVALID_TRAILER_FORMAT
                             : BalsaFrameEnums::INVALID_HEADER_FORMAT);
      return;
    }
    if (is_trailer) {
      continue;
    }
    if (absl::EqualsIgnoreCase(key, kContentLength)) {
      size_t length = 0;
      BalsaHeadersEnums::ContentLengthStatus content_length_status =
          ProcessContentLengthLine(i, &length);
      if (content_length_idx == 0) {
        content_length_idx = i + 1;
        headers->content_length_status_ = content_length_status;
        headers->content_length_ = length;
        content_length_remaining_ = length;
        continue;
      }
      if ((headers->content_length_status_ != content_length_status) ||
          ((headers->content_length_status_ ==
            BalsaHeadersEnums::VALID_CONTENT_LENGTH) &&
           (http_validation_policy().disallow_multiple_content_length ||
            length != headers->content_length_))) {
        HandleError(BalsaFrameEnums::MULTIPLE_CONTENT_LENGTH_KEYS);
        return;
      }
      continue;
    }
    if (absl::EqualsIgnoreCase(key, kTransferEncoding)) {
      if (http_validation_policy().validate_transfer_encoding &&
          transfer_encoding_idx != 0) {
        HandleError(BalsaFrameEnums::MULTIPLE_TRANSFER_ENCODING_KEYS);
        return;
      }
      transfer_encoding_idx = i + 1;
    }
  }
  if (!is_trailer) {
    if (http_validation_policy().validate_transfer_encoding &&
        http_validation_policy()
            .disallow_transfer_encoding_with_content_length &&
        content_length_idx != 0 && transfer_encoding_idx != 0) {
      HandleError(BalsaFrameEnums::BOTH_TRANSFER_ENCODING_AND_CONTENT_LENGTH);
      return;
    }
    if (headers->transfer_encoding_is_chunked_) {
      headers->content_length_ = 0;
      headers->content_length_status_ = BalsaHeadersEnums::NO_CONTENT_LENGTH;
      content_length_remaining_ = 0;
    }
    if (transfer_encoding_idx != 0) {
      ProcessTransferEncodingLine(transfer_encoding_idx - 1);
    }
  }
}
void BalsaFrame::AssignParseStateAfterHeadersHaveBeenParsed() {
  parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
  int response_code = headers_->parsed_response_code_;
  if (!is_request_ && (request_was_head_ ||
                       !BalsaHeaders::ResponseCanHaveBody(response_code))) {
    return;
  }
  if (headers_->transfer_encoding_is_chunked_) {
    parse_state_ = BalsaFrameEnums::READING_CHUNK_LENGTH;
    return;
  }
  switch (headers_->content_length_status_) {
    case BalsaHeadersEnums::VALID_CONTENT_LENGTH:
      if (headers_->content_length_ == 0) {
        parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
      } else {
        parse_state_ = BalsaFrameEnums::READING_CONTENT;
      }
      break;
    case BalsaHeadersEnums::CONTENT_LENGTH_OVERFLOW:
    case BalsaHeadersEnums::INVALID_CONTENT_LENGTH:
      HandleError(BalsaFrameEnums::UNPARSABLE_CONTENT_LENGTH);
      break;
    case BalsaHeadersEnums::NO_CONTENT_LENGTH:
      if (is_request_) {
        const absl::string_view method = headers_->request_method();
        if ((method != "POST" && method != "PUT") ||
            !http_validation_policy().require_content_length_if_body_required) {
          parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
          break;
        } else if (!allow_reading_until_close_for_request_) {
          HandleError(BalsaFrameEnums::REQUIRED_BODY_BUT_NO_CONTENT_LENGTH);
          break;
        }
      }
      parse_state_ = BalsaFrameEnums::READING_UNTIL_CLOSE;
      HandleWarning(BalsaFrameEnums::MAYBE_BODY_BUT_NO_CONTENT_LENGTH);
      break;
    default:
      QUICHE_LOG(FATAL) << "Saw a content_length_status: "
                        << headers_->content_length_status_
                        << " which is unknown.";
  }
}
size_t BalsaFrame::ProcessHeaders(const char* message_start,
                                  size_t message_length) {
  const char* const original_message_start = message_start;
  const char* const message_end = message_start + message_length;
  const char* message_current = message_start;
  const char* checkpoint = message_start;
  if (message_length == 0) {
    return message_current - original_message_start;
  }
  while (message_current < message_end) {
    size_t base_idx = headers_->GetReadableBytesFromHeaderStream();
    if (!saw_non_newline_char_) {
      do {
        const char c = *message_current;
        if (c != '\r' && c != '\n') {
          if (CHAR_LE(c, ' ')) {
            HandleError(BalsaFrameEnums::NO_REQUEST_LINE_IN_REQUEST);
            return message_current - original_message_start;
          }
          break;
        }
        ++message_current;
        if (message_current == message_end) {
          return message_current - original_message_start;
        }
      } while (true);
      saw_non_newline_char_ = true;
      message_start = message_current;
      checkpoint = message_current;
    }
    while (message_current < message_end) {
      if (*message_current != '\n') {
        ++message_current;
        continue;
      }
      const size_t relative_idx = message_current - message_start;
      const size_t message_current_idx = 1 + base_idx + relative_idx;
      lines_.push_back(std::make_pair(last_slash_n_idx_, message_current_idx));
      if (lines_.size() == 1) {
        headers_->WriteFromFramer(checkpoint, 1 + message_current - checkpoint);
        checkpoint = message_current + 1;
        char* begin = headers_->OriginalHeaderStreamBegin();
        QUICHE_DVLOG(1) << "First line "
                        << std::string(begin, lines_[0].second);
        QUICHE_DVLOG(1) << "is_request_: " << is_request_;
        ProcessFirstLine(begin, begin + lines_[0].second);
        if (parse_state_ == BalsaFrameEnums::MESSAGE_FULLY_READ) {
          break;
        }
        if (parse_state_ == BalsaFrameEnums::ERROR) {
          return message_current - original_message_start;
        }
      }
      const size_t chars_since_last_slash_n =
          (message_current_idx - last_slash_n_idx_);
      last_slash_n_idx_ = message_current_idx;
      if (chars_since_last_slash_n > 2) {
        ++message_current;
        continue;
      }
      if ((chars_since_last_slash_n == 1) ||
          (((message_current > message_start) &&
            (*(message_current - 1) == '\r')) ||
           (last_char_was_slash_r_))) {
        break;
      }
      ++message_current;
    }
    if (message_current == message_end) {
      continue;
    }
    ++message_current;
    QUICHE_DCHECK(message_current >= message_start);
    if (message_current > message_start) {
      headers_->WriteFromFramer(checkpoint, message_current - checkpoint);
    }
    if (headers_->GetReadableBytesFromHeaderStream() > max_header_length_) {
      HandleHeadersTooLongError();
      return message_current - original_message_start;
    }
    headers_->DoneWritingFromFramer();
    visitor_->OnHeaderInput(headers_->GetReadablePtrFromHeaderStream());
    ProcessHeaderLines(lines_, false , headers_);
    if (parse_state_ == BalsaFrameEnums::ERROR) {
      return message_current - original_message_start;
    }
    if (use_interim_headers_callback_ &&
        IsInterimResponse(headers_->parsed_response_code()) &&
        headers_->parsed_response_code() != kSwitchingProtocolsStatusCode) {
      visitor_->OnInterimHeaders(
          std::make_unique<BalsaHeaders>(std::move(*headers_)));
      Reset();
      checkpoint = message_start = message_current;
      continue;
    }
    if (continue_headers_ != nullptr &&
        headers_->parsed_response_code_ == kContinueStatusCode) {
      BalsaHeaders saved_continue_headers = std::move(*headers_);
      Reset();
      *continue_headers_ = std::move(saved_continue_headers);
      visitor_->ContinueHeaderDone();
      checkpoint = message_start = message_current;
      continue;
    }
    AssignParseStateAfterHeadersHaveBeenParsed();
    if (parse_state_ == BalsaFrameEnums::ERROR) {
      return message_current - original_message_start;
    }
    visitor_->ProcessHeaders(*headers_);
    visitor_->HeaderDone();
    if (parse_state_ == BalsaFrameEnums::MESSAGE_FULLY_READ) {
      visitor_->MessageDone();
    }
    return message_current - original_message_start;
  }
  last_char_was_slash_r_ = (*(message_end - 1) == '\r');
  QUICHE_DCHECK(message_current >= message_start);
  if (message_current > message_start) {
    headers_->WriteFromFramer(checkpoint, message_current - checkpoint);
  }
  return message_current - original_message_start;
}
size_t BalsaFrame::BytesSafeToSplice() const {
  switch (parse_state_) {
    case BalsaFrameEnums::READING_CHUNK_DATA:
      return chunk_length_remaining_;
    case BalsaFrameEnums::READING_UNTIL_CLOSE:
      return std::numeric_limits<size_t>::max();
    case BalsaFrameEnums::READING_CONTENT:
      return content_length_remaining_;
    default:
      return 0;
  }
}
void BalsaFrame::BytesSpliced(size_t bytes_spliced) {
  switch (parse_state_) {
    case BalsaFrameEnums::READING_CHUNK_DATA:
      if (chunk_length_remaining_ < bytes_spliced) {
        HandleError(BalsaFrameEnums::
                        CALLED_BYTES_SPLICED_AND_EXCEEDED_SAFE_SPLICE_AMOUNT);
        return;
      }
      chunk_length_remaining_ -= bytes_spliced;
      if (chunk_length_remaining_ == 0) {
        parse_state_ = BalsaFrameEnums::READING_CHUNK_TERM;
      }
      return;
    case BalsaFrameEnums::READING_UNTIL_CLOSE:
      return;
    case BalsaFrameEnums::READING_CONTENT:
      if (content_length_remaining_ < bytes_spliced) {
        HandleError(BalsaFrameEnums::
                        CALLED_BYTES_SPLICED_AND_EXCEEDED_SAFE_SPLICE_AMOUNT);
        return;
      }
      content_length_remaining_ -= bytes_spliced;
      if (content_length_remaining_ == 0) {
        parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
        visitor_->MessageDone();
      }
      return;
    default:
      HandleError(BalsaFrameEnums::CALLED_BYTES_SPLICED_WHEN_UNSAFE_TO_DO_SO);
      return;
  }
}
size_t BalsaFrame::ProcessInput(const char* input, size_t size) {
  const char* current = input;
  const char* on_entry = current;
  const char* end = current + size;
  QUICHE_DCHECK(headers_ != nullptr);
  if (headers_ == nullptr) {
    return 0;
  }
  if (parse_state_ == BalsaFrameEnums::READING_HEADER_AND_FIRSTLINE) {
    const size_t header_length = headers_->GetReadableBytesFromHeaderStream();
    if (header_length > max_header_length_ ||
        (header_length == max_header_length_ && size > 0)) {
      HandleHeadersTooLongError();
      return current - input;
    }
    const size_t bytes_to_process =
        std::min(max_header_length_ - header_length, size);
    current += ProcessHeaders(input, bytes_to_process);
    if (parse_state_ == BalsaFrameEnums::READING_HEADER_AND_FIRSTLINE) {
      const size_t header_length_after =
          headers_->GetReadableBytesFromHeaderStream();
      if (header_length_after >= max_header_length_) {
        HandleHeadersTooLongError();
      }
    }
    return current - input;
  }
  if (parse_state_ == BalsaFrameEnums::MESSAGE_FULLY_READ ||
      parse_state_ == BalsaFrameEnums::ERROR) {
    return current - input;
  }
  QUICHE_DCHECK_LE(current, end);
  if (current == end) {
    return current - input;
  }
  while (true) {
    switch (parse_state_) {
      case BalsaFrameEnums::READING_CHUNK_LENGTH:
        QUICHE_DCHECK_LE(current, end);
        while (true) {
          if (current == end) {
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            return current - input;
          }
          const char c = *current;
          ++current;
          static const signed char kBad = -1;
          static const signed char kDelimiter = -2;
          signed char addition = kBad;
          switch (c) {
            case '0': addition = 0; break;
            case '1': addition = 1; break;
            case '2': addition = 2; break;
            case '3': addition = 3; break;
            case '4': addition = 4; break;
            case '5': addition = 5; break;
            case '6': addition = 6; break;
            case '7': addition = 7; break;
            case '8': addition = 8; break;
            case '9': addition = 9; break;
            case 'a': addition = 0xA; break;
            case 'b': addition = 0xB; break;
            case 'c': addition = 0xC; break;
            case 'd': addition = 0xD; break;
            case 'e': addition = 0xE; break;
            case 'f': addition = 0xF; break;
            case 'A': addition = 0xA; break;
            case 'B': addition = 0xB; break;
            case 'C': addition = 0xC; break;
            case 'D': addition = 0xD; break;
            case 'E': addition = 0xE; break;
            case 'F': addition = 0xF; break;
            case '\t':
            case '\n':
            case '\r':
            case ' ':
            case ';':
              addition = kDelimiter;
              break;
            default:
              break;
          }
          if (addition >= 0) {
            chunk_length_character_extracted_ = true;
            size_t length_x_16 = chunk_length_remaining_ * 16;
            const size_t kMaxDiv16 = std::numeric_limits<size_t>::max() / 16;
            if ((chunk_length_remaining_ > kMaxDiv16) ||
                (std::numeric_limits<size_t>::max() - length_x_16) <
                    static_cast<size_t>(addition)) {
              visitor_->OnRawBodyInput(
                  absl::string_view(on_entry, current - on_entry));
              HandleError(BalsaFrameEnums::CHUNK_LENGTH_OVERFLOW);
              return current - input;
            }
            chunk_length_remaining_ = length_x_16 + addition;
            continue;
          }
          if (!chunk_length_character_extracted_ || addition == kBad) {
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            HandleError(BalsaFrameEnums::INVALID_CHUNK_LENGTH);
            return current - input;
          }
          break;
        }
        --current;
        parse_state_ = BalsaFrameEnums::READING_CHUNK_EXTENSION;
        last_char_was_slash_r_ = false;
        visitor_->OnChunkLength(chunk_length_remaining_);
        continue;
      case BalsaFrameEnums::READING_CHUNK_EXTENSION: {
        const char* extensions_start = current;
        size_t extensions_length = 0;
        QUICHE_DCHECK_LE(current, end);
        while (true) {
          if (current == end) {
            visitor_->OnChunkExtensionInput(
                absl::string_view(extensions_start, extensions_length));
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            return current - input;
          }
          const char c = *current;
          if (http_validation_policy_.disallow_lone_cr_in_chunk_extension) {
            const bool cr_followed_by_non_lf =
                c == '\r' && current + 1 < end && *(current + 1) != '\n';
            const bool previous_cr_followed_by_non_lf =
                last_char_was_slash_r_ && current == input && c != '\n';
            if (cr_followed_by_non_lf || previous_cr_followed_by_non_lf) {
              HandleError(BalsaFrameEnums::INVALID_CHUNK_EXTENSION);
              return current - input;
            }
            if (current + 1 == end) {
              last_char_was_slash_r_ = c == '\r';
            }
          }
          if (c == '\r' || c == '\n') {
            extensions_length = (extensions_start == current)
                                    ? 0
                                    : current - extensions_start - 1;
          }
          ++current;
          if (c == '\n') {
            break;
          }
        }
        chunk_length_character_extracted_ = false;
        visitor_->OnChunkExtensionInput(
            absl::string_view(extensions_start, extensions_length));
        if (chunk_length_remaining_ != 0) {
          parse_state_ = BalsaFrameEnums::READING_CHUNK_DATA;
          continue;
        }
        HeaderFramingFound('\n');
        parse_state_ = BalsaFrameEnums::READING_LAST_CHUNK_TERM;
        continue;
      }
      case BalsaFrameEnums::READING_CHUNK_DATA:
        while (current < end) {
          if (chunk_length_remaining_ == 0) {
            break;
          }
          size_t bytes_remaining = end - current;
          size_t consumed_bytes = (chunk_length_remaining_ < bytes_remaining)
                                      ? chunk_length_remaining_
                                      : bytes_remaining;
          const char* tmp_current = current + consumed_bytes;
          visitor_->OnRawBodyInput(
              absl::string_view(on_entry, tmp_current - on_entry));
          visitor_->OnBodyChunkInput(
              absl::string_view(current, consumed_bytes));
          on_entry = current = tmp_current;
          chunk_length_remaining_ -= consumed_bytes;
        }
        if (chunk_length_remaining_ == 0) {
          parse_state_ = BalsaFrameEnums::READING_CHUNK_TERM;
          continue;
        }
        visitor_->OnRawBodyInput(
            absl::string_view(on_entry, current - on_entry));
        return current - input;
      case BalsaFrameEnums::READING_CHUNK_TERM:
        QUICHE_DCHECK_LE(current, end);
        while (true) {
          if (current == end) {
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            return current - input;
          }
          const char c = *current;
          ++current;
          if (c == '\n') {
            break;
          }
        }
        parse_state_ = BalsaFrameEnums::READING_CHUNK_LENGTH;
        continue;
      case BalsaFrameEnums::READING_LAST_CHUNK_TERM:
        QUICHE_DCHECK_LE(current, end);
        while (true) {
          if (current == end) {
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            return current - input;
          }
          const char c = *current;
          if (HeaderFramingFound(c) != 0) {
            ++current;
            parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
            visitor_->OnRawBodyInput(
                absl::string_view(on_entry, current - on_entry));
            visitor_->MessageDone();
            return current - input;
          }
          if (!HeaderFramingMayBeFound()) {
            break;
          }
          ++current;
        }
        parse_state_ = BalsaFrameEnums::READING_TRAILER;
        visitor_->OnRawBodyInput(
            absl::string_view(on_entry, current - on_entry));
        on_entry = current;
        continue;
      case BalsaFrameEnums::READING_TRAILER:
        while (current < end) {
          const char c = *current;
          ++current;
          ++trailer_length_;
          if (trailers_ != nullptr) {
            if (trailer_length_ > max_header_length_) {
              --current;
              HandleError(BalsaFrameEnums::TRAILER_TOO_LONG);
              return current - input;
            }
            if (LineFramingFound(c)) {
              trailer_lines_.push_back(
                  std::make_pair(start_of_trailer_line_, trailer_length_));
              start_of_trailer_line_ = trailer_length_;
            }
          }
          if (HeaderFramingFound(c) != 0) {
            parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
            if (trailers_ != nullptr) {
              trailers_->WriteFromFramer(on_entry, current - on_entry);
              trailers_->DoneWritingFromFramer();
              ProcessHeaderLines(trailer_lines_, true ,
                                 trailers_.get());
              if (parse_state_ == BalsaFrameEnums::ERROR) {
                return current - input;
              }
              visitor_->OnTrailers(std::move(trailers_));
              trailers_ = std::make_unique<BalsaHeaders>();
            }
            visitor_->OnTrailerInput(
                absl::string_view(on_entry, current - on_entry));
            visitor_->MessageDone();
            return current - input;
          }
        }
        if (trailers_ != nullptr) {
          trailers_->WriteFromFramer(on_entry, current - on_entry);
        }
        visitor_->OnTrailerInput(
            absl::string_view(on_entry, current - on_entry));
        return current - input;
      case BalsaFrameEnums::READING_UNTIL_CLOSE: {
        const size_t bytes_remaining = end - current;
        if (bytes_remaining > 0) {
          visitor_->OnRawBodyInput(absl::string_view(current, bytes_remaining));
          visitor_->OnBodyChunkInput(
              absl::string_view(current, bytes_remaining));
          current += bytes_remaining;
        }
        return current - input;
      }
      case BalsaFrameEnums::READING_CONTENT:
        while ((content_length_remaining_ != 0u) && current < end) {
          const size_t bytes_remaining = end - current;
          const size_t consumed_bytes =
              (content_length_remaining_ < bytes_remaining)
                  ? content_length_remaining_
                  : bytes_remaining;
          visitor_->OnRawBodyInput(absl::string_view(current, consumed_bytes));
          visitor_->OnBodyChunkInput(
              absl::string_view(current, consumed_bytes));
          current += consumed_bytes;
          content_length_remaining_ -= consumed_bytes;
        }
        if (content_length_remaining_ == 0) {
          parse_state_ = BalsaFrameEnums::MESSAGE_FULLY_READ;
          visitor_->MessageDone();
        }
        return current - input;
      default:
        QUICHE_LOG(FATAL) << "Unknown state: " << parse_state_  
                          << " memory corruption?!";            
    }
  }
}
void BalsaFrame::HandleHeadersTooLongError() {
  if (parse_truncated_headers_even_when_headers_too_long_) {
    const size_t len = headers_->GetReadableBytesFromHeaderStream();
    const char* stream_begin = headers_->OriginalHeaderStreamBegin();
    if (last_slash_n_idx_ < len && stream_begin[last_slash_n_idx_] != '\r') {
      static const absl::string_view kTwoLineEnds = "\r\n\r\n";
      headers_->WriteFromFramer(kTwoLineEnds.data(), kTwoLineEnds.size());
      lines_.push_back(std::make_pair(last_slash_n_idx_, len + 2));
      lines_.push_back(std::make_pair(len + 2, len + 4));
    }
    ProcessHeaderLines(lines_, false, headers_);
  }
  HandleError(BalsaFrameEnums::HEADERS_TOO_LONG);
}
const int32_t BalsaFrame::kValidTerm1;
const int32_t BalsaFrame::kValidTerm1Mask;
const int32_t BalsaFrame::kValidTerm2;
const int32_t BalsaFrame::kValidTerm2Mask;
}  
#undef CHAR_LT
#undef CHAR_LE
#undef CHAR_GT
#undef CHAR_GE
#undef QUICHE_DCHECK_CHAR_GE