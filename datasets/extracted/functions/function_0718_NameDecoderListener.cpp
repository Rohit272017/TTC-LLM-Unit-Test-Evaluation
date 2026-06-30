#include "quiche/http2/hpack/decoder/hpack_entry_decoder.h"
#include <stddef.h>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include "absl/base/macros.h"
#include "quiche/common/platform/api/quiche_bug_tracker.h"
#include "quiche/common/platform/api/quiche_flag_utils.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
namespace {
class NameDecoderListener {
 public:
  explicit NameDecoderListener(HpackEntryDecoderListener* listener)
      : listener_(listener) {}
  bool OnStringStart(bool huffman_encoded, size_t len) {
    listener_->OnNameStart(huffman_encoded, len);
    return true;
  }
  void OnStringData(const char* data, size_t len) {
    listener_->OnNameData(data, len);
  }
  void OnStringEnd() { listener_->OnNameEnd(); }
 private:
  HpackEntryDecoderListener* listener_;
};
class ValueDecoderListener {
 public:
  explicit ValueDecoderListener(HpackEntryDecoderListener* listener)
      : listener_(listener) {}
  bool OnStringStart(bool huffman_encoded, size_t len) {
    listener_->OnValueStart(huffman_encoded, len);
    return true;
  }
  void OnStringData(const char* data, size_t len) {
    listener_->OnValueData(data, len);
  }
  void OnStringEnd() { listener_->OnValueEnd(); }
 private:
  HpackEntryDecoderListener* listener_;
};
}  
DecodeStatus HpackEntryDecoder::Start(DecodeBuffer* db,
                                      HpackEntryDecoderListener* listener) {
  QUICHE_DCHECK(db != nullptr);
  QUICHE_DCHECK(listener != nullptr);
  QUICHE_DCHECK(db->HasData());
  DecodeStatus status = entry_type_decoder_.Start(db);
  switch (status) {
    case DecodeStatus::kDecodeDone:
      if (entry_type_decoder_.entry_type() == HpackEntryType::kIndexedHeader) {
        listener->OnIndexedHeader(entry_type_decoder_.varint());
        return DecodeStatus::kDecodeDone;
      }
      state_ = EntryDecoderState::kDecodedType;
      return Resume(db, listener);
    case DecodeStatus::kDecodeInProgress:
      QUICHE_DCHECK_EQ(0u, db->Remaining());
      state_ = EntryDecoderState::kResumeDecodingType;
      return status;
    case DecodeStatus::kDecodeError:
      QUICHE_CODE_COUNT_N(decompress_failure_3, 11, 23);
      error_ = HpackDecodingError::kIndexVarintError;
      return status;
  }
  QUICHE_BUG(http2_bug_63_1) << "Unreachable";
  return DecodeStatus::kDecodeError;
}
DecodeStatus HpackEntryDecoder::Resume(DecodeBuffer* db,
                                       HpackEntryDecoderListener* listener) {
  QUICHE_DCHECK(db != nullptr);
  QUICHE_DCHECK(listener != nullptr);
  DecodeStatus status;
  do {
    switch (state_) {
      case EntryDecoderState::kResumeDecodingType:
        QUICHE_DVLOG(1) << "kResumeDecodingType: db->Remaining="
                        << db->Remaining();
        status = entry_type_decoder_.Resume(db);
        if (status == DecodeStatus::kDecodeError) {
          QUICHE_CODE_COUNT_N(decompress_failure_3, 12, 23);
          error_ = HpackDecodingError::kIndexVarintError;
        }
        if (status != DecodeStatus::kDecodeDone) {
          return status;
        }
        state_ = EntryDecoderState::kDecodedType;
        ABSL_FALLTHROUGH_INTENDED;
      case EntryDecoderState::kDecodedType:
        QUICHE_DVLOG(1) << "kDecodedType: db->Remaining=" << db->Remaining();
        if (DispatchOnType(listener)) {
          return DecodeStatus::kDecodeDone;
        }
        continue;
      case EntryDecoderState::kStartDecodingName:
        QUICHE_DVLOG(1) << "kStartDecodingName: db->Remaining="
                        << db->Remaining();
        {
          NameDecoderListener ncb(listener);
          status = string_decoder_.Start(db, &ncb);
        }
        if (status != DecodeStatus::kDecodeDone) {
          state_ = EntryDecoderState::kResumeDecodingName;
          if (status == DecodeStatus::kDecodeError) {
            QUICHE_CODE_COUNT_N(decompress_failure_3, 13, 23);
            error_ = HpackDecodingError::kNameLengthVarintError;
          }
          return status;
        }
        state_ = EntryDecoderState::kStartDecodingValue;
        ABSL_FALLTHROUGH_INTENDED;
      case EntryDecoderState::kStartDecodingValue:
        QUICHE_DVLOG(1) << "kStartDecodingValue: db->Remaining="
                        << db->Remaining();
        {
          ValueDecoderListener vcb(listener);
          status = string_decoder_.Start(db, &vcb);
        }
        if (status == DecodeStatus::kDecodeError) {
          QUICHE_CODE_COUNT_N(decompress_failure_3, 14, 23);
          error_ = HpackDecodingError::kValueLengthVarintError;
        }
        if (status == DecodeStatus::kDecodeDone) {
          return status;
        }
        state_ = EntryDecoderState::kResumeDecodingValue;
        return status;
      case EntryDecoderState::kResumeDecodingName:
        QUICHE_DVLOG(1) << "kResumeDecodingName: db->Remaining="
                        << db->Remaining();
        {
          NameDecoderListener ncb(listener);
          status = string_decoder_.Resume(db, &ncb);
        }
        if (status != DecodeStatus::kDecodeDone) {
          state_ = EntryDecoderState::kResumeDecodingName;
          if (status == DecodeStatus::kDecodeError) {
            QUICHE_CODE_COUNT_N(decompress_failure_3, 15, 23);
            error_ = HpackDecodingError::kNameLengthVarintError;
          }
          return status;
        }
        state_ = EntryDecoderState::kStartDecodingValue;
        break;
      case EntryDecoderState::kResumeDecodingValue:
        QUICHE_DVLOG(1) << "kResumeDecodingValue: db->Remaining="
                        << db->Remaining();
        {
          ValueDecoderListener vcb(listener);
          status = string_decoder_.Resume(db, &vcb);
        }
        if (status == DecodeStatus::kDecodeError) {
          QUICHE_CODE_COUNT_N(decompress_failure_3, 16, 23);
          error_ = HpackDecodingError::kValueLengthVarintError;
        }
        if (status == DecodeStatus::kDecodeDone) {
          return status;
        }
        state_ = EntryDecoderState::kResumeDecodingValue;
        return status;
    }
  } while (true);
}
bool HpackEntryDecoder::DispatchOnType(HpackEntryDecoderListener* listener) {
  const HpackEntryType entry_type = entry_type_decoder_.entry_type();
  const uint32_t varint = static_cast<uint32_t>(entry_type_decoder_.varint());
  switch (entry_type) {
    case HpackEntryType::kIndexedHeader:
      listener->OnIndexedHeader(varint);
      return true;
    case HpackEntryType::kIndexedLiteralHeader:
    case HpackEntryType::kUnindexedLiteralHeader:
    case HpackEntryType::kNeverIndexedLiteralHeader:
      listener->OnStartLiteralHeader(entry_type, varint);
      if (varint == 0) {
        state_ = EntryDecoderState::kStartDecodingName;
      } else {
        state_ = EntryDecoderState::kStartDecodingValue;
      }
      return false;
    case HpackEntryType::kDynamicTableSizeUpdate:
      listener->OnDynamicTableSizeUpdate(varint);
      return true;
  }
  QUICHE_BUG(http2_bug_63_2) << "Unreachable, entry_type=" << entry_type;
  return true;
}
void HpackEntryDecoder::OutputDebugString(std::ostream& out) const {
  out << "HpackEntryDecoder(state=" << state_ << ", " << entry_type_decoder_
      << ", " << string_decoder_ << ")";
}
std::string HpackEntryDecoder::DebugString() const {
  std::stringstream s;
  s << *this;
  return s.str();
}
std::ostream& operator<<(std::ostream& out, const HpackEntryDecoder& v) {
  v.OutputDebugString(out);
  return out;
}
std::ostream& operator<<(std::ostream& out,
                         HpackEntryDecoder::EntryDecoderState state) {
  typedef HpackEntryDecoder::EntryDecoderState EntryDecoderState;
  switch (state) {
    case EntryDecoderState::kResumeDecodingType:
      return out << "kResumeDecodingType";
    case EntryDecoderState::kDecodedType:
      return out << "kDecodedType";
    case EntryDecoderState::kStartDecodingName:
      return out << "kStartDecodingName";
    case EntryDecoderState::kResumeDecodingName:
      return out << "kResumeDecodingName";
    case EntryDecoderState::kStartDecodingValue:
      return out << "kStartDecodingValue";
    case EntryDecoderState::kResumeDecodingValue:
      return out << "kResumeDecodingValue";
  }
  return out << static_cast<int>(state);
}
}  