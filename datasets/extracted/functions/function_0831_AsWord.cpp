#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include <cstdint>
#include <memory>
#include <string>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"
namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
namespace {
using MarkerId = uint16_t;
void AsWord(int value, char* msb, char* lsb) {
  *msb = static_cast<char>(value >> 8);
  *lsb = static_cast<char>(value);
}
class JfifHeaderParser {
 public:
  explicit JfifHeaderParser(const tflite::StringRef& jpeg_image_data)
      : jpeg_image_data_(jpeg_image_data), offset_(0) {
    if (!IsJpegImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Not a valid JPEG image.";
    } else if (!IsJfifImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Image is not in JFIF format.";
      return;
    } else {
      is_valid_image_buffer_ = true;
    }
  }
#define ENSURE_READ_STATUS(a)                           \
  do {                                                  \
    const TfLiteStatus s = (a);                         \
    if (s != kTfLiteOk) {                               \
      return {s, "Error trying to parse JPEG header."}; \
    }                                                   \
  } while (0)
  Status ReadJpegHeader(JpegHeader* result) {
    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }
    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }
    ENSURE_READ_STATUS(SkipBytes(2));  
    char precision;
    ENSURE_READ_STATUS(ReadByte(&precision));
    uint16_t height;
    ENSURE_READ_STATUS(ReadWord(&height));
    uint16_t width;
    ENSURE_READ_STATUS(ReadWord(&width));
    char num_of_components;
    ENSURE_READ_STATUS(ReadByte(&num_of_components));
    if (num_of_components != 1 && num_of_components != 3) {
      return {kTfLiteError,
              "A JFIF image without App14 marker doesn't support a number of "
              "components = " +
                  std::to_string(static_cast<int>(num_of_components))};
    }
    result->width = width;
    result->height = height;
    result->channels = num_of_components;
    result->bits_per_sample = precision;
    return {kTfLiteOk, ""};
  }
  Status ApplyHeaderToImage(const JpegHeader& new_header,
                            std::string& write_to) {
    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }
    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }
    ENSURE_READ_STATUS(SkipBytes(2));  
    if (!HasData(6)) {
      return {kTfLiteError,
              "Invalid SOF marker, image buffer ends before end of marker"};
    }
    char header[6];
    header[0] = static_cast<char>(new_header.bits_per_sample);
    AsWord(new_header.height, header + 1, header + 2);
    AsWord(new_header.width, header + 3, header + 4);
    header[5] = static_cast<char>(new_header.channels);
    write_to.clear();
    write_to.append(jpeg_image_data_.str, offset_);
    write_to.append(header, 6);
    ENSURE_READ_STATUS(SkipBytes(6));
    if (HasData()) {
      write_to.append(jpeg_image_data_.str + offset_,
                      jpeg_image_data_.len - offset_);
    }
    return {kTfLiteOk, ""};
  }
 private:
  const tflite::StringRef jpeg_image_data_;
  int offset_;
  bool is_valid_image_buffer_;
  std::string validation_error_message_;
  Status MoveToStartOfFrameMarker() {
    const MarkerId kStartOfStreamMarkerId = 0xFFDA;  
    offset_ = 0;
    ENSURE_READ_STATUS(SkipBytes(4));  
    ENSURE_READ_STATUS(SkipCurrentMarker());  
    MarkerId curr_marker_id;
    while (HasData(4)) {
      ENSURE_READ_STATUS(ReadWord(&curr_marker_id));
      if (IsStartOfFrameMarkerId(curr_marker_id)) {
        break;
      }
      if (curr_marker_id == kStartOfStreamMarkerId) {
        return {kTfLiteError, "Error trying to parse JPEG header."};
      }
      ENSURE_READ_STATUS(SkipCurrentMarker());
    }
    return {kTfLiteOk, ""};
  }
#undef ENSURE_READ_STATUS
  bool HasData(int min_data_size = 1) {
    return offset_ <= jpeg_image_data_.len - min_data_size;
  }
  TfLiteStatus SkipBytes(int bytes) {
    if (!HasData(bytes)) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Trying to move out of image boundaries from offset %d, "
                 "skipping %d bytes",
                 offset_, bytes);
      return kTfLiteError;
    }
    offset_ += bytes;
    return kTfLiteOk;
  }
  TfLiteStatus ReadByte(char* result) {
    if (!HasData()) {
      return kTfLiteError;
    }
    *result = jpeg_image_data_.str[offset_];
    return SkipBytes(1);
  }
  TfLiteStatus ReadWord(uint16_t* result) {
    TF_LITE_ENSURE_STATUS(ReadWordAt(jpeg_image_data_, offset_, result));
    return SkipBytes(2);
  }
  TfLiteStatus SkipCurrentMarker() {
    uint16_t full_marker_len;
    TF_LITE_ENSURE_STATUS(ReadWord(&full_marker_len));
    if (full_marker_len <= 2) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid marker length %d read at offset %X", full_marker_len,
                 offset_);
      return kTfLiteError;
    }
    return SkipBytes(full_marker_len - 2);
  }
  static TfLiteStatus ReadWordAt(const tflite::StringRef& jpeg_image_data,
                                 int read_offset, uint16_t* result) {
    if (read_offset < 0 || read_offset + 2 > jpeg_image_data.len) {
      return kTfLiteError;
    }
    const unsigned char* buf =
        reinterpret_cast<const unsigned char*>(jpeg_image_data.str);
    *result = (buf[read_offset] << 8) + buf[read_offset + 1];
    return kTfLiteOk;
  }
  static bool IsJpegImage(const tflite::StringRef& jpeg_image_data) {
    const MarkerId kStartOfImageMarkerId = 0xFFD8;
    const MarkerId kEndOfImageMarkerId = 0xFFD9;
    MarkerId soi_marker_id;
    MarkerId eoi_marker_id;
    if (ReadWordAt(jpeg_image_data, 0, &soi_marker_id) != kTfLiteOk) {
      return false;
    }
    if (ReadWordAt(jpeg_image_data, jpeg_image_data.len - 2, &eoi_marker_id) !=
        kTfLiteOk) {
      return false;
    }
    return (soi_marker_id == kStartOfImageMarkerId) &&
           (eoi_marker_id == kEndOfImageMarkerId);
  }
  static bool IsJfifImage(const tflite::StringRef& jpeg_image_data) {
    const MarkerId kApp0MarkerId = 0xFFE0;  
    MarkerId app_marker_id;
    if ((ReadWordAt(jpeg_image_data, 2, &app_marker_id) != kTfLiteOk) ||
        (app_marker_id != kApp0MarkerId)) {
      return false;
    }
    const std::string kJfifIdString{"JFIF\0", 5};
    const int KJfifIdStringStartOffset = 6;
    if (KJfifIdStringStartOffset + kJfifIdString.size() >=
        jpeg_image_data.len) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid image, reached end of data at offset while "
                 "parsing APP0 header");
      return false;
    }
    const std::string actualImgId(
        jpeg_image_data.str + KJfifIdStringStartOffset, kJfifIdString.size());
    if (kJfifIdString != actualImgId) {
      TFLITE_LOG(TFLITE_LOG_WARNING, "Invalid image, invalid APP0 header");
      return false;
    }
    return true;
  }
  static bool IsStartOfFrameMarkerId(MarkerId marker_id) {
    return 0xFFC0 <= marker_id && marker_id < 0xFFCF;
  }
};
}  
Status ReadJpegHeader(const tflite::StringRef& jpeg_image_data,
                      JpegHeader* header) {
  JfifHeaderParser parser(jpeg_image_data);
  return parser.ReadJpegHeader(header);
}
Status BuildImageWithNewHeader(const tflite::StringRef& orig_jpeg_image_data,
                               const JpegHeader& new_header,
                               std::string& new_image_data) {
  JfifHeaderParser parser(orig_jpeg_image_data);
  return parser.ApplyHeaderToImage(new_header, new_image_data);
}
}  
}  
}  