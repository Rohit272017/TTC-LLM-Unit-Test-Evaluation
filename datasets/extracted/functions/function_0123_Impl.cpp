#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"
#include <setjmp.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_handle.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {
const size_t LibjpegDecoder::kMaxImageHeight = 10000;
const size_t LibjpegDecoder::kMaxImageWidth = 10000;
constexpr char kSizeMismatchError[] =
    "JPEG parameter struct mismatch: library thinks size is ";
LibjpegDecoder::Impl::Impl(size_t decompress_struct_size,
                           const LibjpegHandle* handle)
    : decompress_struct_size_(decompress_struct_size),
      handle_(handle),
      cinfo_(decompress_struct_size) {
  cinfo_.get()->err = handle->jpeg_std_error_(&jerr_);
  jerr_.error_exit = ErrorExit;
  cinfo_.get()->client_data = this;
}
void LibjpegDecoder::Impl::ErrorExit(j_common_ptr cinfo) {
  Impl* const impl = reinterpret_cast<Impl*>(cinfo->client_data);
  char message[JMSG_LENGTH_MAX];
  cinfo->err->format_message(cinfo, message);
  impl->status_.code = kTfLiteError;
  impl->status_.error_message = message;
  longjmp(impl->env_, 1);
}
Status ExtractSizeFromErrorMessage(const std::string& error_message,
                                   size_t& expected_size) {
  Status status;
  static const int kExpLengthStart = strlen(kSizeMismatchError);
  int end = kExpLengthStart;
  while (end < error_message.length() && std::isdigit(error_message[end])) {
    end++;
  }
  if (end > kExpLengthStart) {
    expected_size = std::stoi(error_message.substr(kExpLengthStart, end));
  } else {
    status.code = kTfLiteError;
    status.error_message =
        "Couldn't parse the size from message: \'" + error_message + "\'";
  }
  return status;
}
std::unique_ptr<LibjpegDecoder> LibjpegDecoder::Create(Status& status) {
  std::unique_ptr<LibjpegDecoder> decoder(
      new LibjpegDecoder(LibCHandle::Create(status)));
  if (status.code != kTfLiteOk) {
    return nullptr;
  }
  decoder->libjpeg_handle_ = LibjpegHandle::Create(status);
  if (decoder->libjpeg_handle_ == nullptr) {
    return nullptr;
  }
  Impl impl(sizeof(jpeg_decompress_struct), decoder->libjpeg_handle_.get());
  impl.jpeg_CreateDecompress(LibjpegHandle::kLibjpegVersion,
                             sizeof(jpeg_decompress_struct));
  status = impl.status();
  if (status.code == kTfLiteOk) {
    decoder->expected_size_for_decompress_struct_ =
        sizeof(jpeg_decompress_struct);
    return decoder;
  }
  if (!absl::StrContains(status.error_message, kSizeMismatchError)) {
    return nullptr;
  }
  status = ExtractSizeFromErrorMessage(
      status.error_message, decoder->expected_size_for_decompress_struct_);
  if (status.code != kTfLiteOk) {
    return nullptr;
  }
  return decoder;
}
namespace {
std::string JpegHeaderToString(const JpegHeader& header) {
  return "(" + std::to_string(header.height) + ", " +
         std::to_string(header.width) + ", " + std::to_string(header.channels) +
         ", " + std::to_string(header.bits_per_sample) + ")";
}
}  
Status LibjpegDecoder::DecodeImage(const tflite::StringRef& encoded,
                                   const JpegHeader& expected_image_dimensions,
                                   unsigned char* decoded,
                                   const size_t& decoded_size) const {
  if (expected_image_dimensions.bits_per_sample != 8) {
    return {kTfLiteError, "Supporting only images with 8 bits per sample"};
  }
  if (expected_image_dimensions.channels != 1 &&
      expected_image_dimensions.channels != 3) {
    return {kTfLiteError, "Supporting only images with 1 or 3 channels"};
  }
  if (expected_image_dimensions.width > kMaxImageWidth ||
      expected_image_dimensions.height > kMaxImageHeight) {
    return {kTfLiteError, "Image is too big, dimensions (" +
                              std::to_string(expected_image_dimensions.width) +
                              "," +
                              std::to_string(expected_image_dimensions.width) +
                              ") larger than the maximum allowed (" +
                              std::to_string(kMaxImageWidth) + ", " +
                              std::to_string(kMaxImageHeight) + ")"};
  }
  JpegHeader header;
  Status read_header_status = ReadJpegHeader(encoded, &header);
  if (read_header_status.code != kTfLiteOk) {
    return read_header_status;
  }
  if (expected_image_dimensions.channels != header.channels ||
      expected_image_dimensions.width != header.width ||
      expected_image_dimensions.height != header.height ||
      expected_image_dimensions.bits_per_sample != header.bits_per_sample) {
    return {kTfLiteError, "Decoded image size " + JpegHeaderToString(header) +
                              " is different from provided image size " +
                              JpegHeaderToString(expected_image_dimensions)};
  }
  size_t header_image_size = static_cast<size_t>(header.width) *
                             static_cast<size_t>(header.height) *
                             static_cast<size_t>(header.channels);
  if (header_image_size != decoded_size) {
    return {kTfLiteError, "Size of buffer(" + std::to_string(decoded_size) +
                              ") for storing decoded image must be equal to "
                              "the size of decoded image(" +
                              std::to_string(header_image_size) + ")."};
  }
  char* image_buffer = const_cast<char*>(encoded.str);
  size_t image_size = encoded.len;
  std::unique_ptr<FILE, std::function<void(FILE*)>> file(
      libc_handle_.fmemopen(image_buffer, image_size, "r"),
      [](FILE* f) { fclose(f); });
  if (file == nullptr) {
    return {kTfLiteError, "Fmemopen failed."};
  }
  Impl impl(expected_size_for_decompress_struct_, libjpeg_handle_.get());
  if (impl.jpeg_CreateDecompress(LibjpegHandle::kLibjpegVersion,
                                 expected_size_for_decompress_struct_)) {
    return impl.status();
  }
  if (impl.jpeg_stdio_src(file.get())) {
    return impl.status();
  }
  int read_header_result = 0;
  if (impl.jpeg_read_header(read_header_result, true) != kTfLiteOk) {
    return impl.status();
  }
  if (read_header_result != JPEG_HEADER_OK) {
    return {kTfLiteError, "Failed call jpeg_read_header"};
  }
  boolean start_decompress_result = false;
  if (impl.jpeg_start_decompress(start_decompress_result) != kTfLiteOk) {
    return impl.status();
  }
  if (!start_decompress_result) {
    return {kTfLiteError, "Failed call jpeg_start_decompress_"};
  }
  size_t height = header.height;
  size_t row_stride = header.width * header.channels;
  const size_t kMaxImageSize = JPEG_MAX_DIMENSION * 4;
  std::vector<unsigned char> decode_buffer(kMaxImageSize);
  unsigned char* buffer_array[1];
  buffer_array[0] = decode_buffer.data();
  size_t decoded_offset = 0;
  while (height--) {
    unsigned int num_of_scanlines_read = 0;
    if (impl.jpeg_read_scanlines(num_of_scanlines_read, buffer_array, 1) !=
        kTfLiteOk) {
      return impl.status();
    }
    if (num_of_scanlines_read != 1) {
      return {kTfLiteError, "Expected " + std::to_string(header.height) +
                                " lines but found only " +
                                std::to_string(header.height - height) +
                                " read scanlines is " +
                                std::to_string(num_of_scanlines_read)};
    }
    std::copy_n(buffer_array[0], row_stride, decoded + decoded_offset);
    decoded_offset += row_stride;
  }
  boolean finish_decompress_result = false;
  if (impl.jpeg_finish_decompress(finish_decompress_result) != kTfLiteOk) {
    return impl.status();
  }
  if (!finish_decompress_result) {
    return {kTfLiteError, "Failed call jpeg_finish_decompress_"};
  }
  if (impl.jpeg_destroy_decompress() != kTfLiteOk) {
    return impl.status();
  }
  return impl.status();
}
}  
}  
}  