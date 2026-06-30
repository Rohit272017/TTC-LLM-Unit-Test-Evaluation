#include "tensorstore/internal/compression/zip_details.h"
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <ios>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <variant>
#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/prefix_limiting_reader.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bzip2/bzip2_reader.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/xz/xz_reader.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zstd/zstd_reader.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/riegeli/find.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal_zip {
namespace {
using ::riegeli::ReadLittleEndian16;
using ::riegeli::ReadLittleEndian32;
using ::riegeli::ReadLittleEndian64;
using ::riegeli::ReadLittleEndianSigned64;
ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip_details");
const absl::Time kWindowsEpoch =
    ::absl::UnixEpoch() - ::absl::Seconds(11644473600);
absl::Time MakeMSDOSTime(uint16_t date, uint16_t time) {
  struct tm dos_tm;
  dos_tm.tm_mday = (uint16_t)(date & 0x1f);
  dos_tm.tm_mon = (uint16_t)((date >> 5) & 0xf) - 1;
  dos_tm.tm_year = (uint16_t)(date >> 9) + 80;
  dos_tm.tm_hour = (uint16_t)(time >> 11);
  dos_tm.tm_min = (uint16_t)((time >> 5) & 0x1f);
  dos_tm.tm_sec = (uint16_t)(2 * (time & 0x1f));
  dos_tm.tm_isdst = -1;
  return absl::FromTM(dos_tm, absl::UTCTimeZone());
}
absl::Status ReadExtraField_Zip64_0001(riegeli::Reader &reader,
                                       uint16_t tag_size, ZipEntry &entry) {
  assert(tag_size >= 8);
  entry.is_zip64 = true;
  do {
    if (tag_size >= 8 &&
        entry.uncompressed_size == std::numeric_limits<uint32_t>::max()) {
      if (!ReadLittleEndian64(reader, entry.uncompressed_size)) break;
      tag_size -= 8;
    }
    if (tag_size >= 8 &&
        entry.compressed_size == std::numeric_limits<uint32_t>::max()) {
      if (!ReadLittleEndian64(reader, entry.compressed_size)) break;
      tag_size -= 8;
    }
    if (tag_size >= 8 &&
        entry.local_header_offset == std::numeric_limits<uint32_t>::max()) {
      if (!ReadLittleEndian64(reader, entry.local_header_offset)) break;
      tag_size -= 8;
    }
    return absl::OkStatus();
  } while (false);
  return absl::InvalidArgumentError("Failed to read ZIP64 extra field");
}
absl::Status ReadExtraField_Unix_000D(riegeli::Reader &reader,
                                      uint16_t tag_size, ZipEntry &entry) {
  assert(tag_size >= 12);
  uint32_t ignored32;
  uint32_t mtime;
  uint32_t atime;
  if (!ReadLittleEndian32(reader, atime) ||
      !ReadLittleEndian32(reader, mtime) ||
      !ReadLittleEndian32(reader, ignored32) ) {
    return absl::InvalidArgumentError("Failed to read UNIX extra field");
  }
  entry.atime = absl::FromUnixSeconds(atime);
  entry.mtime = absl::FromUnixSeconds(mtime);
  return absl::OkStatus();
}
absl::Status ReadExtraField_NTFS_000A(riegeli::Reader &reader,
                                      uint16_t tag_size, ZipEntry &entry) {
  assert(tag_size >= 8);
  uint32_t ignored32;
  if (!ReadLittleEndian32(reader, ignored32)) {
    return absl::InvalidArgumentError("Failed to read NTFS extra field");
  }
  tag_size -= 4;
  uint16_t ntfs_tag, ntfs_size;
  while (tag_size > 4) {
    if (!ReadLittleEndian16(reader, ntfs_tag) ||
        !ReadLittleEndian16(reader, ntfs_size)) {
      break;
    }
    tag_size -= 4;
    tag_size -= ntfs_size;
    if (ntfs_tag == 0x0001 && ntfs_size == 24) {
      uint64_t mtime;
      uint64_t atime;
      uint64_t ctime;
      if (!ReadLittleEndian64(reader, mtime) ||
          !ReadLittleEndian64(reader, atime) ||
          !ReadLittleEndian64(reader, ctime)) {
        return absl::InvalidArgumentError("Failed to read NTFS extra field");
      }
      entry.mtime = kWindowsEpoch + absl::Nanoseconds(mtime * 100);
      entry.atime = kWindowsEpoch + absl::Nanoseconds(atime * 100);
    } else {
      reader.Skip(ntfs_size);
    }
  }
  return absl::OkStatus();
}
absl::Status ReadExtraField_Unix_5455(riegeli::Reader &reader,
                                      uint16_t tag_size, ZipEntry &entry) {
  assert(tag_size >= 1);
  uint8_t flags = 0;
  uint32_t tstamp = 0;
  do {
    if (!reader.ReadByte(flags)) break;
    --tag_size;
    if (flags & 0x01 && tag_size >= 4) {  
      if (!ReadLittleEndian32(reader, tstamp)) break;
      tag_size -= 4;
      entry.mtime = absl::FromUnixSeconds(tstamp);
    }
    if (flags & 0x02 && tag_size >= 4) {  
      if (!ReadLittleEndian32(reader, tstamp)) break;
      tag_size -= 4;
      entry.atime = absl::FromUnixSeconds(tstamp);
    }
    if (flags & 0x04 && tag_size >= 4) {  
      if (!ReadLittleEndian32(reader, tstamp)) break;
      tag_size -= 4;
    }
    return absl::OkStatus();
  } while (false);
  return absl::InvalidArgumentError(
      "Failed to read unix timestamp extra field");
}
absl::Status ReadExtraField(riegeli::Reader &reader, ZipEntry &entry) {
  uint16_t tag, tag_size;
  absl::Status status;
  while (reader.ok()) {
    if (!ReadLittleEndian16(reader, tag) ||
        !ReadLittleEndian16(reader, tag_size)) {
      return absl::OkStatus();
    }
    ABSL_LOG_IF(INFO, zip_logging)
        << std::hex << "extra tag " << tag << " size " << tag_size;
    auto pos = reader.pos();
    switch (tag) {
      case 0x0001:  
        status.Update(ReadExtraField_Zip64_0001(reader, tag_size, entry));
        break;
      case 0x000d:  
        status.Update(ReadExtraField_Unix_000D(reader, tag_size, entry));
        break;
      case 0x000a:  
        status.Update(ReadExtraField_NTFS_000A(reader, tag_size, entry));
        break;
      case 0x5455:  
        status.Update(ReadExtraField_Unix_5455(reader, tag_size, entry));
        break;
      case 0x7875:  
        break;
      default:
        break;
    }
    assert(reader.pos() <= pos + tag_size);
    reader.Seek(pos + tag_size);
  }
  return status;
}
}  
absl::Status ReadEOCD64Locator(riegeli::Reader &reader,
                               ZipEOCD64Locator &locator) {
  if (!reader.Pull(ZipEOCD64Locator::kRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD64 Locator Entry insufficient data available");
  }
  uint32_t signature;
  ReadLittleEndian32(reader, signature);
  if (signature != 0x07064b50) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to read ZIP64 End of Central Directory Locator signature %08x",
        signature));
  }
  uint32_t ignored32;
  ReadLittleEndian32(reader, locator.disk_number_with_cd);
  ReadLittleEndianSigned64(reader, locator.cd_offset);
  ReadLittleEndian32(reader, ignored32);
  if (locator.cd_offset < 0) {
    ABSL_LOG_IF(INFO, zip_logging && !reader.ok()) << reader.status();
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory Locator");
  }
  return absl::OkStatus();
}
absl::Status ReadEOCD64(riegeli::Reader &reader, ZipEOCD &eocd) {
  if (!reader.Pull(ZipEOCD::kEOCD64RecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD Entry insufficient data available");
  }
  auto eocd_pos = reader.pos();
  uint32_t signature;
  ReadLittleEndian32(reader, signature);
  if (signature != 0x06064b50) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 Central Directory Entry signature");
  }
  uint64_t eocd_size;
  ReadLittleEndian64(reader, eocd_size);
  if (eocd_size < 44 || !reader.Pull(eocd_size)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory");
  }
  riegeli::LimitingReader oecd64_reader(
      &reader,
      riegeli::LimitingReaderBase::Options().set_exact_length(eocd_size));
  uint16_t version_madeby;
  uint16_t version_needed_to_extract;
  uint32_t disk_number;
  uint32_t disk_number_with_cd;
  uint64_t total_num_entries;
  ReadLittleEndian16(oecd64_reader, version_madeby);
  ReadLittleEndian16(oecd64_reader, version_needed_to_extract);
  ReadLittleEndian32(oecd64_reader, disk_number);
  ReadLittleEndian32(oecd64_reader, disk_number_with_cd);
  ReadLittleEndian64(oecd64_reader, eocd.num_entries);
  ReadLittleEndian64(oecd64_reader, total_num_entries);
  ReadLittleEndianSigned64(oecd64_reader, eocd.cd_size);
  ReadLittleEndianSigned64(oecd64_reader, eocd.cd_offset);
  if (disk_number != disk_number_with_cd ||
      eocd.num_entries != total_num_entries ||
      eocd.num_entries == std::numeric_limits<uint16_t>::max() ||
      eocd.cd_size == std::numeric_limits<uint16_t>::max() ||
      eocd.cd_offset == std::numeric_limits<uint32_t>::max() ||
      eocd.cd_size < 0 || eocd.cd_offset < 0) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory");
  }
  oecd64_reader.Seek(eocd_size);
  eocd.record_offset = eocd_pos;
  return absl::OkStatus();
}
absl::Status ReadEOCD(riegeli::Reader &reader, ZipEOCD &eocd) {
  if (!reader.Pull(ZipEOCD::kEOCDRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD Entry insufficient data available");
  }
  auto eocd_pos = reader.pos();
  uint32_t signature;
  ReadLittleEndian32(reader, signature);
  if (signature != 0x06054b50) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry signature");
  }
  uint16_t disk_number;
  uint16_t disk_number_with_cd;
  uint16_t num_entries;
  uint16_t total_num_entries;
  uint32_t cd_size;
  uint32_t cd_offset;
  uint16_t comment_length;
  ReadLittleEndian16(reader, disk_number);
  ReadLittleEndian16(reader, disk_number_with_cd);
  ReadLittleEndian16(reader, num_entries);
  ReadLittleEndian16(reader, total_num_entries);
  ReadLittleEndian32(reader, cd_size);
  ReadLittleEndian32(reader, cd_offset);
  ReadLittleEndian16(reader, comment_length);
  if (num_entries != total_num_entries) {
    ABSL_LOG(INFO) << "ZIP num_entries mismatch " << num_entries << " vs "
                   << total_num_entries;
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  if (disk_number != disk_number_with_cd) {
    ABSL_LOG(INFO) << "ZIP disk_number mismatch " << disk_number << " vs "
                   << disk_number_with_cd;
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  if (comment_length > 0 && !reader.Read(comment_length, eocd.comment)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  reader.VerifyEnd();
  if (!reader.status().ok()) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  eocd.record_offset = eocd_pos;
  eocd.num_entries = num_entries;
  eocd.cd_size = cd_size;
  eocd.cd_offset = cd_offset;
  if (total_num_entries == std::numeric_limits<uint16_t>::max() ||
      cd_offset == std::numeric_limits<uint32_t>::max()) {
    eocd.cd_offset = std::numeric_limits<uint32_t>::max();
  }
  return absl::OkStatus();
}
std::variant<absl::Status, int64_t> TryReadFullEOCD(riegeli::Reader &reader,
                                                    ZipEOCD &eocd,
                                                    int64_t offset_adjustment) {
  if (!internal::FindLast(
          reader, std::string_view(reinterpret_cast<const char *>(kEOCDLiteral),
                                   sizeof(kEOCDLiteral)))) {
    return absl::InvalidArgumentError("Failed to find valid ZIP EOCD");
  }
  int64_t eocd_start = reader.pos();
  ZipEOCD last_eocd{};
  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD(reader, last_eocd));
  if (last_eocd.cd_offset != std::numeric_limits<uint32_t>::max()) {
    eocd = last_eocd;
    reader.Seek(eocd_start + 4);
    return absl::OkStatus();
  }
  if (eocd_start < ZipEOCD64Locator::kRecordSize) {
    return absl::InvalidArgumentError("Block does not contain EOCD64 Locator");
  }
  if (!reader.Seek(eocd_start - ZipEOCD64Locator::kRecordSize)) {
    if (!reader.ok() && !reader.status().ok()) {
      return MaybeAnnotateStatus(reader.status(),
                                 "Failed to read EOCD64 Locator");
    }
    return absl::InvalidArgumentError("Failed to read EOCD64 Locator");
  }
  ZipEOCD64Locator locator;
  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD64Locator(reader, locator));
  if (offset_adjustment < 0) {
    return locator.cd_offset;
  }
  auto target_pos = locator.cd_offset - offset_adjustment;
  if (target_pos < 0) {
    assert(offset_adjustment > 0);
    return locator.cd_offset;
  }
  if (!reader.Seek(target_pos)) {
    if (!reader.ok() && !reader.status().ok()) {
      return MaybeAnnotateStatus(reader.status(), "Failed to read EOCD64");
    }
    return absl::InvalidArgumentError("Failed to read EOCD64");
  }
  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD64(reader, last_eocd));
  eocd = last_eocd;
  reader.Seek(eocd_start + 4);
  return absl::OkStatus();
}
absl::Status ReadCentralDirectoryEntry(riegeli::Reader &reader,
                                       ZipEntry &entry) {
  if (!reader.Pull(ZipEntry::kCentralRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP Central Directory Entry insufficient data available");
  }
  uint32_t signature;
  ReadLittleEndian32(reader, signature);
  if (signature != 0x02014b50) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry signature");
  }
  uint32_t uncompressed_size = 0;
  uint32_t compressed_size;
  uint32_t relative_header_offset = 0;
  uint16_t file_name_length = 0;
  uint16_t extra_field_length = 0;
  uint16_t file_comment_length = 0;
  uint16_t last_mod_time;
  uint16_t last_mod_date;
  uint16_t ignored16;
  uint16_t compression_method;
  ReadLittleEndian16(reader, entry.version_madeby);
  ReadLittleEndian16(reader, ignored16);  
  ReadLittleEndian16(reader, entry.flags);
  ReadLittleEndian16(reader, compression_method);
  ReadLittleEndian16(reader, last_mod_time);
  ReadLittleEndian16(reader, last_mod_date);
  ReadLittleEndian32(reader, entry.crc);
  ReadLittleEndian32(reader, compressed_size);
  ReadLittleEndian32(reader, uncompressed_size);
  ReadLittleEndian16(reader, file_name_length);
  ReadLittleEndian16(reader, extra_field_length);
  ReadLittleEndian16(reader, file_comment_length);
  ReadLittleEndian16(reader, ignored16);  
  ReadLittleEndian16(reader, entry.internal_fa);
  ReadLittleEndian32(reader, entry.external_fa);
  ReadLittleEndian32(reader, relative_header_offset);
  entry.compressed_size = compressed_size;
  entry.uncompressed_size = uncompressed_size;
  entry.local_header_offset = relative_header_offset;
  entry.mtime = MakeMSDOSTime(last_mod_date, last_mod_time);
  entry.compression_method = static_cast<ZipCompression>(compression_method);
  if (file_name_length > 0 && !reader.Read(file_name_length, entry.filename)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry (filename)");
  }
  assert(entry.filename.size() == file_name_length);
  if (extra_field_length > 0) {
    assert(extra_field_length > 4);
    riegeli::LimitingReader extra_reader(
        &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                     extra_field_length));
    extra_reader.SetReadAllHint(true);
    if (auto status = ReadExtraField(extra_reader, entry); !status.ok()) {
      return status;
    }
    extra_reader.Seek(extra_field_length);
  }
  if (file_comment_length > 0 &&
      !reader.Read(file_comment_length, entry.comment)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry (comment)");
  }
  entry.end_of_header_offset = reader.pos();
  entry.estimated_read_size =
      std::max(entry.compressed_size, entry.uncompressed_size) +
      file_name_length + extra_field_length + ZipEntry::kLocalRecordSize +
       (entry.flags & kHasDataDescriptor ? 12 : 0);
  return absl::OkStatus();
}
absl::Status ReadLocalEntry(riegeli::Reader &reader, ZipEntry &entry) {
  if (!reader.Pull(ZipEntry::kLocalRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP Local Entry insufficient data available");
  }
  uint32_t signature;
  ReadLittleEndian32(reader, signature);
  if (signature != 0x04034b50) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Local Entry signature");
  }
  uint16_t ignored16;
  uint16_t compression_method;
  uint16_t last_mod_time;
  uint16_t last_mod_date;
  uint32_t uncompressed_size;
  uint32_t compressed_size;
  uint16_t file_name_length = 0;
  uint16_t extra_field_length = 0;
  ReadLittleEndian16(reader, ignored16);  
  ReadLittleEndian16(reader, entry.flags);
  ReadLittleEndian16(reader, compression_method);
  ReadLittleEndian16(reader, last_mod_time);
  ReadLittleEndian16(reader, last_mod_date);
  ReadLittleEndian32(reader, entry.crc);
  ReadLittleEndian32(reader, compressed_size);
  ReadLittleEndian32(reader, uncompressed_size);
  ReadLittleEndian16(reader, file_name_length);
  ReadLittleEndian16(reader, extra_field_length);
  entry.version_madeby = 0;
  entry.internal_fa = 0;
  entry.external_fa = 0;
  entry.local_header_offset = 0;
  entry.estimated_read_size = 0;
  entry.compressed_size = compressed_size;
  entry.uncompressed_size = uncompressed_size;
  entry.mtime = MakeMSDOSTime(last_mod_date, last_mod_time);
  entry.compression_method = static_cast<ZipCompression>(compression_method);
  if (file_name_length > 0 && !reader.Read(file_name_length, entry.filename)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Local Entry (filename)");
  }
  assert(entry.filename.size() == file_name_length);
  entry.end_of_header_offset = reader.pos() + extra_field_length;
  if (extra_field_length > 0) {
    assert(extra_field_length > 4);
    riegeli::LimitingReader extra_reader(
        &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                     extra_field_length));
    extra_reader.SetReadAllHint(true);
    if (auto status = ReadExtraField(extra_reader, entry); !status.ok()) {
      return status;
    }
    extra_reader.Seek(extra_field_length);
  }
  return absl::OkStatus();
}
absl::Status ValidateEntryIsSupported(const ZipEntry &entry) {
  if (entry.flags & 0x01 ||                 
      entry.flags & (uint16_t{1} << 6) ||   
      entry.flags & (uint16_t{1} << 13) ||  
      entry.compression_method == ZipCompression::kAes) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("ZIP encryption is not supported"));
  }
  if (entry.compression_method != ZipCompression::kStore &&
      entry.compression_method != ZipCompression::kDeflate &&
      entry.compression_method != ZipCompression::kBzip2 &&
      entry.compression_method != ZipCompression::kZStd &&
      entry.compression_method != ZipCompression::kXZ) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("ZIP compression method ", entry.compression_method,
                            " is not supported"));
  }
  if (absl::EndsWith(entry.filename, "/")) {
    return absl::InvalidArgumentError("ZIP directory entries cannot be read");
  }
  return absl::OkStatus();
}
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetRawReader(
    riegeli::Reader *reader, ZipEntry &entry) {
  assert(reader != nullptr);
  if (entry.flags & kHasDataDescriptor) {
    const auto start_pos = reader->pos();
    if (!reader->Skip(entry.compressed_size)) {
      return reader->status();
    }
    static constexpr size_t kZipDataDescriptorSize = 16;
    static constexpr size_t kZip64DataDescriptorSize = 24;
    if (!reader->Pull(entry.is_zip64 ? kZip64DataDescriptorSize
                                     : kZipDataDescriptorSize)) {
      return absl::DataLossError("Failed to read ZIP DataDescriptor");
    }
    uint32_t signature, crc32;
    ReadLittleEndian32(*reader, signature);
    ReadLittleEndian32(*reader, crc32);
    if (signature != 0x08074b50) {
      return absl::DataLossError(absl::StrFormat(
          "Failed to read ZIP DataDescriptor signature %08x", signature));
    }
    if (entry.crc == 0) entry.crc = crc32;
    if (entry.is_zip64) {
      uint64_t compressed_size, uncompressed_size;
      ReadLittleEndian64(*reader, compressed_size);
      ReadLittleEndian64(*reader, uncompressed_size);
      if (entry.compressed_size == 0) entry.compressed_size = compressed_size;
      if (entry.uncompressed_size == 0)
        entry.uncompressed_size = uncompressed_size;
    } else {
      uint32_t compressed_size, uncompressed_size;
      ReadLittleEndian32(*reader, compressed_size);
      ReadLittleEndian32(*reader, uncompressed_size);
      if (entry.compressed_size == 0) {
        entry.compressed_size = compressed_size;
      }
      if (entry.uncompressed_size == 0) {
        entry.uncompressed_size = uncompressed_size;
      }
    }
    if (!reader->Seek(start_pos)) {
      return reader->status();
    }
  }
  using Reader = riegeli::LimitingReader<riegeli::Reader *>;
  return std::make_unique<Reader>(
      reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                  entry.compressed_size));
}
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetReader(
    riegeli::Reader *reader, ZipEntry &entry) {
  TENSORSTORE_ASSIGN_OR_RETURN(std::unique_ptr<riegeli::Reader> base_reader,
                               GetRawReader(reader, entry));
  switch (entry.compression_method) {
    case ZipCompression::kStore: {
      using PLReader =
          riegeli::PrefixLimitingReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<PLReader>(
          std::move(base_reader),
          PLReader::Options().set_base_pos(reader->pos()));
    }
    case ZipCompression::kDeflate: {
      using DeflateReader =
          riegeli::ZlibReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<DeflateReader>(
          std::move(base_reader),
          DeflateReader::Options().set_header(DeflateReader::Header::kRaw));
    }
    case ZipCompression::kBzip2: {
      using Bzip2Reader =
          riegeli::Bzip2Reader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<Bzip2Reader>(std::move(base_reader));
    }
    case ZipCompression::kZStd: {
      using ZStdReader = riegeli::ZstdReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<ZStdReader>(std::move(base_reader));
    }
    case ZipCompression::kXZ: {
      using XzReader = riegeli::XzReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<XzReader>(
          std::move(base_reader), XzReader::Options()
                                      .set_container(XzReader::Container::kXz)
                                      .set_concatenate(true)
      );
    }
    default:
      break;
  }
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Unsupported ZIP compression method ", entry.compression_method));
}
}  
}  