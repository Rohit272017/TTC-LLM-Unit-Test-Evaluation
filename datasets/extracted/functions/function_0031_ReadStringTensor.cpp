#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/io/buffered_file.h"
#include "xla/tsl/util/byte_swap_array.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/versions.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_tensor.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_slice_util.h"
#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif
namespace tensorflow {
const int kTensorBundleMinProducer = 0;
const int kTensorBundleMinConsumer = 0;
const int kTensorBundleVersion = 1;
static const int kBufferSize = 1024 * 1024;
const char* const kHeaderEntryKey = "";
const int64_t kLargeTensorThreshold = static_cast<int64_t>(1) << 32;
const int kMaxFileReadThreads = 8;
const int64_t kMinSectionSize = static_cast<int64_t>(1) << 31;
namespace {
Status ReadStringTensor(io::InputBuffer* buffered_file, size_t num_elements,
                        size_t offset, size_t size, tstring* destination,
                        uint32* actual_crc32c, bool need_to_swap_bytes) {
  if (size == 0) return absl::OkStatus();
  CHECK_GT(size, 0);
  TF_RETURN_IF_ERROR(buffered_file->Seek(offset));
  TF_RETURN_IF_ERROR(buffered_file->Hint(size));
  std::vector<uint64> string_lengths(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    TF_RETURN_IF_ERROR(buffered_file->ReadVarint64(&string_lengths[i]));
    if (string_lengths[i] <= UINT32_MAX) {
      uint32 elem_size_uint32 = static_cast<uint32>(string_lengths[i]);
      if (need_to_swap_bytes) {
        elem_size_uint32 = BYTE_SWAP_32(elem_size_uint32);
      }
      *actual_crc32c = crc32c::Extend(
          *actual_crc32c, reinterpret_cast<const char*>(&elem_size_uint32),
          sizeof(uint32));
    } else {
      uint64 length = string_lengths[i];
      if (need_to_swap_bytes) {
        length = BYTE_SWAP_64(length);
      }
      *actual_crc32c =
          crc32c::Extend(*actual_crc32c, reinterpret_cast<const char*>(&length),
                         sizeof(uint64));
    }
  }
  if (offset + size < buffered_file->Tell()) {
    return errors::DataLoss("String lengths longer than expected offset ",
                            offset + size);
  }
  uint32 raw_length_checksum = 0;  
  uint32 length_checksum = 0;      
  size_t unused_bytes_read = 0;
  TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(
      sizeof(uint32), reinterpret_cast<char*>(&raw_length_checksum),
      &unused_bytes_read));
  length_checksum = need_to_swap_bytes ? BYTE_SWAP_32(raw_length_checksum)
                                       : raw_length_checksum;
  if (crc32c::Unmask(length_checksum) != *actual_crc32c) {
    return errors::DataLoss(
        "The length checksum does not match: expected ",
        strings::Printf("%08u", crc32c::Unmask(length_checksum)),
        " but actual is ", strings::Printf("%08u", *actual_crc32c));
  }
  *actual_crc32c = crc32c::Extend(*actual_crc32c,
                                  reinterpret_cast<char*>(&raw_length_checksum),
                                  sizeof(uint32));
  for (size_t i = 0; i < num_elements; ++i) {
    const uint64 string_length = string_lengths[i];
    tstring* buffer = &destination[i];
    buffer->resize(string_length);
    size_t bytes_read = 0;
    TF_RETURN_IF_ERROR(
        buffered_file->ReadNBytes(string_length, &(*buffer)[0], &bytes_read));
    *actual_crc32c = crc32c::Extend(*actual_crc32c, buffer->data(), bytes_read);
  }
  return absl::OkStatus();
}
Status ReadVariantTensor(io::InputBuffer* buffered_file, Tensor* ret,
                         size_t offset, size_t size, uint32* actual_crc32c) {
  if (size == 0) return absl::OkStatus();
  size_t num_elements = ret->NumElements();
  TF_RETURN_IF_ERROR(buffered_file->Seek(offset));
  TF_RETURN_IF_ERROR(buffered_file->Hint(size));
  for (size_t i = 0; i < num_elements; ++i) {
    uint64 string_length = 0;
    TF_RETURN_IF_ERROR(buffered_file->ReadVarint64(&string_length));
    *actual_crc32c = crc32c::Extend(
        *actual_crc32c, reinterpret_cast<const char*>(&string_length),
        sizeof(uint64));
    string buffer;
    buffer.resize(string_length);
    size_t bytes_read = 0;
    TF_RETURN_IF_ERROR(
        buffered_file->ReadNBytes(string_length, &buffer[0], &bytes_read));
    *actual_crc32c = crc32c::Extend(*actual_crc32c, buffer.data(), bytes_read);
    VariantTensorDataProto proto;
    if (!proto.ParseFromString(buffer)) {
      return errors::DataLoss("Unable to parse VariantTensorDataProto from ",
                              "buffer of size ", string_length, ". ",
                              "Bundle entry offset: ", offset, " size: ", size);
    }
    Variant v = proto;
    if (!DecodeUnaryVariant(&v)) {
      return errors::Internal("Could not decode variant with type_name: \"",
                              v.TypeName(), "\".  Perhaps you forgot to ",
                              "register a decoder via ",
                              "REGISTER_UNARY_VARIANT_DECODE_FUNCTION?");
    }
    uint32 checksum = 0;
    size_t unused_bytes_read = 0;
    TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(
        sizeof(uint32), reinterpret_cast<char*>(&checksum),
        &unused_bytes_read));
    if (crc32c::Unmask(checksum) != *actual_crc32c) {
      return errors::DataLoss(
          "The checksum after Variant ", i, " does not match.",
          " Expected: ", strings::Printf("%08u", crc32c::Unmask(checksum)),
          " Actual: ", strings::Printf("%08u", *actual_crc32c));
    }
    *actual_crc32c = crc32c::Extend(
        *actual_crc32c, reinterpret_cast<char*>(&checksum), sizeof(uint32));
    ret->flat<Variant>()(i) = std::move(v);
  }
  return absl::OkStatus();
}
char* GetBackingBuffer(const Tensor& val) {
  CHECK(DataTypeCanUseMemcpy(val.dtype())) << val.dtype();
  return const_cast<char*>(val.tensor_data().data());
}
tstring* GetStringBackingBuffer(const Tensor& val) {
  CHECK_EQ(DT_STRING, val.dtype());
  return const_cast<tstring*>(val.flat<tstring>().data());
}
Status ParseEntryProto(StringPiece key, StringPiece value,
                       protobuf::MessageLite* out) {
  if (!out->ParseFromArray(value.data(), value.size())) {
    return errors::DataLoss("Entry for key ", key, " not parseable.");
  }
  return absl::OkStatus();
}
Status WriteTensor(const Tensor& val, tsl::BufferedWritableFile* out,
                   size_t* bytes_written) {
  DCHECK_NE(val.dtype(), DT_STRING);
  DCHECK_NE(val.dtype(), DT_VARIANT);
  *bytes_written = val.TotalBytes();
  char* buf = GetBackingBuffer(val);
  VLOG(1) << "Appending " << *bytes_written << " bytes to file";
  return out->Append(StringPiece(buf, *bytes_written));
}
Status WriteStringTensor(const Tensor& val, tsl::BufferedWritableFile* out,
                         size_t* bytes_written, uint32* crc32c) {
  DCHECK_EQ(val.dtype(), DT_STRING);
  const tstring* strings = GetStringBackingBuffer(val);
  string lengths;
  lengths.reserve(val.NumElements());  
  *crc32c = 0;
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    const tstring* elem = &strings[i];
    DCHECK_EQ(elem->size(), static_cast<uint64>(elem->size()));
    const uint64 elem_size = static_cast<uint64>(elem->size());
    core::PutVarint64(&lengths, elem_size);
    if (elem_size <= UINT32_MAX) {
      const uint32 elem_size_uint32 = static_cast<uint32>(elem_size);
      *crc32c = crc32c::Extend(*crc32c,
                               reinterpret_cast<const char*>(&elem_size_uint32),
                               sizeof(uint32));
    } else {
      *crc32c = crc32c::Extend(
          *crc32c, reinterpret_cast<const char*>(&elem_size), sizeof(uint64));
    }
  }
  TF_RETURN_IF_ERROR(out->Append(lengths));
  *bytes_written = lengths.size();
  const uint32 length_checksum = crc32c::Mask(*crc32c);
  TF_RETURN_IF_ERROR(out->Append(StringPiece(
      reinterpret_cast<const char*>(&length_checksum), sizeof(uint32))));
  *crc32c = crc32c::Extend(
      *crc32c, reinterpret_cast<const char*>(&length_checksum), sizeof(uint32));
  *bytes_written += sizeof(uint32);
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    const tstring* string = &strings[i];
    TF_RETURN_IF_ERROR(out->Append(*string));
    *bytes_written += string->size();
    *crc32c = crc32c::Extend(*crc32c, string->data(), string->size());
  }
  return absl::OkStatus();
}
Status WriteVariantTensor(const Tensor& val, tsl::BufferedWritableFile* out,
                          size_t* bytes_written, uint32* crc32c) {
  DCHECK_EQ(val.dtype(), DT_VARIANT);
  *crc32c = 0;
  *bytes_written = 0;
  for (int64_t i = 0; i < val.NumElements(); ++i) {
    VariantTensorData data;
    val.flat<Variant>()(i).Encode(&data);
    VariantTensorDataProto proto;
    data.ToProto(&proto);
    string elem;
    if (!proto.SerializeToString(&elem)) {
      return errors::Unknown(
          "Failed to serialize tensor data of size ", proto.ByteSizeLong(),
          ". Tensor: ", val.flat<Variant>()(i).DebugString());
    }
    DCHECK_EQ(elem.size(), static_cast<uint64>(elem.size()));
    const auto elem_size = static_cast<uint64>(elem.size());
    string len;
    core::PutVarint64(&len, elem_size);
    TF_RETURN_IF_ERROR(out->Append(len));
    *crc32c = crc32c::Extend(*crc32c, reinterpret_cast<const char*>(&elem_size),
                             sizeof(uint64));
    *bytes_written += len.size();
    TF_RETURN_IF_ERROR(out->Append(elem));
    *crc32c = crc32c::Extend(*crc32c, elem.data(), elem.size());
    *bytes_written += elem.size();
    const uint32 length_checksum = crc32c::Mask(*crc32c);
    TF_RETURN_IF_ERROR(out->Append(StringPiece(
        reinterpret_cast<const char*>(&length_checksum), sizeof(uint32))));
    *crc32c =
        crc32c::Extend(*crc32c, reinterpret_cast<const char*>(&length_checksum),
                       sizeof(uint32));
    *bytes_written += sizeof(uint32);
  }
  return absl::OkStatus();
}
bool IsFullSlice(const TensorSlice& slice_spec,
                 const TensorShape& full_tensor_shape) {
  if (slice_spec.IsFull()) {
    return true;
  } else {
    TensorShape sliced_shape;
    slice_spec.SliceTensorShape(full_tensor_shape, &sliced_shape).IgnoreError();
    return sliced_shape == full_tensor_shape;
  }
}
Status CorruptFileError(const Status& in_status, const string& filename,
                        const string& detail) {
  if (in_status.ok()) {
    return errors::Internal("Unable to read file (", filename,
                            "). Perhaps the file is corrupt or was produced by "
                            "a newer version of TensorFlow with format changes "
                            "(",
                            detail, ")");
  }
  return Status(
      in_status.code(),
      strings::StrCat("Unable to read file (", filename,
                      "). Perhaps the file is corrupt or was produced by a "
                      "newer version of TensorFlow with format changes (",
                      detail, "): ", in_status.message()));
}
table::Options TableBuilderOptions() {
  table::Options o;
  o.compression = table::kNoCompression;
  return o;
}
Status PadAlignment(tsl::BufferedWritableFile* out, int alignment,
                    int64_t* size) {
  int bytes_over = *size % alignment;
  if (bytes_over == 0) {
    return absl::OkStatus();
  }
  int bytes_to_write = alignment - bytes_over;
  Status status = out->Append(string(bytes_to_write, '\0'));
  if (status.ok()) {
    *size += bytes_to_write;
  }
  return status;
}
}  
BundleWriter::BundleWriter(Env* env, StringPiece prefix, const Options& options)
    : env_(env), options_(options), prefix_(prefix), out_(nullptr), size_(0) {
  status_ = env_->HasAtomicMove(prefix_, &use_temp_file_);
  if (!status_.ok()) return;
  data_path_ = DataFilename(prefix_, 0, 1);
  metadata_path_ = MetaFilename(prefix_);
  if (use_temp_file_) {
    data_path_ = strings::StrCat(data_path_, ".tempstate", random::New64());
    metadata_path_ =
        strings::StrCat(metadata_path_, ".tempstate", random::New64());
  }
  status_ = env_->CreateDir(string(io::Dirname(prefix_)));
  if (!status_.ok() && !errors::IsAlreadyExists(status_)) {
    return;
  }
  std::unique_ptr<WritableFile> wrapper;
  status_ = env_->NewWritableFile(data_path_, &wrapper);
  if (!status_.ok()) return;
  out_ = std::make_unique<tsl::BufferedWritableFile>(
      std::move(wrapper), 8 << 20 );
  VLOG(1) << "Writing to file " << data_path_;
}
Status BundleWriter::Add(StringPiece key, const Tensor& val) {
  if (!status_.ok()) return status_;
  CHECK_NE(key, kHeaderEntryKey);
  const string key_string(key);
  if (entries_.find(key_string) != entries_.end()) {
    status_ = errors::InvalidArgument("Adding duplicate key: ", key);
    return status_;
  }
  BundleEntryProto* entry = &entries_[key_string];
  entry->set_dtype(val.dtype());
  val.shape().AsProto(entry->mutable_shape());
  entry->set_shard_id(0);
  entry->set_offset(size_);
  size_t data_bytes_written = 0;
  uint32 crc32c = 0;
  out_->reset_crc32();
  if (val.dtype() == DT_STRING) {
    status_ = WriteStringTensor(val, out_.get(), &data_bytes_written, &crc32c);
  } else if (val.dtype() == DT_VARIANT) {
    status_ = WriteVariantTensor(val, out_.get(), &data_bytes_written, &crc32c);
  } else {
    status_ = WriteTensor(val, out_.get(), &data_bytes_written);
    crc32c = out_->crc32();
  }
  if (status_.ok()) {
    entry->set_size(data_bytes_written);
    entry->set_crc32c(crc32c::Mask(crc32c));
    size_ += data_bytes_written;
    status_ = PadAlignment(out_.get(), options_.data_alignment, &size_);
  }
  return status_;
}
Status BundleWriter::AddSlice(StringPiece full_tensor_key,
                              const TensorShape& full_tensor_shape,
                              const TensorSlice& slice_spec,
                              const Tensor& slice_tensor) {
  if (!status_.ok()) return status_;
  CHECK_NE(full_tensor_key, kHeaderEntryKey);
  if (IsFullSlice(slice_spec, full_tensor_shape)) {
    return Add(full_tensor_key, slice_tensor);
  }
  const string full_tensor_key_string(full_tensor_key);
  BundleEntryProto* full_entry = &entries_[full_tensor_key_string];
  if (full_entry->dtype() != DT_INVALID) {
    CHECK_EQ(full_entry->dtype(), slice_tensor.dtype());
  }
  if (full_entry->has_shape()) {
    CHECK(TensorShape(full_entry->shape()) == full_tensor_shape);
  }
  full_entry->set_dtype(slice_tensor.dtype());
  full_tensor_shape.AsProto(full_entry->mutable_shape());
  TensorSliceProto* slice_proto = full_entry->add_slices();
  slice_spec.AsProto(slice_proto);
  const string slice_name =
      checkpoint::EncodeTensorNameSlice(full_tensor_key_string, slice_spec);
  status_ = Add(slice_name, slice_tensor);
  return status_;
}
Status BundleWriter::Finish() {
  if (out_) {
    status_.Update(out_->Close());
    out_ = nullptr;
    if (status_.ok()) {
      if (use_temp_file_) {
        status_ =
            Env::Default()->RenameFile(data_path_, DataFilename(prefix_, 0, 1));
      }
    } else {
      Env::Default()->DeleteFile(data_path_).IgnoreError();
    }
  }
  if (!status_.ok()) return status_;
  std::unique_ptr<WritableFile> file;
  status_ = env_->NewWritableFile(metadata_path_, &file);
  if (!status_.ok()) return status_;
  {
    table::Options options;
    options.compression = table::kNoCompression;
    table::TableBuilder builder(options, file.get());
    BundleHeaderProto header;
    header.set_num_shards(1);
    header.set_endianness(BundleHeaderProto::LITTLE);
    if (!port::kLittleEndian) header.set_endianness(BundleHeaderProto::BIG);
    VersionDef* version = header.mutable_version();
    version->set_producer(kTensorBundleVersion);
    version->set_min_consumer(kTensorBundleMinConsumer);
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    for (const auto& p : entries_) {
      builder.Add(p.first, p.second.SerializeAsString());
    }
    status_ = builder.Finish();
  }
  status_.Update(file->Close());
  if (!status_.ok()) {
    Env::Default()->DeleteFile(metadata_path_).IgnoreError();
    return status_;
  } else if (use_temp_file_) {
    status_ = Env::Default()->RenameFile(metadata_path_, MetaFilename(prefix_));
    if (!status_.ok()) return status_;
  }
  status_ = errors::Internal("BundleWriter is closed");
  return absl::OkStatus();
}
struct MergeState {
  int num_shards = 0;
  bool seen_first_bundle = false;
  BundleHeaderProto_Endianness endianness;
  VersionDef version;
  std::map<string, BundleEntryProto> entries;
  std::unordered_map<string, int32> shard_ids;
};
static Status MergeOneBundle(Env* env, StringPiece prefix,
                             MergeState* merge_state) {
  VLOG(1) << "Merging bundle:" << prefix;
  const string filename = MetaFilename(prefix);
  uint64 file_size;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
  table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      table::Table::Open(TableBuilderOptions(), file.get(), file_size, &table));
  std::unique_ptr<table::Table> table_deleter(table);
  std::unique_ptr<table::Iterator> iter(table->NewIterator());
  int num_shards;
  {
    iter->Seek(kHeaderEntryKey);
    if (!iter->Valid()) {
      return CorruptFileError(iter->status(), filename,
                              "failed to seek to header entry");
    }
    BundleHeaderProto header;
    Status s = ParseEntryProto(iter->key(), iter->value(), &header);
    if (!s.ok()) return CorruptFileError(s, filename, "unable to parse header");
    merge_state->num_shards += header.num_shards();
    if (!merge_state->seen_first_bundle) {
      merge_state->seen_first_bundle = true;
      merge_state->endianness = header.endianness();
      merge_state->version = header.version();
    } else {
      if (merge_state->endianness != header.endianness()) {
        return errors::InvalidArgument(
            "Merging bundles with conflicting endianness; inputs corrupted?");
      }
      string curr_version, merge_version;
      header.version().SerializeToString(&curr_version);
      merge_state->version.SerializeToString(&merge_version);
      if (curr_version != merge_version) {
        return errors::InvalidArgument(
            "Merging bundles with different format versions: merged ",
            merge_version, " vs. curr ", curr_version);
      }
    }
    num_shards = header.num_shards();
    iter->Next();
  }
  BundleEntryProto to_merge_entry;
  for (; iter->Valid(); iter->Next()) {
    const string key(iter->key());
    const auto entry_iter = merge_state->entries.find(key);
    if (entry_iter != merge_state->entries.end() &&
        entry_iter->second.slices().empty()) {
      return errors::InvalidArgument(
          "Duplicate tensor keyed by ", key,
          " encountered, when merging prefix: ", prefix);
    }
    TF_RETURN_IF_ERROR(
        ParseEntryProto(iter->key(), iter->value(), &to_merge_entry));
    if (entry_iter != merge_state->entries.end()) {
      BundleEntryProto& existing_entry = entry_iter->second;
      if (to_merge_entry.slices().empty()) {
        return errors::Internal(
            "Duplicate tensor keyed by ", key,
            "; attempting to merge in a non-slice bundle entry");
      }
      for (int i = 0; i < to_merge_entry.slices_size(); ++i) {
        TensorSliceProto* slot = existing_entry.add_slices();
        *slot = to_merge_entry.slices(i);
      }
      CHECK_EQ(existing_entry.dtype(), to_merge_entry.dtype());
      CHECK(TensorShape(existing_entry.shape()) ==
            TensorShape(to_merge_entry.shape()));
      continue;
    }
    auto result = merge_state->shard_ids.insert(
        {DataFilename(prefix, to_merge_entry.shard_id(), num_shards),
         merge_state->shard_ids.size()});
    to_merge_entry.set_shard_id(result.first->second);
    merge_state->entries[key] = to_merge_entry;
  }
  return absl::OkStatus();
}
Status MergeBundles(Env* env, absl::Span<const tstring> prefixes,
                    StringPiece merged_prefix, bool allow_missing_files) {
  MergeState merge;
  Status status = env->CreateDir(string(io::Dirname(merged_prefix)));
  if (!status.ok() && !errors::IsAlreadyExists(status)) return status;
  bool atleast_one_file_exists = false;
  for (auto& prefix : prefixes) {
    if (!env->FileExists(MetaFilename(prefix)).ok()) {
      if (allow_missing_files) continue;
      return errors::InvalidArgument(
          "allow_missing_files was set to false and ", prefix,
          " did not exist.", env->FileExists(prefix).ToString());
    }
    atleast_one_file_exists = true;
    TF_RETURN_IF_ERROR(MergeOneBundle(env, prefix, &merge));
  }
  if (!atleast_one_file_exists) {
    return errors::InvalidArgument(
        "At least one prefix checkpoint file must exist, but none existed.");
  }
  for (const auto& p : merge.shard_ids) {
    VLOG(1) << "Renaming " << p.first << " to "
            << DataFilename(merged_prefix, p.second, merge.shard_ids.size());
    TF_RETURN_IF_ERROR(env->RenameFile(
        p.first,
        DataFilename(merged_prefix, p.second, merge.shard_ids.size())));
  }
  std::unique_ptr<WritableFile> merged_metadata;
  TF_RETURN_IF_ERROR(
      env->NewWritableFile(MetaFilename(merged_prefix), &merged_metadata));
  {
    table::TableBuilder builder(TableBuilderOptions(), merged_metadata.get());
    BundleHeaderProto header;
    header.set_num_shards(merge.num_shards);
    header.set_endianness(merge.endianness);
    *header.mutable_version() = merge.version;
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    for (const auto& p : merge.entries) {
      builder.Add(p.first, p.second.SerializeAsString());
    }
    status = builder.Finish();
  }
  status.Update(merged_metadata->Close());
  if (!status.ok()) return status;
  VLOG(1) << "Merged bundles to:" << merged_prefix;
  for (const tstring& prefix : prefixes) {
    env->DeleteFile(MetaFilename(prefix)).IgnoreError();
  }
  return status;
}
BundleReader::BundleReader(
    Env* env, StringPiece prefix,
    bool enable_multi_threading_for_testing )
    : BundleReader(env, prefix, {nullptr, enable_multi_threading_for_testing}) {
}
BundleReader::BundleReader(Env* env, StringPiece prefix, Options options)
    : env_(env),
      prefix_(prefix),
      cache_(options.cache),
      metadata_(nullptr),
      table_(nullptr),
      index_cache_(nullptr),
      iter_(nullptr),
      need_to_swap_bytes_(false),
      enable_multi_threading_for_testing_(
          options.enable_multi_threading_for_testing) {
  if (cache_ == nullptr) {
    owned_cache_ = std::make_unique<BundleCache>(env);
    cache_ = owned_cache_.get();
  }
  const string filename = MetaFilename(prefix_);
  uint64 file_size;
  status_ = env_->GetFileSize(filename, &file_size);
  if (!status_.ok()) return;
  std::unique_ptr<RandomAccessFile> wrapper;
  status_ = env_->NewRandomAccessFile(filename, &wrapper);
  if (!status_.ok()) return;
  metadata_ = wrapper.release();
  table::Options o;
  int64_t cache_size;
  Status s =
      ReadInt64FromEnvVar("TF_TABLE_INDEX_CACHE_SIZE_IN_MB", 0, &cache_size);
  if (s.ok() && cache_size > 0) {
    index_cache_ = table::NewLRUCache(cache_size << 20);
    o.block_cache = index_cache_;
  }
  status_ = table::Table::Open(o, metadata_, file_size, &table_);
  if (!status_.ok()) return;
  iter_ = table_->NewIterator();
  iter_->Seek(kHeaderEntryKey);
  if (!iter_->Valid()) {
    status_ = CorruptFileError(iter_->status(), filename,
                               "failed to seek to header entry");
    return;
  }
  BundleHeaderProto header;
  status_ = ParseEntryProto(iter_->key(), iter_->value(), &header);
  if (!status_.ok()) {
    status_ = CorruptFileError(status_, filename, "unable to parse header");
    return;
  }
  num_shards_ = header.num_shards();
  if ((header.endianness() == BundleHeaderProto::BIG && port::kLittleEndian) ||
      (header.endianness() == BundleHeaderProto::LITTLE &&
       !port::kLittleEndian)) {
    need_to_swap_bytes_ = true;
  }
  status_ = CheckVersions(header.version(), kTensorBundleVersion,
                          kTensorBundleMinProducer, "Checkpoint", "checkpoint");
}
BundleReader::~BundleReader() {
  delete metadata_;
  delete iter_;
  delete table_;
  if (index_cache_) {
    delete index_cache_;
  }
  for (auto& temp : data_) {
    delete temp.second;
  }
  for (auto& temp : tensor_slices_) {
    delete temp.second;
  }
  data_.clear();
  tensor_slices_.clear();
}
Status BundleReader::GetBundleEntryProto(StringPiece key,
                                         BundleEntryProto* entry) {
  entry->Clear();
  TF_CHECK_OK(status_);
  Seek(key);
  if (!iter_->Valid() || iter_->key() != key) {
    return errors::NotFound("Key ", key, " not found in checkpoint");
  }
  BundleEntryProto entry_copy;
  TF_RETURN_IF_ERROR(
      ParseEntryProto(iter_->key(), iter_->value(), &entry_copy));
  if (!TensorShape::IsValid(entry_copy.shape())) {
    return errors::DataLoss("Invalid tensor shape: ", key, " ",
                            entry_copy.shape().ShortDebugString());
  }
  entry->Swap(&entry_copy);
  return absl::OkStatus();
}
Status BundleReader::GetValue(const BundleEntryProto& entry, Tensor* val) {
  Tensor* ret = val;
  const TensorShape stored_shape(TensorShape(entry.shape()));
  if (val->NumElements() == 0) {
    ret = new Tensor(entry.dtype(), stored_shape);
  }
  if (entry.dtype() != DT_STRING && entry.dtype() != DT_VARIANT) {
    if (entry.size() != ret->TotalBytes()) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size ", ret->TotalBytes());
    }
  } else if (entry.dtype() == DT_STRING) {
    const size_t lower_bound = ret->NumElements() + ret->TotalBytes() -
                               sizeof(tstring) * ret->NumElements();
    if (entry.size() < lower_bound) {
      return errors::DataLoss("Invalid size in bundle entry: key ", key(),
                              "; stored size ", entry.size(),
                              "; expected size is at least ", lower_bound);
    }
  }
  io::InputBuffer* buffered_file = data_[entry.shard_id()];
  if (buffered_file == nullptr) {
    RandomAccessFile* file = nullptr;
    TF_RETURN_IF_ERROR(cache_->GetFile(
        DataFilename(prefix_, entry.shard_id(), num_shards_), &file));
    buffered_file = new io::InputBuffer(file, kBufferSize);
    data_[entry.shard_id()] = buffered_file;
  }
  CHECK(buffered_file != nullptr);
  TF_RETURN_IF_ERROR(buffered_file->Seek(entry.offset()));
  uint32 actual_crc32c = 0;
  if (DataTypeCanUseMemcpy(entry.dtype())) {
    char* backing_buffer = const_cast<char*>((ret->tensor_data().data()));
    size_t unused_bytes_read;
    if (entry.size() > kBufferSize || enable_multi_threading_for_testing_) {
      StringPiece sp;
      if (!enable_multi_threading_for_testing_ &&
          entry.size() < kLargeTensorThreshold) {
        TF_RETURN_IF_ERROR(buffered_file->file()->Read(
            entry.offset(), entry.size(), &sp, backing_buffer));
        if (sp.data() != backing_buffer) {
          memmove(backing_buffer, sp.data(), entry.size());
        }
      } else {
        int64_t section_size = kMinSectionSize;
        int64_t thread_pool_size =
            (entry.size() + kMinSectionSize - 1) / kMinSectionSize;
        if (thread_pool_size > kMaxFileReadThreads ||
            enable_multi_threading_for_testing_) {
          thread_pool_size = kMaxFileReadThreads;
          section_size =
              (entry.size() + kMaxFileReadThreads - 1) / kMaxFileReadThreads;
        }
        std::vector<Status> statuses(thread_pool_size);
        auto reader_pool = std::make_unique<thread::ThreadPool>(
            Env::Default(), "restore_large_tensor", thread_pool_size);
        for (int i = 0; i < thread_pool_size; ++i) {
          reader_pool->Schedule([&, i]() {
            int64_t offset = i * section_size;
            int64_t size = i == thread_pool_size - 1 ? entry.size() - offset
                                                     : section_size;
            std::unique_ptr<RandomAccessFile> section_reader = nullptr;
            StringPiece sp;
            if (auto file_status = env_->NewRandomAccessFile(
                    DataFilename(prefix_, entry.shard_id(), num_shards_),
                    &section_reader);
                !file_status.ok()) {
              statuses[i] = file_status;
              return;
            }
            auto backing_buffer_current_pos = backing_buffer + offset;
            auto status = section_reader->Read(entry.offset() + offset, size,
                                               &sp, backing_buffer_current_pos);
            if (sp.data() != backing_buffer_current_pos) {
              memmove(backing_buffer_current_pos, sp.data(), size);
            }
            statuses[i] = std::move(status);
          });
        }
        reader_pool = nullptr;  
        for (const auto& status : statuses) {
          TF_RETURN_IF_ERROR(status);
        }
      }
    } else {
      TF_RETURN_IF_ERROR(buffered_file->ReadNBytes(entry.size(), backing_buffer,
                                                   &unused_bytes_read));
    }
    actual_crc32c = crc32c::Value(backing_buffer, entry.size());
    if (need_to_swap_bytes_) {
      TF_RETURN_IF_ERROR(ByteSwapTensor(ret));
    }
  } else if (entry.dtype() == DT_VARIANT) {
    if (need_to_swap_bytes_) {
      return errors::Unimplemented(
          "TensorBundle at ", prefix_,
          "is of a different endianness than this machine's hardware, and "
          "the bundle contains a variant (arbitrary C++ type) tensor. "
          "Byte-swapping of variant tensors is not currently implemented.");
    }
    TF_RETURN_IF_ERROR(ReadVariantTensor(buffered_file, ret, entry.offset(),
                                         entry.size(), &actual_crc32c));
  } else {
    TF_RETURN_IF_ERROR(ReadStringTensor(
        buffered_file, ret->NumElements(), entry.offset(), entry.size(),
        GetStringBackingBuffer(*ret), &actual_crc32c, need_to_swap_bytes_));
  }
  if (crc32c::Unmask(entry.crc32c()) != actual_crc32c) {
    return errors::DataLoss(
        "TensorBundle at ", prefix_, " shard ", entry.shard_id(), " (",
        entry.size(), " bytes): Checksum does not match: stored ",
        strings::Printf("%08u", crc32c::Unmask(entry.crc32c())),
        " vs. calculated on the restored bytes ", actual_crc32c);
  }
  *val = *ret;
  if (ret != val) delete ret;
  return absl::OkStatus();
}
Status BundleReader::Lookup(StringPiece key, Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));
  if (entry.slices().empty()) {
    return GetValue(entry, val);
  } else {
    return GetSliceValue(
        key, entry,
         TensorSlice(TensorShape(entry.shape()).dims()), val);
  }
}
Status BundleReader::ReadCurrent(Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(ParseEntryProto(iter_->key(), iter_->value(), &entry));
  if (!TensorShape::IsValid(entry.shape())) {
    return errors::DataLoss("Invalid tensor shape: ", iter_->key(), " ",
                            entry.shape().ShortDebugString());
  }
  if (entry.slices().empty()) {
    return GetValue(entry, val);
  } else {
    return GetSliceValue(
        iter_->key(), entry,
         TensorSlice(TensorShape(entry.shape()).dims()), val);
  }
}
Status BundleReader::LookupTensorSlices(StringPiece key,
                                        std::vector<TensorSlice>* slices) {
  slices->clear();
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));
  slices->reserve(entry.slices_size());
  for (const auto& slice : entry.slices()) {
    slices->emplace_back(slice);
  }
  return absl::OkStatus();
}
Status BundleReader::LookupSlice(StringPiece full_tensor_key,
                                 const TensorSlice& slice_spec, Tensor* val) {
  CHECK(val != nullptr);
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(full_tensor_key, &entry));
  return GetSliceValue(full_tensor_key, entry, slice_spec, val);
}
Status BundleReader::GetSliceValue(StringPiece full_tensor_key,
                                   const BundleEntryProto& full_tensor_entry,
                                   const TensorSlice& slice_spec, Tensor* val) {
  using checkpoint::RegisterTensorSlice;
  using checkpoint::TensorSliceSet;
  DCHECK_GE(full_tensor_entry.slices_size(), 0);
  const TensorShape full_shape(TensorShape(full_tensor_entry.shape()));
  std::vector<std::pair<TensorSlice, string>> details;
  const string full_tensor_key_string(full_tensor_key);
  const TensorSliceSet* tss =
      gtl::FindPtrOrNull(tensor_slices_, full_tensor_key_string);
  if (tss == nullptr) {
    if (full_tensor_entry.slices().empty()) {
      TF_RETURN_IF_ERROR(RegisterTensorSlice(
          full_tensor_key_string, full_shape, full_tensor_entry.dtype(),
           "",
           TensorSlice(full_shape.dims()), &tensor_slices_));
    }
    for (const TensorSliceProto& slice : full_tensor_entry.slices()) {
      TF_RETURN_IF_ERROR(RegisterTensorSlice(
          full_tensor_key_string, full_shape, full_tensor_entry.dtype(),
           "", TensorSlice(slice), &tensor_slices_));
    }
    tss = gtl::FindPtrOrNull(tensor_slices_, full_tensor_key_string);
    CHECK_NE(tss, nullptr);
  }
  if (!tss->QueryMeta(slice_spec, &details)) {
    return errors::InvalidArgument(
        "Does not have sufficient slices for partitioned tensor ",
        full_tensor_key,
        " to restore in slice_spec: ", slice_spec.DebugString());
  }
  BundleEntryProto stored_slice_entry = full_tensor_entry;
  for (const auto& slice_tag_pair : details) {
    const TensorSlice& stored_slice = slice_tag_pair.first;
    if (!stored_slice.IsFull()) {
      const string encoded_stored_slice_name =
          checkpoint::EncodeTensorNameSlice(full_tensor_key_string,
                                            stored_slice);
      status_ =
          GetBundleEntryProto(encoded_stored_slice_name, &stored_slice_entry);
      if (!status_.ok()) return status_;
    }
    TensorShape stored_slice_shape(stored_slice_entry.shape());
    if (stored_slice == slice_spec ||
        (stored_slice_shape == val->shape() &&
         IsFullSlice(stored_slice, stored_slice_shape) &&
         IsFullSlice(slice_spec, stored_slice_shape))) {
      VLOG(1) << "Optimized for common case: directly copying into "
                 "pre-allocated buffer; spec: "
              << slice_spec.DebugString();
      status_ = GetValue(stored_slice_entry, val);
      return status_;
    }
    Tensor stored_slice_tensor(stored_slice_entry.dtype(), stored_slice_shape);
    status_ = GetValue(stored_slice_entry, &stored_slice_tensor);
    if (!status_.ok()) return status_;
    const DataType common_dtype = full_tensor_entry.dtype();
    switch (common_dtype) {
#define HANDLE_COPY(T)                                                 \
  case DataTypeToEnum<T>::value:                                       \
    CHECK(CopyDataFromTensorSliceToTensorSlice(                        \
        full_shape, stored_slice, slice_spec,                          \
        stored_slice_tensor.flat<T>().data(), val->flat<T>().data())); \
    break;
      HANDLE_COPY(float)
      HANDLE_COPY(double)
      HANDLE_COPY(int32)
      HANDLE_COPY(uint8)
      HANDLE_COPY(int16)
      HANDLE_COPY(int8)
      HANDLE_COPY(complex64)
      HANDLE_COPY(complex128)
      HANDLE_COPY(int64_t)
      HANDLE_COPY(bool)
      HANDLE_COPY(qint32)
      HANDLE_COPY(quint8)
      HANDLE_COPY(qint8)
      HANDLE_COPY(bfloat16)
      HANDLE_COPY(int4)
      HANDLE_COPY(uint4)
      default:
        return errors::InvalidArgument("Dtype ", DataTypeString(common_dtype),
                                       " not supported.");
    }
#undef HANDLE_COPY
  }
  return absl::OkStatus();
}
bool BundleReader::Contains(StringPiece key) {
  Seek(key);
  return Valid() && (this->key() == key);
}
Status BundleReader::LookupDtypeAndShape(StringPiece key, DataType* dtype,
                                         TensorShape* shape) {
  BundleEntryProto entry;
  TF_RETURN_IF_ERROR(GetBundleEntryProto(key, &entry));
  *dtype = entry.dtype();
  *shape = TensorShape(entry.shape());
  return absl::OkStatus();
}
Status BundleReader::LookupTensorShape(StringPiece key, TensorShape* shape) {
  DataType ignored;
  return LookupDtypeAndShape(key, &ignored, shape);
}
string BundleReader::DebugString() {
  string shape_str;
  BundleEntryProto entry;
  Seek(kHeaderEntryKey);
  for (Next(); Valid(); Next()) {
    CHECK(entry.ParseFromArray(value().data(), value().size()));
    if (entry.slices_size() > 0) continue;  
    strings::StrAppend(&shape_str, key(), " (", DataType_Name(entry.dtype()),
                       ") ", TensorShape(entry.shape()).DebugString());
    strings::StrAppend(&shape_str, "\n");
  }
  return shape_str;
}
BundleCache::BundleCache(Env* env) : env_(env) {}
BundleCache::FileState* BundleCache::EnsureOpened(std::string name) {
  FileState* f;
  {
    absl::MutexLock l(&mu_);
    auto& slot = opened_files_[name];
    if (slot == nullptr) {
      slot = std::make_unique<FileState>();
    }
    f = slot.get();
  }
  absl::call_once(f->once, [this, name = std::move(name), f] {
    f->open_status = env_->NewRandomAccessFile(name, &f->file);
  });
  return f;
}
Status BundleCache::GetFile(const std::string& fname, RandomAccessFile** file) {
  FileState* f = EnsureOpened(fname);
  *file = f->file.get();
  return f->open_status;
}
namespace {
inline char* AlignedMalloc(size_t size) {
  char* buffer = static_cast<char*>(port::AlignedMalloc(size, 64));
  DCHECK(buffer);
  return buffer;
}
}  
}  