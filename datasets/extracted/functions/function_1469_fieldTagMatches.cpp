#include "tensorflow/cc/saved_model/fingerprinting_utils.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"  
#include "riegeli/records/record_reader.h"  
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tensorflow/tools/proto_splitter/merge.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace tensorflow::saved_model::fingerprinting {
using ::tensorflow::proto_splitter::ChunkedField;
using ::tensorflow::proto_splitter::ChunkedMessage;
using ::tensorflow::proto_splitter::ChunkInfo;
using ::tensorflow::proto_splitter::ChunkMetadata;
using ::tensorflow::proto_splitter::FieldIndex;
using tools::proto_splitter::Field;
using tools::proto_splitter::FieldType;
using tools::proto_splitter::GetChunkMetadata;
using tools::proto_splitter::GetFieldTypes;
using tools::proto_splitter::GetMutableField;
using tools::proto_splitter::GetRiegeliReader;
using tools::proto_splitter::Merger;
using tools::proto_splitter::MutableFieldResult;
using tools::proto_splitter::ReadChunk;
namespace fingerprinting_utils_internal {
using ::tensorflow::protobuf::Map;
using ::tensorflow::protobuf::Message;
using ::tensorflow::protobuf::RepeatedPtrField;
using ::tensorflow::protobuf::io::CodedOutputStream;
using ::tensorflow::protobuf::io::StringOutputStream;
absl::StatusOr<int> fieldTagMatches(const RepeatedPtrField<FieldIndex>& a,
                                    const RepeatedPtrField<FieldIndex>& b) {
  int matches = 0;
  for (int i = 0; i == matches && i < a.size() && i < b.size(); i++) {
    switch (b[i].kind_case()) {
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kField:
        if (a.at(i).has_field() && a.at(i).field() == b.at(i).field()) {
          matches += 1;
        }
        break;
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kIndex:
        if (a.at(i).has_index() && a.at(i).index() == b.at(i).index()) {
          matches += 1;
        }
        break;
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kMapKey:
        if (a.at(i).has_map_key()) {
          const ::tensorflow::proto_splitter::FieldIndex_MapKey& key =
              b.at(i).map_key();
          const ::tensorflow::proto_splitter::FieldIndex_MapKey& chunked_key =
              a.at(i).map_key();
          switch (key.type_case()) {
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kS:
              if (chunked_key.has_s() && chunked_key.s() == key.s()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                kBoolean:
              if (chunked_key.has_boolean() &&
                  chunked_key.boolean() == key.boolean()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                kUi32:
              if (chunked_key.has_ui32() && chunked_key.ui32() == key.ui32()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                kUi64:
              if (chunked_key.has_ui64() && chunked_key.ui64() == key.ui64()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                kI32:
              if (chunked_key.has_i32() && chunked_key.i32() == key.i32()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                kI64:
              if (chunked_key.has_i64() && chunked_key.i64() == key.i64()) {
                matches += 1;
              }
              break;
            case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
                TYPE_NOT_SET:
            default:
              return absl::FailedPreconditionError(
                  "Encountered unknown field_tag.map_key type.");
          }
        }
        break;
      case FieldIndex::KindCase::KIND_NOT_SET:
      default:
        return absl::FailedPreconditionError(
            "Encountered unknown field_tag kind.");
    }
  }
  return matches;
}
absl::StatusOr<::tensorflow::proto_splitter::ChunkedMessage>
PruneChunkedMessage(
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    std::vector<ChunkInfo> chunks_info,
    std::vector<RepeatedPtrField<FieldIndex>> target_fields_list) {
  ::tensorflow::proto_splitter::ChunkedMessage pruned_chunked_message;
  if (chunked_message.has_chunk_index()) {
    pruned_chunked_message.set_chunk_index(chunked_message.chunk_index());
  }
  for (const ChunkedField& chunked_field : chunked_message.chunked_fields()) {
    for (const auto& target_fields : target_fields_list) {
      TF_ASSIGN_OR_RETURN(
          int matches,
          fieldTagMatches(chunked_field.field_tag(), target_fields));
      if (matches == chunked_field.field_tag_size()) {
        auto cf = std::make_unique<proto_splitter::ChunkedField>();
        cf->mutable_field_tag()->CopyFrom(chunked_field.field_tag());
        TF_ASSIGN_OR_RETURN(
            *cf->mutable_message(),
            PruneChunkedMessage(chunked_field.message(), reader, chunks_info,
                                target_fields_list));
        pruned_chunked_message.mutable_chunked_fields()->AddAllocated(
            cf.release());
      }
    }
  }
  return pruned_chunked_message;
}
std::string SerializeProto(const Message& message) {
  std::string serialized_message;
  {
    StringOutputStream stream(&serialized_message);
    CodedOutputStream output(&stream);
    output.SetSerializationDeterministic(true);
    message.SerializeToCodedStream(&output);
  }
  return serialized_message;
}
absl::StatusOr<uint64_t> HashFields(
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info,
    const RepeatedPtrField<FieldIndex>& field_tags, Message* merged_message) {
  uint64_t field_checksum = 0;
  for (const ChunkedField& chunked_field : chunked_message.chunked_fields()) {
    const RepeatedPtrField<FieldIndex> chunked_field_tags =
        chunked_field.field_tag();
    const ChunkedMessage& chunked_message = chunked_field.message();
    TF_ASSIGN_OR_RETURN(int matches,
                        fieldTagMatches(chunked_field_tags, field_tags));
    if (chunked_message.has_chunk_index() && matches == field_tags.size()) {
      TF_ASSIGN_OR_RETURN(
          std::string chunk,
          ReadChunk(reader, chunks_info[chunked_message.chunk_index()]));
      field_checksum = FingerprintCat64(field_checksum, Fingerprint64(chunk));
    } else if (matches == field_tags.size()) {
      TF_ASSIGN_OR_RETURN(uint64_t hash,
                          HashFields(chunked_message, reader, chunks_info,
                                     field_tags, merged_message));
      field_checksum = FingerprintCat64(field_checksum, hash);
    } else if (chunked_message.has_chunk_index() &&
               matches == chunked_field_tags.size()) {
      TF_ASSIGN_OR_RETURN(std::vector<Field> fields,
                          GetFieldTypes(chunked_field_tags));
      for (const auto& field : fields) {
        TF_ASSIGN_OR_RETURN(MutableFieldResult mfr,
                            GetMutableField(merged_message, field));
        merged_message =
            mfr.parent->GetReflection()->MutableMessage(mfr.parent, mfr.field);
      }
      TF_ASSIGN_OR_RETURN(
          std::string chunk,
          ReadChunk(reader, chunks_info[chunked_message.chunk_index()]));
      merged_message->ParseFromString(chunk);
      TF_ASSIGN_OR_RETURN(uint64_t hash,
                          HashFields(chunked_message, reader, chunks_info,
                                     field_tags, merged_message));
      field_checksum = FingerprintCat64(field_checksum, hash);
    } else if (matches == chunked_field_tags.size()) {
      for (const ChunkedField& cf : chunked_message.chunked_fields()) {
        TF_ASSIGN_OR_RETURN(uint64_t hash,
                            HashFields(cf.message(), reader, chunks_info,
                                       field_tags, merged_message));
        field_checksum = FingerprintCat64(field_checksum, hash);
      }
    }
  }
  return field_checksum;
}
inline RepeatedPtrField<FieldIndex> GraphDefFieldTags() {
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex graph_def_field_tag;
  graph_def_field_tag.set_field(2);
  RepeatedPtrField<FieldIndex> graph_def_field_tags;
  graph_def_field_tags.Add(FieldIndex(meta_graph_field_tag));
  graph_def_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  graph_def_field_tags.Add(FieldIndex(graph_def_field_tag));
  return graph_def_field_tags;
}
inline RepeatedPtrField<FieldIndex> SignatureDefFieldTags() {
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex signature_def_field_tag;
  signature_def_field_tag.set_field(5);
  RepeatedPtrField<FieldIndex> signature_def_field_tags;
  signature_def_field_tags.Add(FieldIndex(meta_graph_field_tag));
  signature_def_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  signature_def_field_tags.Add(FieldIndex(signature_def_field_tag));
  return signature_def_field_tags;
}
inline RepeatedPtrField<FieldIndex> SavedObjectGraphFieldTags() {
  FieldIndex meta_graph_field_tag;
  meta_graph_field_tag.set_field(2);
  FieldIndex meta_graph_index_field_tag;
  meta_graph_index_field_tag.set_index(0);
  FieldIndex saved_object_graph_field_tag;
  saved_object_graph_field_tag.set_field(7);
  RepeatedPtrField<FieldIndex> saved_object_graph_field_tags;
  saved_object_graph_field_tags.Add(FieldIndex(meta_graph_field_tag));
  saved_object_graph_field_tags.Add(FieldIndex(meta_graph_index_field_tag));
  saved_object_graph_field_tags.Add(FieldIndex(saved_object_graph_field_tag));
  return saved_object_graph_field_tags;
}
absl::StatusOr<SavedModel> PrunedSavedModel(
    absl::string_view export_dir,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info, ChunkMetadata& chunk_metadata) {
  SavedModel saved_model;
  ChunkMetadata pruned_chunk_metadata;
  pruned_chunk_metadata.mutable_chunks()->CopyFrom(chunk_metadata.chunks());
  TF_ASSIGN_OR_RETURN(
      *pruned_chunk_metadata.mutable_message(),
      PruneChunkedMessage(chunk_metadata.message(), reader, chunks_info,
                          {GraphDefFieldTags(), SignatureDefFieldTags(),
                           SavedObjectGraphFieldTags()}));
  TF_RETURN_IF_ERROR(
      Merger::ReadPartial(io::JoinPath(export_dir, kSavedModelFilenamePrefix),
                          pruned_chunk_metadata, &saved_model));
  return saved_model;
}
absl::StatusOr<uint64_t> HashMessage(
    Message* message, const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info,
    const RepeatedPtrField<FieldIndex>& field_tags) {
  uint64_t total_message_hash = Fingerprint64(SerializeProto(*message));
  TF_ASSIGN_OR_RETURN(
      uint64_t message_hash,
      HashFields(chunked_message, reader, chunks_info, field_tags, message));
  return FingerprintCat64(total_message_hash, message_hash);
}
absl::StatusOr<uint64_t> HashGraphDef(
    ::tensorflow::GraphDef* graph_def, const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  return HashMessage(graph_def, chunked_message, reader, chunks_info,
                     GraphDefFieldTags());
}
absl::StatusOr<uint64_t> HashSignatureDef(
    const Map<std::string, ::tensorflow::SignatureDef>& signature_def_map,
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  uint64_t signature_def_hash = 0;
  std::vector<std::pair<std::string, ::tensorflow::SignatureDef>>
      signature_def_sorted(signature_def_map.begin(), signature_def_map.end());
  std::sort(signature_def_sorted.begin(), signature_def_sorted.end(),
            [](const std::pair<std::string, ::tensorflow::SignatureDef>& a,
               const std::pair<std::string, ::tensorflow::SignatureDef>& b) {
              return a.first < b.first;
            });
  for (const auto& signature_def : signature_def_sorted) {
    uint64_t signature_def_pair_hash =
        FingerprintCat64(Fingerprint64(signature_def.first),
                         Fingerprint64(SerializeProto(signature_def.second)));
    signature_def_hash =
        FingerprintCat64(signature_def_hash, signature_def_pair_hash);
    SignatureDef signature_def_val = signature_def.second;
    TF_ASSIGN_OR_RETURN(
        uint64_t signature_def_entry_hash,
        HashFields(chunked_message, reader, chunks_info,
                   SignatureDefFieldTags(), &signature_def_val));
    signature_def_hash =
        FingerprintCat64(signature_def_hash, signature_def_entry_hash);
  }
  return signature_def_hash;
}
absl::StatusOr<uint64_t> HashSavedObjectGraph(
    ::tensorflow::SavedObjectGraph* saved_object_graph,
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info) {
  return HashMessage(saved_object_graph, chunked_message, reader, chunks_info,
                     SavedObjectGraphFieldTags());
}
}  
using fingerprinting_utils_internal::HashFields;
using fingerprinting_utils_internal::HashGraphDef;
using fingerprinting_utils_internal::HashSavedObjectGraph;
using fingerprinting_utils_internal::HashSignatureDef;
using fingerprinting_utils_internal::PrunedSavedModel;
using fingerprinting_utils_internal::SerializeProto;
uint64_t HashCheckpointIndexFile(absl::string_view model_dir) {
  std::string meta_filename = MetaFilename(io::JoinPath(
      model_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename));
  std::string data;
  absl::Status read_status =
      ReadFileToString(Env::Default(), meta_filename, &data);
  if (read_status.ok()) {
    return tensorflow::Fingerprint64(data);
  } else {
    return 0;
  }
}
absl::StatusOr<FingerprintDef> CreateFingerprintDefCpb(
    absl::string_view export_dir, std::string cpb_file) {
  const int kFingerprintProducer = 2;
  TF_ASSIGN_OR_RETURN(auto reader, GetRiegeliReader(cpb_file));
  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    return absl::FailedPreconditionError(
        absl::StrCat("Couldn't read ChunkMetadata from chunked proto.\n",
                     read_metadata.status().ToString()));
  }
  ChunkMetadata chunk_metadata = read_metadata.value();
  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());
  FingerprintDef fingerprint_def;
  SavedModel saved_model;
  TF_ASSIGN_OR_RETURN(uint64_t saved_model_hash,
                      HashFields(chunk_metadata.message(), reader, chunks_info,
                                 {}, &saved_model));
  saved_model_hash = FingerprintCat64(
      saved_model_hash, Fingerprint64(SerializeProto(saved_model)));
  fingerprint_def.set_saved_model_checksum(saved_model_hash);
  TF_ASSIGN_OR_RETURN(
      saved_model,
      PrunedSavedModel(export_dir, reader, chunks_info, chunk_metadata));
  TF_ASSIGN_OR_RETURN(
      uint64_t graph_def_program_hash,
      HashGraphDef(saved_model.mutable_meta_graphs(0)->mutable_graph_def(),
                   chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_graph_def_program_hash(graph_def_program_hash);
  TF_ASSIGN_OR_RETURN(
      uint64_t signature_def_hash,
      HashSignatureDef(saved_model.meta_graphs(0).signature_def(),
                       chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_signature_def_hash(signature_def_hash);
  TF_ASSIGN_OR_RETURN(
      uint64_t saved_object_graph_hash,
      HashSavedObjectGraph(
          saved_model.mutable_meta_graphs(0)->mutable_object_graph_def(),
          chunk_metadata.message(), reader, chunks_info));
  fingerprint_def.set_saved_object_graph_hash(saved_object_graph_hash);
  fingerprint_def.set_checkpoint_hash(HashCheckpointIndexFile(export_dir));
  reader.Close();
  VersionDef* version = fingerprint_def.mutable_version();
  version->set_producer(kFingerprintProducer);
  return fingerprint_def;
}
}  