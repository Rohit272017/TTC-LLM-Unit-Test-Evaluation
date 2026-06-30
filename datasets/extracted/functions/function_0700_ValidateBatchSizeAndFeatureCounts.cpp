#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"
#include <cstdint>
#include <functional>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  
namespace tensorflow {
namespace {
absl::Status ValidateBatchSizeAndFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config) {
  if (config.batch_size_per_tensor_core() <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid batch_size_per_tensor_core: %d found in the TPU embedding "
        "configuration. Valid values are >0.",
        config.batch_size_per_tensor_core()));
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid num_features: %d found for table: %s in the TPU embedding "
          "configuration. Valid values are >0.",
          table_config.num_features(), table_config.name()));
    }
  }  
  return absl::OkStatus();
}
absl::Status ValidateBatchSizeAndFeatureCountsAreEmpty(
    const tpu::TPUEmbeddingConfiguration& config) {
  if (config.batch_size_per_tensor_core() != 0) {
    return absl::InvalidArgumentError(
        "Invalid TPU embedding configuration. The batch_size_per_tensor_core "
        "field must NOT be populated when the feature_descriptor fields are "
        "filled in.");
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid TPU embedding configuration. The "
          "TableDescriptor.num_features field must NOT be populated when the "
          "feature_descriptor fields are filled in, num_features is set to %d "
          "for table %s.",
          table_config.num_features(), table_config.name()));
    }
  }  
  return absl::OkStatus();
}
absl::Status ValidateFeatureDescriptors(
    const tpu::TPUEmbeddingConfiguration& config) {
  const int table_count = config.table_descriptor_size();
  std::vector<bool> tables_present(table_count, false);
  for (const auto& feature_config : config.feature_descriptor()) {
    const int table_id = feature_config.table_id();
    const auto& input_shape = feature_config.input_shape();
    if (table_id < 0 || table_id >= table_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid table_id: %d found in feature_descriptor: %s, all table_ids "
          "must be in the range[0, %d)",
          table_id, feature_config.ShortDebugString(), table_count));
    }
    if (input_shape.empty()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "The input_shape field cannot be empty in feature_descriptor: %s",
          feature_config.ShortDebugString()));
    }
    for (const int dim_size : input_shape) {
      if (dim_size <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "The input_shape dimension sizes must all be >0 in "
            "feature_descriptor: %s, found dimension size set to %d",
            feature_config.ShortDebugString(), dim_size));
      }
    }
    tables_present[table_id] = true;
  }  
  for (int table_id = 0; table_id < table_count; ++table_id) {
    if (!tables_present[table_id]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No feature_descriptor fields found for table: %s (ID: %d) in "
          "the TPU embedding configuration.",
          config.table_descriptor(table_id).name(), table_id));
    }
  }
  return absl::OkStatus();
}
void PopulateFeatureDescriptors(tpu::TPUEmbeddingConfiguration* config) {
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    tpu::TPUEmbeddingConfiguration::FeatureDescriptor* feature_descriptor =
        config->add_feature_descriptor();
    feature_descriptor->set_table_id(table_id);
    feature_descriptor->add_input_shape(
        config->batch_size_per_tensor_core() *
        config->table_descriptor(table_id).num_features());
  }  
}
std::vector<int> ComputeInputFeatureBatchSizes(
    const tpu::TPUEmbeddingConfiguration& config) {
  std::vector<int32_t> input_feature_batch_sizes;
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    const int32_t batch_size =
        absl::c_accumulate(config.feature_descriptor(i).input_shape(),
                           1, std::multiplies<>());
    input_feature_batch_sizes.push_back(batch_size);
  }
  return input_feature_batch_sizes;
}
int ComputeBatchSizePerTensorCore(
    absl::Span<const int> input_feature_batch_sizes) {
  uint32_t batch_size = input_feature_batch_sizes[0];
  for (const uint32_t input_feature_batch_size : input_feature_batch_sizes) {
    batch_size =
        tensorflow::MathUtil::GCD(batch_size, input_feature_batch_size);
  }
  return batch_size;
}
std::vector<int> ComputeTpuFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config,
    absl::Span<const int> input_feature_batch_sizes,
    int batch_size_per_tensor_core) {
  DCHECK_EQ(input_feature_batch_sizes.size(), config.feature_descriptor_size());
  std::vector<int> tpu_feature_counts(config.table_descriptor_size(), 0);
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    DCHECK_EQ(input_feature_batch_sizes[i] % batch_size_per_tensor_core, 0);
    tpu_feature_counts[config.feature_descriptor(i).table_id()] +=
        (input_feature_batch_sizes[i] / batch_size_per_tensor_core);
  }
  return tpu_feature_counts;
}
void PopulateBatchSizeAndFeatureCounts(tpu::TPUEmbeddingConfiguration* config) {
  const std::vector<int> input_feature_batch_sizes =
      ComputeInputFeatureBatchSizes(*config);
  const int batch_size_per_tensor_core =
      ComputeBatchSizePerTensorCore(input_feature_batch_sizes);
  const std::vector<int> tpu_feature_counts = ComputeTpuFeatureCounts(
      *config, input_feature_batch_sizes, batch_size_per_tensor_core);
  config->set_batch_size_per_tensor_core(batch_size_per_tensor_core);
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    auto* table_config = config->mutable_table_descriptor(table_id);
    table_config->set_num_features(tpu_feature_counts[table_id]);
  }  
}
}  
absl::Status PopulateMissingFieldsInTPUEmbeddingConfig(
    tpu::TPUEmbeddingConfiguration* config) {
  if (config->feature_descriptor_size() == 0) {
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCounts(*config));
    PopulateFeatureDescriptors(config);
  } else {
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCountsAreEmpty(*config));
    TF_RETURN_IF_ERROR(ValidateFeatureDescriptors(*config));
    PopulateBatchSizeAndFeatureCounts(config);
  }
  return absl::OkStatus();
}
}  