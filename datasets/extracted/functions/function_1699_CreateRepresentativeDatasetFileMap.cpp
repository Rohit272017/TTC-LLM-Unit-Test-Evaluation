#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/representative_dataset.h"
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
namespace stablehlo::quantization {
using ::tensorflow::quantization::RepresentativeDatasetFile;
absl::StatusOr<absl::flat_hash_map<std::string, RepresentativeDatasetFile>>
CreateRepresentativeDatasetFileMap(absl::Span<const RepresentativeDatasetConfig>
                                       representative_dataset_configs) {
  absl::flat_hash_map<std::string, RepresentativeDatasetFile>
      repr_dataset_file_map{};
  for (const RepresentativeDatasetConfig& dataset_config :
       representative_dataset_configs) {
    RepresentativeDatasetFile repr_dataset_file;
    repr_dataset_file.set_tfrecord_file_path(dataset_config.tf_record().path());
    const std::string signature_key = dataset_config.has_signature_key()
                                          ? dataset_config.signature_key()
                                          : "serving_default";
    if (repr_dataset_file_map.contains(signature_key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("RepresentativeDatasetConfig should not contain "
                       "duplicate signature key: ",
                       signature_key));
    }
    repr_dataset_file_map[signature_key] = std::move(repr_dataset_file);
  }
  return repr_dataset_file_map;
}
}  