#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include <utility>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
namespace stablehlo::quantization {
namespace {
void PopulateDefaultCalibrationOptions(QuantizationConfig& quant_config) {
  if (!quant_config.has_calibration_options() ||
      quant_config.calibration_options().calibration_method() ==
          CalibrationOptions::CALIBRATION_METHOD_UNSPECIFIED) {
    quant_config.mutable_calibration_options()->set_calibration_method(
        CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);
  }
  switch (quant_config.calibration_options().calibration_method()) {
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC:
      if (quant_config.calibration_options()
              .calibration_parameters()
              .num_bins() == 0) {
        quant_config.mutable_calibration_options()
            ->mutable_calibration_parameters()
            ->set_num_bins(512);
      }
      if (quant_config.calibration_options().calibration_method() ==
          CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE) {
        if (quant_config.calibration_options()
                .calibration_parameters()
                .min_percentile() == 0) {
          quant_config.mutable_calibration_options()
              ->mutable_calibration_parameters()
              ->set_min_percentile(0.001);
        }
        if (quant_config.calibration_options()
                .calibration_parameters()
                .max_percentile() == 0) {
          quant_config.mutable_calibration_options()
              ->mutable_calibration_parameters()
              ->set_max_percentile(99.999);
        }
      }
      break;
    default:
      break;
  }
}
QuantizationSpec GetDefaultStaticRangePtqSpec(StaticRangePtqPreset preset) {
  QuantizationSpec spec{};
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      preset.enable_full_int_quantization() ? ".*"
                                            : "^.*(dot_general|gather).*");
  spec.mutable_method()->mutable_static_range_ptq();
  return spec;
}
QuantizationSpec GetDefaultWeightOnlyPtqSpec() {
  QuantizationSpec spec{};
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      "^.*(conv|dot_general).*");
  WeightOnlyPtq& weight_only_ptq_spec =
      *spec.mutable_method()->mutable_weight_only_ptq();
  if (auto [iter, inserted] =
          weight_only_ptq_spec.mutable_input_quantized_types()->try_emplace(1);
      inserted) {
    iter->second.mutable_dimension_specs();
  }
  return spec;
}
QuantizationSpec GetPtqSpecForConvolution(Method::MethodCase method_case) {
  QuantizationSpec spec{};
  if (method_case != Method::kStaticRangePtq) {
    return spec;
  }
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      "composite_conv.*");
  QuantizedType conv_weight_quantized_type{};
  conv_weight_quantized_type.mutable_dimension_specs()->set_dimension(3);
  StaticRangePtq& static_range_ptq_spec =
      *spec.mutable_method()->mutable_static_range_ptq();
  static_range_ptq_spec.mutable_input_quantized_types()->try_emplace(
      1, std::move(conv_weight_quantized_type));
  return spec;
};
void ExpandStaticRangePtqPreset(const StaticRangePtqPreset& preset,
                                QuantizationConfig& config) {
  if (config.calibration_options().representative_datasets().empty()) {
    auto preset_datasets = preset.representative_datasets();
    config.mutable_calibration_options()
        ->mutable_representative_datasets()
        ->Add(preset_datasets.begin(), preset_datasets.end());
  }
  QuantizationSpecs new_specs{};
  *new_specs.add_specs() =
      GetDefaultStaticRangePtqSpec(config.static_range_ptq_preset());
  *new_specs.add_specs() =
      GetPtqSpecForConvolution(Method::MethodCase::kStaticRangePtq);
  const QuantizationSpecs& previous_specs = config.specs();
  new_specs.mutable_specs()->Add(previous_specs.specs().begin(),
                                 previous_specs.specs().end());
  config.clear_static_range_ptq_preset();
  config.mutable_specs()->Swap(&new_specs);
}
void ExpandWeightOnlyPtqPreset(QuantizationConfig& config) {
  QuantizationSpecs new_specs{};
  *new_specs.add_specs() = GetDefaultWeightOnlyPtqSpec();
  const QuantizationSpecs& previous_specs = config.specs();
  new_specs.mutable_specs()->Add(previous_specs.specs().begin(),
                                 previous_specs.specs().end());
  config.clear_weight_only_ptq_preset();
  config.mutable_specs()->Swap(&new_specs);
}
}  
QuantizationConfig ExpandPresets(const QuantizationConfig& config) {
  QuantizationConfig new_config = config;
  switch (config.preset_case()) {
    case QuantizationConfig::kStaticRangePtqPreset:
      ExpandStaticRangePtqPreset(config.static_range_ptq_preset(), new_config);
      break;
    case QuantizationConfig::kWeightOnlyPtqPreset:
      ExpandWeightOnlyPtqPreset(new_config);
      break;
    default:
      break;
  }
  return new_config;
}
bool HasQuantizationMethod(const QuantizationSpecs& specs,
                           Method::MethodCase method_case) {
  for (const auto& spec : specs.specs()) {
    if (spec.method().method_case() == method_case) {
      return true;
    }
  }
  return false;
}
QuantizationConfig PopulateDefaults(
    const QuantizationConfig& user_provided_config) {
  QuantizationConfig config = user_provided_config;
  PopulateDefaultCalibrationOptions(config);
  PipelineConfig& pipeline_config = *config.mutable_pipeline_config();
  if (!pipeline_config.has_unpack_quantized_types()) {
    pipeline_config.set_unpack_quantized_types(true);
  }
  return config;
}
}  