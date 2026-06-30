#include "tensorflow/compiler/aot/codegen.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/aot/embedded_protocol_buffers.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/buffer_info_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace tfcompile {
namespace {
using BufferInfo = xla::cpu_function_runtime::BufferInfo;
bool IsAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
bool IsAlphaNum(char c) { return IsAlpha(c) || (c >= '0' && c <= '9'); }
Status XLATypeToCpp(xla::PrimitiveType type, string* str) {
  switch (type) {
    case xla::PRED:
      *str = "bool";
      break;
    case xla::S8:
      *str = "tensorflow::int8";
      break;
    case xla::S16:
      *str = "tensorflow::int16";
      break;
    case xla::S32:
      *str = "tensorflow::int32";
      break;
    case xla::S64:
      *str = "int64_t";
      break;
    case xla::U8:
      *str = "tensorflow::uint8";
      break;
    case xla::U16:
      *str = "tensorflow::uint16";
      break;
    case xla::U32:
      *str = "tensorflow::uint32";
      break;
    case xla::U64:
      *str = "tensorflow::uint64";
      break;
    case xla::F32:
      *str = "float";
      break;
    case xla::F64:
      *str = "double";
      break;
    default:
      return errors::Unimplemented("XLA type ", xla::PrimitiveType_Name(type),
                                   " has no equivalent in C++");
  }
  return absl::OkStatus();
}
size_t TotalBufferBytes(const std::vector<BufferInfo>& buffer_infos) {
  return std::accumulate(buffer_infos.begin(), buffer_infos.end(), size_t{0},
                         [](size_t size, const BufferInfo& buffer_info) {
                           return size + buffer_info.size();
                         });
}
std::vector<BufferInfo> ExtractEntryParamBufferInfos(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<BufferInfo> result;
  std::copy_if(buffer_infos.begin(), buffer_infos.end(),
               std::back_inserter(result), [](const BufferInfo& buffer_info) {
                 return buffer_info.is_entry_parameter();
               });
  return result;
}
std::vector<BufferInfo> ExtractTempBufferInfos(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<BufferInfo> result;
  std::copy_if(buffer_infos.begin(), buffer_infos.end(),
               std::back_inserter(result), [](const BufferInfo& buffer_info) {
                 return buffer_info.is_temp_buffer();
               });
  return result;
}
Status AddRewritesForShape(int i, const xla::Shape& shape,
                           std::vector<std::pair<string, string>>* rewrites) {
  string type;
  TF_RETURN_IF_ERROR(XLATypeToCpp(shape.element_type(), &type));
  std::vector<string> dim_vars;
  string dim_sizes, indices;
  int count = 1;
  if (shape.rank() == 0 ||
      (shape.dimensions_size() == 1 && shape.dimensions(0) == 1)) {
    dim_sizes = "[1]";
    indices = "[0]";
  } else {
    for (int dim = 0; dim < shape.dimensions_size(); ++dim) {
      dim_vars.push_back(absl::StrCat("size_t dim", dim));
      dim_sizes += absl::StrCat("[", shape.dimensions(dim), "]");
      indices += absl::StrCat("[dim", dim, "]");
      count *= shape.dimensions(dim);
    }
  }
  rewrites->push_back({"{{I}}", absl::StrCat(i)});
  rewrites->push_back({"{{TYPE}}", type});
  rewrites->push_back({"{{DIM_VARS}}", absl::StrJoin(dim_vars, ", ")});
  rewrites->push_back({"{{DIM_SIZES}}", dim_sizes});
  rewrites->push_back({"{{INDICES}}", indices});
  rewrites->push_back({"{{COUNT}}", absl::StrCat(count)});
  return absl::OkStatus();
}
string RewriteWithName(const string& name, string code,
                       const std::vector<std::pair<string, string>>& rewrites) {
  absl::StrReplaceAll(rewrites, &code);
  absl::StrReplaceAll({{"{{NAME}}", name}}, &code);
  return code;
}
Status GenArgMethods(const tf2xla::Config& config,
                     const xla::ProgramShapeProto& ps,
                     const CompileResult& compile_result, string* methods) {
  const int num_args = ps.parameters_size();
  if (config.feed_size() + config.variable_size() < num_args) {
    return errors::InvalidArgument(
        "mismatch between feed_size(", config.feed_size(), ")+variable_size(",
        config.variable_size(), ") and num_args(", num_args, ")");
  }
  for (int i = 0; i < config.feed_size(); ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(
        AddRewritesForShape(i, xla::Shape(ps.parameters(i)), &rewrites));
    const string code = R"(
  void set_arg{{NAME}}_data(const void* data) {
    set_arg_data({{I}}, data);
  }
  {{TYPE}}* arg{{NAME}}_data() {
    return static_cast<{{TYPE}}*>(arg_data({{I}}));
  }
  {{TYPE}}& arg{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* arg{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(arg_data({{I}}));
  }
  const {{TYPE}}& arg{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  int arg{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int arg{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    *methods += RewriteWithName(absl::StrCat(i), code, rewrites);
    if (!config.feed(i).name().empty()) {
      *methods += RewriteWithName("_" + config.feed(i).name(), code, rewrites);
    }
  }
  return absl::OkStatus();
}
Status GenResultMethods(const tf2xla::Config& config,
                        const xla::ProgramShapeProto& ps, string* methods) {
  if (ps.result().element_type() != xla::TUPLE) {
    return errors::Internal("codegen requires the XLA result to be a tuple");
  }
  size_t num_results = ps.result().tuple_shapes_size();
  int readonly_variables = absl::c_count_if(
      config.variable(),
      [](const tf2xla::Variable& var) { return var.readonly(); });
  const int actual_num_results =
      config.fetch_size() + config.variable_size() - readonly_variables;
  if (actual_num_results != num_results) {
    return errors::InvalidArgument("mismatch between fetch_size(",
                                   config.fetch_size(), ")+variable_size(",
                                   config.variable_size(), ") and tuple_size(",
                                   ps.result().tuple_shapes_size(), ")");
  }
  for (int i = 0; i < config.fetch_size(); ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(AddRewritesForShape(
        i, xla::Shape(ps.result().tuple_shapes(i)), &rewrites));
    string code = R"(
  {{TYPE}}* result{{NAME}}_data() {
    return static_cast<{{TYPE}}*>(result_data({{I}}));
  }
  {{TYPE}}& result{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{TYPE}}(*){{DIM_SIZES}}>(
        result_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* result{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(result_data({{I}}));
  }
  const {{TYPE}}& result{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        result_data({{I}}))){{INDICES}};
  }
  int result{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int result{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    *methods += RewriteWithName(absl::StrCat(i), code, rewrites);
    if (!config.fetch(i).name().empty()) {
      *methods += RewriteWithName("_" + config.fetch(i).name(), code, rewrites);
    }
  }
  return absl::OkStatus();
}
Status GenVariableMethods(const tf2xla::Config& config,
                          const xla::ProgramShapeProto& ps, string* methods) {
  const int num_args = ps.parameters_size();
  for (int i = config.feed_size(); i < num_args; ++i) {
    std::vector<std::pair<string, string>> rewrites;
    TF_RETURN_IF_ERROR(
        AddRewritesForShape(i, xla::Shape(ps.parameters(i)), &rewrites));
    const string code = R"(
  void set_var_{{NAME}}_data({{MAYBE_CONST}}{{TYPE}}* data) {
    set_arg_data({{I}}, data);
  }
  {{MAYBE_CONST}}{{TYPE}}* var_{{NAME}}_data() {
    return static_cast<{{MAYBE_CONST}}{{TYPE}}*>(arg_data({{I}}));
  }
  {{MAYBE_CONST}}{{TYPE}}& var_{{NAME}}({{DIM_VARS}}) {
    return (*static_cast<{{MAYBE_CONST}}{{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  const {{TYPE}}* var_{{NAME}}_data() const {
    return static_cast<const {{TYPE}}*>(arg_data({{I}}));
  }
  const {{TYPE}}& var_{{NAME}}({{DIM_VARS}}) const {
    return (*static_cast<const {{TYPE}}(*){{DIM_SIZES}}>(
        arg_data({{I}}))){{INDICES}};
  }
  int var_{{NAME}}_size() const {
    return {{COUNT}} * sizeof({{TYPE}});
  }
  int var_{{NAME}}_count() const {
    return {{COUNT}};
  }
)";
    const tf2xla::Variable& var = config.variable(i - config.feed_size());
    rewrites.emplace_back("{{MAYBE_CONST}}", var.readonly() ? "const " : "");
    *methods += RewriteWithName(
        var.name().empty() ? var.node_name() : var.name(), code, rewrites);
  }
  return absl::OkStatus();
}
Status GenArgShapeInfos(const xla::ProgramShapeProto& ps, string* infos) {
  for (int i = 0; i < ps.parameters_size(); ++i) {
    const xla::ShapeProto& shape = ps.parameters(i);
    if (shape.element_type() == xla::TUPLE) {
      return absl::InternalError(
          absl::StrCat("parameter ", i,
                       ": codegen requires XLA parameters to "
                       "be non-tuples."));
    }
    *infos += absl::Substitute(R"(  static constexpr int32_t kArg$0Shapes[] = {
$1
  };
)",
                               i,
                               shape.dimensions_size() > 0
                                   ? absl::StrJoin(shape.dimensions(), ", ")
                                   : "-1");
  }
  *infos += R"(  static const ShapeInfo* ArgShapeInfos() {
    static constexpr ShapeInfo kArgShapeInfoTable[kNumArgs] = {
)";
  for (int i = 0; i < ps.parameters_size(); ++i) {
    const xla::ShapeProto& shape = ps.parameters(i);
    *infos +=
        absl::Substitute("{ kArg$0Shapes, $1 },\n", i, shape.dimensions_size());
  }
  *infos += R"(    };
    return kArgShapeInfoTable;
  })";
  return absl::OkStatus();
}
Status GenResultShapeInfos(const xla::ProgramShapeProto& ps, string* infos) {
  if (ps.result().element_type() != xla::TUPLE) {
    return absl::InternalError("codegen requires the XLA result to be a tuple");
  }
  for (int i = 0; i < ps.result().tuple_shapes_size(); ++i) {
    const xla::ShapeProto& shape = ps.result().tuple_shapes(i);
    *infos += absl::Substitute(
        R"(  static constexpr int32_t kResult$0Shapes[] = {
$1
  };
)",
        i,
        shape.dimensions_size() > 0 ? absl::StrJoin(shape.dimensions(), ", ")
                                    : "-1");
  }
  *infos += R"(  static const ShapeInfo* ResultShapeInfos() {
    static constexpr ShapeInfo kResultShapeInfoTable[kNumResults] = {
)";
  for (int i = 0; i < ps.result().tuple_shapes_size(); ++i) {
    const xla::ShapeProto& shape = ps.result().tuple_shapes(i);
    *infos += absl::Substitute("{ kResult$0Shapes, $1 },\n", i,
                               shape.dimensions_size());
  }
  *infos += R"(    };
    return kResultShapeInfoTable;
  })";
  return absl::OkStatus();
}
template <typename T>
string GenNameToIndexCode(const T& entries, bool generate) {
  if (!generate) {
    return "{\n    return nullptr;\n  }";
  }
  int end = entries.size();
  for (int i = entries.size() - 1; i >= 0; --i) {
    if (!entries[i].name().empty()) {
      break;
    }
    end = i;
  }
  string code = "{\n    static const char* kNames[] = {";
  for (int i = 0; i < end; ++i) {
    if (i > 0) {
      code += ", ";
    }
    code += "\"";
    code += entries[i].name();
    code += "\"";
  }
  if (end > 0) {
    code += ", ";
  }
  code += "nullptr};\n    return kNames;\n  }";
  return code;
}
Status ValidateFeedFetchCppNames(const tf2xla::Config& config) {
  for (const tf2xla::Feed& feed : config.feed()) {
    if (!feed.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(feed.name(), "feed name"));
    }
  }
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    if (!fetch.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(fetch.name(), "fetch name"));
    }
  }
  for (const tf2xla::Variable& variable : config.variable()) {
    if (!variable.name().empty()) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(variable.name(), "variable name"));
    } else {
      TF_RETURN_IF_ERROR(
          ValidateCppIdent(variable.node_name(), "variable name"));
    }
  }
  return absl::OkStatus();
}
std::vector<string> BufferInfosToCppExpression(
    const std::vector<BufferInfo>& buffer_infos) {
  std::vector<string> buffer_infos_as_strings;
  std::transform(buffer_infos.begin(), buffer_infos.end(),
                 std::back_inserter(buffer_infos_as_strings),
                 [](const BufferInfo& buffer_info) {
                   xla::cpu_function_runtime::EncodedBufferInfo encoded =
                       buffer_info.Encode();
                   auto param_to_str = [](uint32_t param) -> std::string {
                     return param == ~0U ? "~0U" : absl::StrCat(param, "U");
                   };
                   return absl::StrCat(
                       "::xla::cpu_function_runtime::BufferInfo("
                       "::xla::cpu_function_runtime::EncodedBufferInfo{",
                       encoded.packed_kind_and_size, "ULL, ",
                       param_to_str(encoded.entry_param_number), ", ",
                       param_to_str(encoded.result_param_number), "})");
                 });
  return buffer_infos_as_strings;
}
Status CheckEqual(size_t a, size_t b, absl::string_view error_msg) {
  if (a != b) {
    return absl::InternalError(
        absl::StrCat(error_msg, ". Expected ", a, ", got ", b, "."));
  }
  return absl::OkStatus();
}
}  
Status GenerateHeader(const CodegenOpts& opts, const tf2xla::Config& config,
                      const CompileResult& compile_result,
                      const MetadataResult& metadata_result, string* header) {
  TF_RETURN_IF_ERROR(ValidateConfig(config));
  TF_RETURN_IF_ERROR(ValidateFeedFetchCppNames(config));
  const int64_t result_index = compile_result.aot->result_buffer_index();
  const std::vector<BufferInfo>& buffer_infos =
      compile_result.aot->buffer_infos();
  const std::vector<int32> arg_index_table =
      ::xla::cpu::CreateArgIndexTableFromBufferInfos(buffer_infos);
  const std::vector<int32> result_index_table =
      ::xla::cpu::CreateResultIndexTableFromBufferInfos(buffer_infos);
  std::vector<string> buffer_infos_as_strings =
      BufferInfosToCppExpression(buffer_infos);
  const int64_t buffer_infos_size = buffer_infos.size();
  if (result_index < 0 || result_index >= buffer_infos_size) {
    return errors::InvalidArgument("result index: ", result_index,
                                   " is outside the range of temp sizes: [0,",
                                   buffer_infos.size(), ")");
  }
  std::vector<BufferInfo> buffer_infos_for_args =
      ExtractEntryParamBufferInfos(buffer_infos);
  std::vector<BufferInfo> buffer_infos_for_temps =
      ExtractTempBufferInfos(buffer_infos);
  const xla::ProgramShapeProto& ps = compile_result.program_shape;
  string methods_arg, methods_result, methods_variable;
  TF_RETURN_IF_ERROR(GenArgMethods(config, ps, compile_result, &methods_arg));
  TF_RETURN_IF_ERROR(GenResultMethods(config, ps, &methods_result));
  TF_RETURN_IF_ERROR(GenVariableMethods(config, ps, &methods_variable));
  string arg_shape_infos, result_shape_infos;
  TF_RETURN_IF_ERROR(GenArgShapeInfos(ps, &arg_shape_infos));
  TF_RETURN_IF_ERROR(
      CheckEqual(ps.parameters_size(), arg_index_table.size(),
                 "Arg number mismatch, proto vs. arg_index_table"));
  TF_RETURN_IF_ERROR(GenResultShapeInfos(ps, &result_shape_infos));
  TF_RETURN_IF_ERROR(
      CheckEqual(ps.result().tuple_shapes_size(), result_index_table.size(),
                 "Result number mismatch, proto vs. result_index_table"));
  const size_t arg_bytes_aligned =
      xla::cpu_function_runtime::AlignedBufferBytes(
          buffer_infos_for_args.data(), buffer_infos_for_args.size(),
          true);
  const size_t arg_bytes_total = TotalBufferBytes(buffer_infos_for_args);
  const size_t temp_bytes_aligned =
      xla::cpu_function_runtime::AlignedBufferBytes(
          buffer_infos_for_temps.data(), buffer_infos_for_temps.size(),
          true);
  const size_t temp_bytes_total = TotalBufferBytes(buffer_infos_for_temps);
  string ns_start;
  for (const string& n : opts.namespaces) {
    ns_start += absl::StrCat("namespace ", n, " {\n");
  }
  ns_start += "\n";
  string ns_end("\n");
  for (int i = opts.namespaces.size() - 1; i >= 0; --i) {
    const string& n = opts.namespaces[i];
    ns_end += absl::StrCat("}  
  }
  const string arg_names_code =
      GenNameToIndexCode(config.feed(), opts.gen_name_to_index);
  auto variable_copy = config.variable();
  for (auto& var : variable_copy) {
    if (var.name().empty()) {
      var.set_name(var.node_name());
    }
  }
  const string variable_names_code =
      GenNameToIndexCode(variable_copy, opts.gen_name_to_index);
  const string result_names_code =
      GenNameToIndexCode(config.fetch(), opts.gen_name_to_index);
  const string include_xla_data_proto =
      opts.gen_program_shape
          ? R"(#include "xla/xla_data.pb.h")"
          : "";
  const string include_hlo_profile_printer_data_proto =
      opts.gen_hlo_profile_printer_data
          ? R"(#include "xla/service/hlo_profile_printer_data.pb.h")"
          : "";
  const string assign_profile_counters_size =
      opts.gen_hlo_profile_printer_data
          ? "set_static_data_profile_counters_size(data, "
            "get_static_data_hlo_profile_printer_data(data)->"
            "profile_counters_size());"
          : "";
  *header =
      R"(
#ifndef TFCOMPILE_GENERATED_{{ENTRY}}_H_  
#define TFCOMPILE_GENERATED_{{ENTRY}}_H_  
{{INCLUDE_XLA_DATA_PROTO}}
{{INCLUDE_HLO_PROFILE_PRINTER_DATA_PROTO}}
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/platform/types.h"
namespace Eigen { struct ThreadPoolDevice; }
namespace xla { class ExecutableRunOptions; }
extern "C" void {{ENTRY}}(
    void* result, const ::xla::ExecutableRunOptions* run_options,
    const void** args, void** temps, XlaCustomCallStatus* status,
    int64_t* profile_counters);
{{DECLS_FROM_OBJ_FILE}}
{{NS_START}}
class {{CLASS}} final : public tensorflow::XlaCompiledCpuFunction {
 public:
  static constexpr size_t kNumArgs = {{ARG_NUM}};
  static constexpr size_t kNumResults = {{RESULT_NUM}};
  static constexpr size_t kNumVariables = {{VARIABLE_NUM}};
  static const ::int64_t ArgSize(::tensorflow::int32 index) {
    return BufferInfos()[ArgIndexToBufferIndex()[index]].size();
  }
  static const tensorflow::XlaCompiledCpuFunction::StaticData& StaticData() {
    static XlaCompiledCpuFunction::StaticData* kStaticData = [](){
      XlaCompiledCpuFunction::StaticData* data =
        new XlaCompiledCpuFunction::StaticData;
      set_static_data_raw_function(data, {{ENTRY}});
      set_static_data_buffer_infos(data, BufferInfos());
      set_static_data_num_buffers(data, kNumBuffers);
      set_static_data_result_index_table(data, ResultIndexToBufferIndex());
      set_static_data_num_results(data, kNumResults);
      set_static_data_arg_index_table(data, ArgIndexToBufferIndex());
      set_static_data_num_args(data, kNumArgs);
      set_static_data_num_variables(data, kNumVariables);
      set_static_data_result_index(data, kResultIndex);
      set_static_data_arg_shape_infos(data, ArgShapeInfos());
      set_static_data_result_shape_infos(data, ResultShapeInfos());
      set_static_data_arg_names(data, StaticArgNames());
      set_static_data_variable_names(data, StaticVariableNames());
      set_static_data_result_names(data, StaticResultNames());
      set_static_data_program_shape(data, StaticProgramShape());
      set_static_data_hlo_profile_printer_data(
          data, StaticHloProfilePrinterData());
{{ASSIGN_PROFILE_COUNTERS_SIZE}}
      return data;
    }();
    return *kStaticData;
  }
  {{CLASS}}(AllocMode alloc_mode =
            AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS)
      : XlaCompiledCpuFunction(StaticData(), alloc_mode) {}
  {{CLASS}}(const {{CLASS}}&) = delete;
  {{CLASS}}& operator=(const {{CLASS}}&) = delete;
{{METHODS_ARG}}
{{METHODS_RESULT}}
{{METHODS_VARIABLE}}
 private:
  static constexpr size_t kNumBuffers = {{NUM_BUFFERS}};
  static const ::xla::cpu_function_runtime::BufferInfo* BufferInfos() {
    static const ::xla::cpu_function_runtime::BufferInfo
      kBufferInfos[kNumBuffers] = {
{{BUFFER_INFOS_AS_STRING}}
      };
    return kBufferInfos;
  }
  static const ::tensorflow::int32* ResultIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kResultIndexToBufferIndex[kNumResults] = {
{{RESULT_INDEX_TABLE}}
    };
    return kResultIndexToBufferIndex;
  }
  static const ::tensorflow::int32* ArgIndexToBufferIndex() {
    static constexpr ::tensorflow::int32 kArgIndexToBufferIndex[kNumArgs] = {
{{ARG_INDEX_TABLE}}
    };
    return kArgIndexToBufferIndex;
  }
  static constexpr size_t kResultIndex = {{RESULT_INDEX}};
{{ARG_SHAPE_INFOS}};
{{RESULT_SHAPE_INFOS}};
  static const char** StaticArgNames() {{ARG_NAMES_CODE}}
  static const char** StaticVariableNames() {{VARIABLE_NAMES_CODE}}
  static const char** StaticResultNames() {{RESULT_NAMES_CODE}}
  static const ::xla::ProgramShapeProto* StaticProgramShape() {
    static const ::xla::ProgramShapeProto* kShape = {{PROGRAM_SHAPE_SHIM_EXPRESSION}};
    return kShape;
  }
  static const ::xla::HloProfilePrinterData* StaticHloProfilePrinterData() {
    static const ::xla::HloProfilePrinterData* kHloProfilePrinterData =
      {{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}};
    return kHloProfilePrinterData;
  }
};
{{NS_END}}
#endif  
)";
  const std::vector<std::pair<string, string>> rewrites = {
      {"{{ARG_BYTES_ALIGNED}}", absl::StrCat(arg_bytes_aligned)},
      {"{{ARG_BYTES_TOTAL}}", absl::StrCat(arg_bytes_total)},
      {"{{ARG_NAMES_CODE}}", arg_names_code},
      {"{{ARG_NUM}}", absl::StrCat(arg_index_table.size())},
      {"{{ARG_SHAPE_INFOS}}", arg_shape_infos},
      {"{{VARIABLE_NUM}}", absl::StrCat(config.variable_size())},
      {"{{ARG_INDEX_TABLE}}", absl::StrJoin(arg_index_table, ", ")},
      {"{{RESULT_NUM}}", absl::StrCat(result_index_table.size())},
      {"{{RESULT_INDEX_TABLE}}", absl::StrJoin(result_index_table, ", ")},
      {"{{ASSIGN_PROFILE_COUNTERS_SIZE}}", assign_profile_counters_size},
      {"{{CLASS}}", opts.class_name},
      {"{{DECLS_FROM_OBJ_FILE}}",
       absl::StrJoin(metadata_result.header_variable_decls, "\n")},
      {"{{ENTRY}}", compile_result.entry_point},
      {"{{HLO_PROFILE_PRINTER_DATA_SHIM_EXPRESSION}}",
       metadata_result.hlo_profile_printer_data_access_shim},
      {"{{INCLUDE_XLA_DATA_PROTO}}", include_xla_data_proto},
      {"{{INCLUDE_HLO_PROFILE_PRINTER_DATA_PROTO}}",
       include_hlo_profile_printer_data_proto},
      {"{{METHODS_ARG}}\n", methods_arg},
      {"{{METHODS_RESULT}}\n", methods_result},
      {"{{METHODS_VARIABLE}}\n", methods_variable},
      {"{{NS_END}}\n", ns_end},
      {"{{NS_START}}\n", ns_start},
      {"{{PROGRAM_SHAPE}}", xla::ShapeUtil::HumanString(xla::ProgramShape(ps))},
      {"{{PROGRAM_SHAPE_SHIM_EXPRESSION}}",
       metadata_result.program_shape_access_shim},
      {"{{VARIABLE_NAMES_CODE}}", variable_names_code},
      {"{{RESULT_INDEX}}", absl::StrCat(result_index)},
      {"{{RESULT_NAMES_CODE}}", result_names_code},
      {"{{RESULT_SHAPE_INFOS}}", result_shape_infos},
      {"{{TEMP_BYTES_ALIGNED}}", absl::StrCat(temp_bytes_aligned)},
      {"{{TEMP_BYTES_TOTAL}}", absl::StrCat(temp_bytes_total)},
      {"{{NUM_BUFFERS}}", absl::StrCat(buffer_infos.size())},
      {"{{BUFFER_INFOS_AS_STRING}}",
       absl::StrJoin(buffer_infos_as_strings, ",\n")}};
  absl::StrReplaceAll(rewrites, header);
  return absl::OkStatus();
}
static string CreateUniqueIdentifier(const CodegenOpts& opts,
                                     absl::string_view suffix) {
  string result = "__tfcompile";
  for (const string& n : opts.namespaces) {
    absl::StrAppend(&result, "_", n);
  }
  absl::StrAppend(&result, "_", opts.class_name, "_", suffix);
  return result;
}
Status GenerateMetadata(const CodegenOpts& opts,
                        const CompileResult& compile_result,
                        MetadataResult* metadata_result) {
  std::unique_ptr<xla::ProgramShapeProto> program_shape;
  if (opts.gen_program_shape) {
    program_shape =
        std::make_unique<xla::ProgramShapeProto>(compile_result.program_shape);
    program_shape->clear_parameter_names();
  }
  ProtobufToEmbed program_shape_protobuf{
      CreateUniqueIdentifier(opts, "ProgramShapeProto"),
      "::xla::ProgramShapeProto", program_shape.get()};
  ProtobufToEmbed hlo_profile_printer_data_protobuf{
      CreateUniqueIdentifier(opts, "HloProfilePrinterData"),
      "::xla::HloProfilePrinterData",
      compile_result.aot->hlo_profile_printer_data()};
  TF_ASSIGN_OR_RETURN(
      EmbeddedProtocolBuffers embedded_protobufs,
      CreateEmbeddedProtocolBuffers(
          opts.target_triple,
          {program_shape_protobuf, hlo_profile_printer_data_protobuf}));
  metadata_result->program_shape_access_shim =
      std::move(embedded_protobufs.cpp_shims[0].expression);
  metadata_result->hlo_profile_printer_data_access_shim =
      std::move(embedded_protobufs.cpp_shims[1].expression);
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[0].variable_decl));
  metadata_result->header_variable_decls.emplace_back(
      std::move(embedded_protobufs.cpp_shims[1].variable_decl));
  metadata_result->object_file_data =
      std::move(embedded_protobufs.object_file_data);
  return absl::OkStatus();
}
Status ParseCppClass(const string& cpp_class, string* class_name,
                     std::vector<string>* namespaces) {
  class_name->clear();
  namespaces->clear();
  if (cpp_class.empty()) {
    return errors::InvalidArgument("empty cpp_class: " + cpp_class);
  }
  std::vector<string> parts = absl::StrSplit(cpp_class, "::");
  if (parts.front().empty()) {
    parts.erase(parts.begin());
  }
  for (int i = 0, end = parts.size(); i < end; ++i) {
    if (i < end - 1) {
      TF_RETURN_IF_ERROR(ValidateCppIdent(
          parts[i], "in namespace component of cpp_class: " + cpp_class));
      namespaces->push_back(parts[i]);
    } else {
      TF_RETURN_IF_ERROR(ValidateCppIdent(
          parts[i], "in class name of cpp_class: " + cpp_class));
      *class_name = parts[i];
    }
  }
  return absl::OkStatus();
}
Status ValidateCppIdent(absl::string_view ident, absl::string_view msg) {
  if (ident.empty()) {
    return errors::InvalidArgument("empty identifier: ", msg);
  }
  if (ident[0] != '_' && !IsAlpha(ident[0])) {
    return errors::InvalidArgument("illegal leading char: ", msg);
  }
  for (size_t pos = 1; pos < ident.size(); ++pos) {
    if (ident[pos] != '_' && !IsAlphaNum(ident[pos])) {
      return errors::InvalidArgument("illegal char: ", msg);
    }
  }
  return absl::OkStatus();
}
}  
}  