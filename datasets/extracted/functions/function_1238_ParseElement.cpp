#include "tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h"
#include <string>
#include <utility>
#include <variant>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
namespace tflite {
namespace gpu {
namespace gl {
namespace object_accessor_internal {
IndexedElement ParseElement(absl::string_view input) {
  auto i = input.find('[');
  if (i == std::string::npos || input.back() != ']') {
    return {};
  }
  return {input.substr(0, i),
          absl::StrSplit(input.substr(i + 1, input.size() - i - 2), ',',
                         absl::SkipWhitespace())};
}
}  
namespace {
void MaybeConvertToHalf(DataType data_type, absl::string_view value,
                        std::string* output) {
  if (data_type == DataType::FLOAT16) {
    absl::StrAppend(output, "Vec4ToHalf(", value, ")");
  } else {
    absl::StrAppend(output, value);
  }
}
void MaybeConvertFromHalf(DataType data_type, absl::string_view value,
                          std::string* output) {
  if (data_type == DataType::FLOAT16) {
    absl::StrAppend(output, "Vec4FromHalf(", value, ")");
  } else {
    absl::StrAppend(output, value);
  }
}
struct ReadFromTextureGenerator {
  RewriteStatus operator()(size_t) const {
    if (element.indices.size() != 1) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    if (sampler_textures) {
      absl::StrAppend(result, "texelFetch(", element.object_name, ", ivec2(",
                      element.indices[0], ", 0), 0)");
    } else {
      absl::StrAppend(result, "imageLoad(", element.object_name, ", ivec2(",
                      element.indices[0], ", 0))");
    }
    return RewriteStatus::SUCCESS;
  }
  template <typename Shape>
  RewriteStatus operator()(const Shape&) const {
    if (element.indices.size() != Shape::size()) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    if (sampler_textures) {
      absl::StrAppend(result, "texelFetch(", element.object_name, ", ivec",
                      Shape::size(), "(", absl::StrJoin(element.indices, ", "),
                      "), 0)");
    } else {
      absl::StrAppend(result, "imageLoad(", element.object_name, ", ivec",
                      Shape::size(), "(", absl::StrJoin(element.indices, ", "),
                      "))");
    }
    return RewriteStatus::SUCCESS;
  }
  const object_accessor_internal::IndexedElement& element;
  const bool sampler_textures;
  std::string* result;
};
struct ReadFromBufferGenerator {
  RewriteStatus operator()(size_t) const {
    if (element.indices.size() != 1) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    MaybeConvertFromHalf(
        data_type,
        absl::StrCat(element.object_name, ".data[", element.indices[0], "]"),
        result);
    return RewriteStatus::SUCCESS;
  }
  RewriteStatus operator()(const uint2& size) const {
    if (element.indices.size() == 1) {
      return (*this)(1U);
    }
    if (element.indices.size() != 2) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    MaybeConvertFromHalf(
        data_type,
        absl::StrCat(element.object_name, ".data[", element.indices[0], " + $",
                     element.object_name, "_w$ * (", element.indices[1], ")]"),
        result);
    *requires_sizes = true;
    return RewriteStatus::SUCCESS;
  }
  RewriteStatus operator()(const uint3& size) const {
    if (element.indices.size() == 1) {
      return (*this)(1U);
    }
    if (element.indices.size() != 3) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    MaybeConvertFromHalf(
        data_type,
        absl::StrCat(element.object_name, ".data[", element.indices[0], " + $",
                     element.object_name, "_w$ * (", element.indices[1], " + $",
                     element.object_name, "_h$ * (", element.indices[2], "))]"),
        result);
    *requires_sizes = true;
    return RewriteStatus::SUCCESS;
  }
  DataType data_type;
  const object_accessor_internal::IndexedElement& element;
  std::string* result;
  bool* requires_sizes;
};
RewriteStatus GenerateReadAccessor(
    const Object& object,
    const object_accessor_internal::IndexedElement& element,
    bool sampler_textures, std::string* result, bool* requires_sizes) {
  switch (object.object_type) {
    case ObjectType::BUFFER:
      return std::visit(ReadFromBufferGenerator{object.data_type, element,
                                                result, requires_sizes},
                        object.size);
    case ObjectType::TEXTURE:
      return std::visit(
          ReadFromTextureGenerator{element, sampler_textures, result},
          object.size);
    case ObjectType::UNKNOWN:
      return RewriteStatus::ERROR;
  }
}
struct WriteToBufferGenerator {
  RewriteStatus operator()(size_t) const {
    if (element.indices.size() != 1) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    absl::StrAppend(result, element.object_name, ".data[", element.indices[0],
                    "] = ");
    MaybeConvertToHalf(data_type, value, result);
    return RewriteStatus::SUCCESS;
  }
  RewriteStatus operator()(const uint2& size) const {
    if (element.indices.size() == 1) {
      return (*this)(1U);
    }
    if (element.indices.size() != 2) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    absl::StrAppend(result, element.object_name, ".data[", element.indices[0],
                    " + $", element.object_name, "_w$ * (", element.indices[1],
                    ")] = ");
    MaybeConvertToHalf(data_type, value, result);
    *requires_sizes = true;
    return RewriteStatus::SUCCESS;
  }
  RewriteStatus operator()(const uint3& size) const {
    if (element.indices.size() == 1) {
      return (*this)(1U);
    }
    if (element.indices.size() != 3) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    absl::StrAppend(result, element.object_name, ".data[", element.indices[0],
                    " + $", element.object_name, "_w$ * (", element.indices[1],
                    " + $", element.object_name, "_h$ * (", element.indices[2],
                    "))] = ");
    MaybeConvertToHalf(data_type, value, result);
    *requires_sizes = true;
    return RewriteStatus::SUCCESS;
  }
  DataType data_type;
  const object_accessor_internal::IndexedElement& element;
  absl::string_view value;
  std::string* result;
  bool* requires_sizes;
};
struct WriteToTextureGenerator {
  RewriteStatus operator()(size_t) const {
    if (element.indices.size() != 1) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    absl::StrAppend(result, "imageStore(", element.object_name, ", ivec2(",
                    element.indices[0], ", 0), ", value, ")");
    return RewriteStatus::SUCCESS;
  }
  template <typename Shape>
  RewriteStatus operator()(const Shape&) const {
    if (element.indices.size() != Shape::size()) {
      result->append("WRONG_NUMBER_OF_INDICES");
      return RewriteStatus::ERROR;
    }
    absl::StrAppend(result, "imageStore(", element.object_name, ", ivec",
                    Shape::size(), "(", absl::StrJoin(element.indices, ", "),
                    "), ", value, ")");
    return RewriteStatus::SUCCESS;
  }
  const object_accessor_internal::IndexedElement& element;
  absl::string_view value;
  std::string* result;
};
RewriteStatus GenerateWriteAccessor(
    const Object& object,
    const object_accessor_internal::IndexedElement& element,
    absl::string_view value, std::string* result, bool* requires_sizes) {
  switch (object.object_type) {
    case ObjectType::BUFFER:
      return std::visit(WriteToBufferGenerator{object.data_type, element, value,
                                               result, requires_sizes},
                        object.size);
    case ObjectType::TEXTURE:
      return std::visit(WriteToTextureGenerator{element, value, result},
                        object.size);
    case ObjectType::UNKNOWN:
      return RewriteStatus::ERROR;
  }
}
std::string ToAccessModifier(AccessType access, bool use_readonly_modifier) {
  switch (access) {
    case AccessType::READ:
      return use_readonly_modifier ? " readonly" : "";
    case AccessType::WRITE:
      return " writeonly";
    case AccessType::READ_WRITE:
      return " restrict";
  }
  return " unknown_access";
}
std::string ToBufferType(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
      return "uvec4";
    case DataType::UINT64:
      return "u64vec4_not_available_in_glsl";
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
      return "ivec4";
    case DataType::INT64:
      return "i64vec4_not_available_in_glsl";
    case DataType::FLOAT16:
      return "uvec2";
    case DataType::BOOL:
    case DataType::FLOAT32:
      return "vec4";
    case DataType::FLOAT64:
      return "dvec4";
    case DataType::UNKNOWN:
      return "unknown_buffer_type";
  }
}
struct TextureImageTypeGetter {
  std::string operator()(size_t) const {
    return (*this)(uint2());
  }
  std::string operator()(const uint2&) const {
    switch (type) {
      case DataType::UINT16:
      case DataType::UINT32:
        return "uimage2D";
      case DataType::INT16:
      case DataType::INT32:
        return "iimage2D";
      case DataType::FLOAT16:
      case DataType::FLOAT32:
        return "image2D";
      default:
        return "unknown_image_2d";
    }
  }
  std::string operator()(const uint3&) const {
    switch (type) {
      case DataType::UINT16:
      case DataType::UINT32:
        return "uimage2DArray";
      case DataType::INT16:
      case DataType::INT32:
        return "iimage2DArray";
      case DataType::FLOAT16:
      case DataType::FLOAT32:
        return "image2DArray";
      default:
        return "unknown_image_2d_array";
    }
  }
  DataType type;
};
struct TextureSamplerTypeGetter {
  std::string operator()(size_t) const {
    return (*this)(uint2());
  }
  std::string operator()(const uint2&) const {
    switch (type) {
      case DataType::FLOAT16:
      case DataType::FLOAT32:
        return "sampler2D";
      case DataType::INT32:
      case DataType::INT16:
        return "isampler2D";
      case DataType::UINT32:
      case DataType::UINT16:
        return "usampler2D";
      default:
        return "unknown_sampler2D";
    }
  }
  std::string operator()(const uint3&) const {
    switch (type) {
      case DataType::FLOAT16:
      case DataType::FLOAT32:
        return "sampler2DArray";
      case DataType::INT32:
      case DataType::INT16:
        return "isampler2DArray";
      case DataType::UINT32:
      case DataType::UINT16:
        return "usampler2DArray";
      default:
        return "unknown_sampler2DArray";
    }
  }
  DataType type;
};
std::string ToImageType(const Object& object, bool sampler_textures) {
  if (sampler_textures && (object.access == AccessType::READ)) {
    return std::visit(TextureSamplerTypeGetter{object.data_type}, object.size);
  } else {
    return std::visit(TextureImageTypeGetter{object.data_type}, object.size);
  }
}
std::string ToImageLayoutQualifier(DataType type) {
  switch (type) {
    case DataType::UINT16:
      return "rgba16ui";
    case DataType::UINT32:
      return "rgba32ui";
    case DataType::INT16:
      return "rgba16i";
    case DataType::INT32:
      return "rgba32i";
    case DataType::FLOAT16:
      return "rgba16f";
    case DataType::FLOAT32:
      return "rgba32f";
    default:
      return "unknown_image_layout";
  }
}
std::string ToImagePrecision(DataType type) {
  switch (type) {
    case DataType::UINT16:
    case DataType::INT16:
    case DataType::FLOAT16:
      return "mediump";
    case DataType::UINT32:
    case DataType::INT32:
    case DataType::FLOAT32:
      return "highp";
    default:
      return "unknown_image_precision";
  }
}
struct SizeParametersAdder {
  void operator()(size_t) const {}
  void operator()(const uint2& size) const {
    variable_accessor->AddUniformParameter(
        {absl::StrCat(object_name, "_w"), static_cast<int32_t>(size.x)});
  }
  void operator()(const uint3& size) const {
    variable_accessor->AddUniformParameter(
        {absl::StrCat(object_name, "_w"), static_cast<int32_t>(size.x)});
    variable_accessor->AddUniformParameter(
        {absl::StrCat(object_name, "_h"), static_cast<int32_t>(size.y)});
  }
  absl::string_view object_name;
  VariableAccessor* variable_accessor;
};
void AddSizeParameters(absl::string_view object_name, const Object& object,
                       VariableAccessor* parameters) {
  std::visit(SizeParametersAdder{object_name, parameters}, object.size);
}
void GenerateObjectDeclaration(absl::string_view name, const Object& object,
                               std::string* declaration, bool is_mali,
                               bool sampler_textures) {
  switch (object.object_type) {
    case ObjectType::BUFFER:
      absl::StrAppend(declaration, "layout(binding = ", object.binding, ")",
                      ToAccessModifier(object.access, !is_mali), " buffer B",
                      object.binding, " { ", ToBufferType(object.data_type),
                      " data[]; } ", name, ";\n");
      break;
    case ObjectType::TEXTURE:
      if (sampler_textures && (object.access == AccessType::READ)) {
        absl::StrAppend(declaration, "layout(binding = ", object.binding,
                        ") uniform ", ToImagePrecision(object.data_type), " ",
                        ToImageType(object, sampler_textures), " ", name,
                        ";\n");
      } else {
        absl::StrAppend(
            declaration, "layout(", ToImageLayoutQualifier(object.data_type),
            ", binding = ", object.binding, ")",
            ToAccessModifier(object.access, true), " uniform ",
            ToImagePrecision(object.data_type), " ",
            ToImageType(object, sampler_textures), " ", name, ";\n");
      }
      break;
    case ObjectType::UNKNOWN:
      break;
  }
}
}  
RewriteStatus ObjectAccessor::Rewrite(absl::string_view input,
                                      std::string* output) {
  std::pair<absl::string_view, absl::string_view> n =
      absl::StrSplit(input, absl::MaxSplits('=', 1), absl::SkipWhitespace());
  if (n.first.empty()) {
    return RewriteStatus::NOT_RECOGNIZED;
  }
  if (n.second.empty()) {
    return RewriteRead(absl::StripAsciiWhitespace(n.first), output);
  }
  return RewriteWrite(absl::StripAsciiWhitespace(n.first),
                      absl::StripAsciiWhitespace(n.second), output);
}
RewriteStatus ObjectAccessor::RewriteRead(absl::string_view location,
                                          std::string* output) {
  auto element = object_accessor_internal::ParseElement(location);
  if (element.object_name.empty()) {
    return RewriteStatus::NOT_RECOGNIZED;
  }
  auto it = name_to_object_.find(
      std::string(element.object_name.data(), element.object_name.size()));
  if (it == name_to_object_.end()) {
    return RewriteStatus::NOT_RECOGNIZED;
  }
  bool requires_sizes = false;
  auto status = GenerateReadAccessor(it->second, element, sampler_textures_,
                                     output, &requires_sizes);
  if (requires_sizes) {
    AddSizeParameters(it->first, it->second, variable_accessor_);
  }
  return status;
}
RewriteStatus ObjectAccessor::RewriteWrite(absl::string_view location,
                                           absl::string_view value,
                                           std::string* output) {
  auto element = object_accessor_internal::ParseElement(location);
  if (element.object_name.empty()) {
    return RewriteStatus::NOT_RECOGNIZED;
  }
  auto it = name_to_object_.find(
      std::string(element.object_name.data(), element.object_name.size()));
  if (it == name_to_object_.end()) {
    return RewriteStatus::NOT_RECOGNIZED;
  }
  bool requires_sizes = false;
  auto status = GenerateWriteAccessor(it->second, element, value, output,
                                      &requires_sizes);
  if (requires_sizes) {
    AddSizeParameters(it->first, it->second, variable_accessor_);
  }
  return status;
}
bool ObjectAccessor::AddObject(const std::string& name, Object object) {
  if (object.object_type == ObjectType::UNKNOWN) {
    return false;
  }
  return name_to_object_.insert({name, std::move(object)}).second;
}
std::string ObjectAccessor::GetObjectDeclarations() const {
  std::string declarations;
  for (auto& o : name_to_object_) {
    GenerateObjectDeclaration(o.first, o.second, &declarations, is_mali_,
                              sampler_textures_);
  }
  return declarations;
}
std::string ObjectAccessor::GetFunctionsDeclarations() const {
  for (const auto& o : name_to_object_) {
    if (o.second.data_type == DataType::FLOAT16 &&
        o.second.object_type == ObjectType::BUFFER) {
      return absl::StrCat(
          "#define Vec4FromHalf(v) vec4(unpackHalf2x16(v.x), "
          "unpackHalf2x16(v.y))\n",
          "#define Vec4ToHalf(v) uvec2(packHalf2x16(v.xy), "
          "packHalf2x16(v.zw))");
    }
  }
  return "";
}
std::vector<Object> ObjectAccessor::GetObjects() const {
  std::vector<Object> objects;
  objects.reserve(name_to_object_.size());
  for (auto& o : name_to_object_) {
    objects.push_back(o.second);
  }
  return objects;
}
}  
}  
}  