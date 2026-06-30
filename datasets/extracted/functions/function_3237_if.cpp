#ifndef XLA_SERVICE_PATTERN_MATCHER_H_
#define XLA_SERVICE_PATTERN_MATCHER_H_
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/utility/utility.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
namespace xla {
struct MatchOption {
  bool capture;
  bool single_user_only;
  std::ostream* explain_os;
};
template <typename Value, typename Pattern>
bool Match(Value* value, const Pattern& pattern,
           MatchOption option = {true,
                                 false,
                                 nullptr}) {
  if (option.capture) {
    auto new_option = option;
    new_option.capture = false;
    if (!pattern.Match(value, new_option)) {
      return false;
    }
  }
  return pattern.Match(value, option);
}
template <typename Value, typename Pattern>
bool MatchSingleUserOnly(Value* value, const Pattern& pattern) {
  MatchOption option = {true, true,
                        nullptr};
  return Match(value, pattern, option);
}
template <typename FilterPattern, typename Pattern>
bool MatchAndLogIfFailed(HloInstruction* instr, absl::string_view desc,
                         const Pattern& pattern, bool enable_logging,
                         const FilterPattern& filter_pattern) {
  bool matched = Match(instr, pattern);
  if (matched || !enable_logging || !Match(instr, filter_pattern)) {
    return matched;
  }
  std::stringstream os;
  CHECK(!Match(
      instr, pattern,
      {false, false, &os}));
  LOG(ERROR) << "Failed to match " << desc << ":\n" << os.str();
  return false;
}
namespace match {
namespace detail {
#pragma push_macro("EXPLAIN")
#define EXPLAIN \
  if (option.explain_os) *option.explain_os
enum {
  kIndentInc = 2,
};
inline void Indent(std::ostream* os, int64_t indent) {
  *os << "\n";
  for (int64_t i = 0; i < indent; ++i) {
    *os << " ";
  }
}
template <typename T, typename Dummy = void>
struct IsTrivialMatcher {
  static constexpr bool value = false;
};
template <typename T>
struct IsTrivialMatcher<T,
                        typename std::enable_if<T::kIsTrivialMatcher>::type> {
  static constexpr bool value = true;
};
template <typename Item, typename... Patterns>
class AllOfPattern {
 public:
  explicit AllOfPattern(const Patterns&... patterns) : patterns_(patterns...) {}
  bool Match(const Item* item, MatchOption option) const {
    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    DCHECK(matched || !option.capture);
    return matched;
  }
  bool Match(Item* item, MatchOption option) const {
    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    DCHECK(matched || !option.capture);
    return matched;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }
  const std::tuple<Patterns...>& patterns() const { return patterns_; }
 private:
  template <typename ItemType, size_t index>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, index>) const {
    return std::get<index>(patterns_).Match(item, option) &&
           MatchImpl(item, option, std::integral_constant<size_t, index + 1>());
  }
  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return true;
  }
  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64_t indent) const {
    constexpr bool first_is_trivial =
        IsTrivialMatcher<typename std::remove_reference<decltype(std::get<0>(
            patterns_))>::type>::value;
    constexpr bool is_last = index == sizeof...(Patterns) - 1;
    const auto& submatcher = std::get<index>(patterns_);
    auto print_bulleted_item = [&] {
      *os << " * ";
      submatcher.DescribeTo(os, indent + 3);
      if (!is_last) {
        *os << " AND";
        Indent(os, indent);
      }
    };
    if (index == 0) {
      if (first_is_trivial || is_last) {
        submatcher.DescribeTo(os, indent + kIndentInc);
        if (sizeof...(Patterns) > 2) {
          *os << ":";
          Indent(os, indent);
        }
      } else {
        *os << "all of:";
        Indent(os, indent);
        print_bulleted_item();
      }
    } else if (first_is_trivial && index == 1 && sizeof...(Patterns) == 2) {
      *os << " ";
      submatcher.DescribeTo(os, indent);
    } else {
      print_bulleted_item();
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }
  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64_t indent) const {}
  std::tuple<Patterns...> patterns_;
};
}  
template <typename Item, typename... Patterns>
auto AllOf(const Patterns&... patterns) {
  return detail::AllOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}
template <typename Item, typename... InnerPs, typename... OuterPs>
auto AllOf(const detail::AllOfPattern<Item, InnerPs...>& inner_p,
           const OuterPs&... outer_ps) {
  auto make_all_of = [](const InnerPs&... inner_ps,
                        const OuterPs&... outer_ps) {
    return detail::AllOfPattern<typename std::remove_const<Item>::type,
                                InnerPs..., OuterPs...>(inner_ps...,
                                                        outer_ps...);
  };
  return absl::apply(make_all_of, std::tuple_cat(inner_p.patterns(),
                                                 std::make_tuple(outer_ps...)));
}
namespace detail {
template <typename LayoutType, typename Impl>
class LayoutPattern;
class LayoutPatternBaseImpl {
 public:
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (layout == nullptr) {
      EXPLAIN << "Layout is null";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "a layout";
  }
  static constexpr bool kIsTrivialMatcher = true;
};
class LayoutPatternEqualImpl {
 public:
  explicit constexpr LayoutPatternEqualImpl(const ::xla::Layout* layout)
      : layout_(layout) {}
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (!LayoutUtil::Equal(*layout_, *layout)) {
      EXPLAIN << "Layout " << LayoutUtil::HumanString(*layout)
              << " is not equal to expected "
              << LayoutUtil::HumanString(*layout_);
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "equal to " << LayoutUtil::HumanString(*layout_);
  }
 private:
  const ::xla::Layout* layout_;
};
class LayoutPatternMinorToMajorImpl {
 public:
  explicit LayoutPatternMinorToMajorImpl(
      absl::Span<const int64_t> minor_to_major)
      : minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (layout->minor_to_major() != minor_to_major_) {
      EXPLAIN << "Layout does not have minor to major ["
              << absl::StrJoin(minor_to_major_, ",") << "]";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with minor to major [" << absl::StrJoin(minor_to_major_, ",")
        << "]";
  }
 private:
  absl::InlinedVector<int64_t, 8> minor_to_major_;
};
template <typename LayoutType, typename Impl>
class LayoutPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
    auto new_allof = AllOf<::xla::Layout>(impl_, std::move(new_impl));
    return LayoutPattern<LayoutType, decltype(new_allof)>(std::move(new_allof),
                                                          matched_layout_);
  }
 public:
  explicit constexpr LayoutPattern(const Impl& impl,
                                   LayoutType** matched_layout)
      : impl_(impl), matched_layout_(matched_layout) {}
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }
  bool Match(::xla::Layout* layout, MatchOption option) const {
    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    impl_.DescribeTo(os, indent);
  }
  constexpr auto EqualTo(const ::xla::Layout* layout) const {
    return AppendImpl(LayoutPatternEqualImpl(layout));
  }
  constexpr auto WithMinorToMajor(
      absl::Span<const int64_t> minor_to_major) const {
    return AppendImpl(LayoutPatternMinorToMajorImpl(minor_to_major));
  }
 private:
  Impl impl_;
  LayoutType** matched_layout_;
};
template <typename Item, typename... Patterns>
class AnyOfPattern {
 public:
  explicit AnyOfPattern(const Patterns&... patterns) : patterns_(patterns...) {}
  bool Match(const Item* item, MatchOption option) const {
    return MatchImpl(item, option);
  }
  bool Match(Item* item, MatchOption option) const {
    return MatchImpl(item, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "any of:";
    Indent(os, indent);
    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }
 private:
  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option) const {
    std::optional<std::stringstream> explanation;
    MatchOption new_option = option;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }
    bool rv = MatchRecursiveImpl(item, new_option,
                                 std::integral_constant<size_t, 0>());
    if (!rv && option.explain_os) {
      EXPLAIN << "None of the following matchers succeeded:";
      EXPLAIN << explanation->str();
    }
    return rv;
  }
  template <typename ItemType, size_t index>
  bool MatchRecursiveImpl(ItemType* item, MatchOption option,
                          std::integral_constant<size_t, index>) const {
    auto new_option = option;
    new_option.capture = false;
    std::optional<std::stringstream> explanation;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }
    if (std::get<index>(patterns_).Match(item, new_option)) {
      if (option.capture) {
        bool matched = std::get<index>(patterns_).Match(item, option);
        DCHECK(matched);
      }
      return true;
    }
    if (option.explain_os) {
      EXPLAIN << "\nMatcher #" << index + 1;
      EXPLAIN << "\n - ";
      std::get<index>(patterns_).DescribeTo(option.explain_os, 3);
      EXPLAIN << "\nfailed with";
      EXPLAIN << "\n - ";
      EXPLAIN << absl::StrReplaceAll(explanation->str(), {{"\n", "\n   "}});
    }
    return MatchRecursiveImpl(item, option,
                              std::integral_constant<size_t, index + 1>());
  }
  template <typename ItemType>
  bool MatchRecursiveImpl(
      ItemType* item, MatchOption option,
      std::integral_constant<size_t, sizeof...(Patterns)>) const {
    return false;
  }
  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64_t indent) const {
    *os << " - ";
    std::get<index>(patterns_).DescribeTo(os, indent + 3);
    if (index != sizeof...(Patterns) - 1) {
      *os << " OR";
      Indent(os, indent);
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }
  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64_t indent) const {}
  std::tuple<Patterns...> patterns_;
};
}  
inline constexpr auto Layout(const ::xla::Layout** matched_layout = nullptr) {
  return detail::LayoutPattern<const ::xla::Layout,
                               detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}
inline constexpr auto Layout(::xla::Layout** matched_layout) {
  return detail::LayoutPattern<::xla::Layout, detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}
namespace detail {
template <typename ShapeType, typename Impl>
class ShapePattern;
class ShapePatternBaseImpl {
 public:
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape == nullptr) {
      EXPLAIN << "Shape is null";
    }
    return shape != nullptr;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "a shape";
  }
  static constexpr bool kIsTrivialMatcher = true;
};
class ShapePatternEqualImpl {
 public:
  explicit constexpr ShapePatternEqualImpl(const ::xla::Shape* shape)
      : shape_(shape) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::Equal(*shape_, *shape)) {
      EXPLAIN << "Shape not equal to "
              << ShapeUtil::HumanStringWithLayout(*shape_);
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "equal to " << ShapeUtil::HumanStringWithLayout(*shape_);
  }
 private:
  const ::xla::Shape* shape_;
};
class ShapePatternCompatibleImpl {
 public:
  explicit constexpr ShapePatternCompatibleImpl(const ::xla::Shape* shape)
      : shape_(shape) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::Compatible(*shape_, *shape)) {
      EXPLAIN << "Shape not compatible with "
              << ShapeUtil::HumanString(*shape_);
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "compatible with " << ShapeUtil::HumanString(*shape_);
  }
 private:
  const ::xla::Shape* shape_;
};
class ShapePatternElementTypeImpl {
 public:
  explicit constexpr ShapePatternElementTypeImpl(PrimitiveType element_type)
      : element_type_(element_type) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape->element_type() != element_type_) {
      EXPLAIN << "Shape does not have element type "
              << PrimitiveType_Name(element_type_);
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with element type " << PrimitiveType_Name(element_type_);
  }
 private:
  PrimitiveType element_type_;
};
class ShapePatternDimsImpl {
 public:
  explicit ShapePatternDimsImpl(absl::Span<const int64_t> dims)
      : dims_(dims.begin(), dims.end()) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape->dimensions() != dims_) {
      EXPLAIN << "Shape does not have dimensions [" << absl::StrJoin(dims_, ",")
              << "]";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with dimensions [" << absl::StrJoin(dims_, ",") << "]";
  }
 private:
  absl::InlinedVector<int64_t, 8> dims_;
};
class ShapePatternIsScalarImpl {
 public:
  explicit constexpr ShapePatternIsScalarImpl() = default;
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsScalar(*shape)) {
      EXPLAIN << "Shape is not a scalar";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "that represents a scalar";
  }
};
class ShapePatternIsArrayImpl {
 public:
  explicit constexpr ShapePatternIsArrayImpl() = default;
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!shape->IsArray()) {
      EXPLAIN << "Shape is not an array";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "that represents an array";
  }
};
class ShapePatternIsDenseArrayImpl {
 public:
  explicit constexpr ShapePatternIsDenseArrayImpl() = default;
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!LayoutUtil::IsDenseArray(*shape)) {
      EXPLAIN << "Shape is not a dense array";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "that represents a dense array";
  }
};
class ShapePatternIsTupleImpl {
 public:
  explicit constexpr ShapePatternIsTupleImpl() = default;
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!shape->IsTuple()) {
      EXPLAIN << "Shape is not a tuple";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "that represents a tuple";
  }
};
class ShapePatternEffectiveScalarImpl {
 public:
  explicit constexpr ShapePatternEffectiveScalarImpl() = default;
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (!ShapeUtil::IsEffectiveScalar(*shape)) {
      EXPLAIN << "Shape is not an effective scalar";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "that is an effective scalar";
  }
};
class ShapePatternRankImpl {
 public:
  explicit constexpr ShapePatternRankImpl(int64_t rank) : rank_(rank) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (shape->rank() != rank_) {
      if (rank_ == 0) {
        EXPLAIN << "Shape is not a scalar";
      } else {
        EXPLAIN << "Shape does not have rank " << rank_;
      }
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    if (rank_ == 0) {
      *os << "that is a scalar";
    } else {
      *os << "that has " << rank_ << " dimension" << (rank_ != 1 ? "s" : "");
    }
  }
 private:
  int64_t rank_;
};
template <typename LayoutType, typename LayoutImpl>
class ShapePatternLayoutImpl {
 public:
  explicit constexpr ShapePatternLayoutImpl(
      const LayoutPattern<LayoutType, LayoutImpl>& layout)
      : layout_(layout) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return LayoutUtil::HasLayout(*shape) &&
           layout_.Match(&shape->layout(), option);
  }
  bool Match(::xla::Shape* shape, MatchOption option) const {
    if (!LayoutUtil::HasLayout(*shape)) {
      EXPLAIN << "Shape does not have a layout";
      return false;
    }
    if (!layout_.Match(shape->mutable_layout(), option)) {
      EXPLAIN << "\nin layout";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with";
    Indent(os, indent + kIndentInc);
    layout_.DescribeTo(os, indent + kIndentInc);
  }
 private:
  LayoutPattern<LayoutType, LayoutImpl> layout_;
};
template <typename SubshapeType, typename SubshapeImpl>
class ShapePatternSubshapeImpl {
 public:
  explicit ShapePatternSubshapeImpl(
      ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape)
      : index_(index), subshape_(subshape) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    return MatchImpl(shape, option);
  }
  bool Match(::xla::Shape* shape, MatchOption option) const {
    return MatchImpl(shape, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with subshape at index " << ShapeIndex(index_) << " which is";
    Indent(os, indent + kIndentInc);
    subshape_.DescribeTo(os, indent + kIndentInc);
  }
 private:
  ::xla::Shape* GetSubshape(::xla::Shape* shape) const {
    return ShapeUtil::GetMutableSubshape(shape, index_);
  }
  const ::xla::Shape* GetSubshape(const ::xla::Shape* shape) const {
    return &ShapeUtil::GetSubshape(*shape, index_);
  }
  template <typename ShapeType>
  bool MatchImpl(ShapeType* shape, MatchOption option) const {
    if (!ShapeUtil::IndexIsValid(*shape, index_)) {
      EXPLAIN << "No subshape at " << ShapeIndex(index_);
      return false;
    }
    if (!subshape_.Match(GetSubshape(shape), option)) {
      EXPLAIN << "\nin subshape at " << ShapeIndex(index_);
      return false;
    }
    return true;
  }
  ShapeIndexView index_;
  ShapePattern<SubshapeType, SubshapeImpl> subshape_;
};
template <typename ShapeType, typename Impl>
class ShapePattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
    auto new_all_of = AllOf<::xla::Shape>(impl_, std::move(new_impl));
    return ShapePattern<ShapeType, decltype(new_all_of)>(std::move(new_all_of),
                                                         matched_shape_);
  }
 public:
  explicit constexpr ShapePattern(const Impl& impl, ShapeType** matched_shape)
      : impl_(impl), matched_shape_(matched_shape) {}
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    if (shape) {
      EXPLAIN << "\nin "
              << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                      : ShapeUtil::HumanString(*shape));
    }
    return false;
  }
  bool Match(::xla::Shape* shape, MatchOption option) const {
    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    EXPLAIN << "\nin "
            << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                    : ShapeUtil::HumanString(*shape));
    return false;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    return impl_.DescribeTo(os, indent);
  }
  constexpr auto EqualTo(const ::xla::Shape* shape) const {
    return AppendImpl(ShapePatternEqualImpl(shape));
  }
  constexpr auto CompatibleTo(const ::xla::Shape* shape) const {
    return AppendImpl(ShapePatternCompatibleImpl(shape));
  }
  constexpr auto WithElementType(PrimitiveType element_type) const {
    return AppendImpl(ShapePatternElementTypeImpl(element_type));
  }
  constexpr auto WithDims(absl::Span<const int64_t> dims) const {
    return AppendImpl(ShapePatternDimsImpl(dims));
  }
  constexpr auto IsScalar() const {
    return AppendImpl(ShapePatternIsScalarImpl());
  }
  constexpr auto IsArray() const {
    return AppendImpl(ShapePatternIsArrayImpl());
  }
  constexpr auto IsTuple() const {
    return AppendImpl(ShapePatternIsTupleImpl());
  }
  constexpr auto IsEffectiveScalar() const {
    return AppendImpl(ShapePatternEffectiveScalarImpl());
  }
  constexpr auto WithRank(int64_t rank) const {
    return AppendImpl(ShapePatternRankImpl(rank));
  }
  template <typename LayoutType, typename LayoutImpl>
  auto WithLayout(const LayoutPattern<LayoutType, LayoutImpl>& layout) const {
    return AppendImpl(ShapePatternLayoutImpl<LayoutType, LayoutImpl>(layout));
  }
  constexpr auto WithLayout(absl::Span<const int64_t> minor_to_major) const {
    return WithLayout(Layout().WithMinorToMajor(minor_to_major));
  }
  constexpr auto WithLayoutEqualTo(const ::xla::Layout* layout) const {
    return WithLayout(Layout().EqualTo(layout));
  }
  constexpr auto IsDenseArray() const {
    return AppendImpl(ShapePatternIsDenseArrayImpl());
  }
  template <typename SubshapeType, typename SubshapeImpl>
  auto WithSubshape(
      ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape) const {
    return AppendImpl(
        ShapePatternSubshapeImpl<SubshapeType, SubshapeImpl>(index, subshape));
  }
  ShapePattern<ShapeType,
               AllOfPattern<::xla::Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternEqualImpl>>>>
  WithSubshapeEqualTo(ShapeIndexView index, const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .EqualTo(shape));
  }
  ShapePattern<ShapeType,
               AllOfPattern<::xla::Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternCompatibleImpl>>>>
  WithSubshapeCompatibleTo(ShapeIndexView index,
                           const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .CompatibleTo(shape));
  }
 private:
  Impl impl_;
  ShapeType** matched_shape_;
};
}  
inline constexpr auto Shape(const ::xla::Shape** matched_shape = nullptr) {
  return detail::ShapePattern<const ::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}
inline constexpr auto Shape(::xla::Shape** matched_shape) {
  return detail::ShapePattern<::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}
namespace detail {
inline HloInstruction* HloOperand(HloInstruction* instr, int64_t idx) {
  return instr->mutable_operand(idx);
}
inline const HloInstruction* HloOperand(const HloInstruction* instr,
                                        int64_t idx) {
  return instr->operand(idx);
}
inline std::string InstToString(const HloInstruction* inst) {
  return inst->ToString(
      HloPrintOptions().set_print_metadata(false).set_print_percent(false));
}
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern;
class HloInstructionPatternBaseImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst == nullptr) {
      EXPLAIN << "HloInstruction* is null";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "an HloInstruction";
  }
  static constexpr bool kIsTrivialMatcher = true;
};
class HloInstructionPatternNameImpl {
 public:
  explicit HloInstructionPatternNameImpl(absl::string_view name)
      : name_(name) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->name() != name_) {
      EXPLAIN << "HloInstruction not named \"" << name_ << "\"";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "named \"" << name_ << "\"";
  }
 private:
  absl::string_view name_;
};
class HloInstructionIsImpl {
 public:
  explicit HloInstructionIsImpl(const HloInstruction* inst) : inst_(inst) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst != inst_) {
      EXPLAIN << "HloInstruction " << std::hex << std::nouppercase
              << std::showbase << reinterpret_cast<uint64_t>(inst) << " is not "
              << reinterpret_cast<uint64_t>(inst_) << " ("
              << InstToString(inst_) << ")";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which is " << std::hex << std::nouppercase << std::showbase
        << reinterpret_cast<uint64_t>(inst_) << " (" << InstToString(inst_)
        << ")";
  }
 private:
  const HloInstruction* inst_;
};
class HloInstructionPatternOpcodeImpl {
 public:
  explicit constexpr HloInstructionPatternOpcodeImpl(HloOpcode opcode,
                                                     bool invert)
      : opcode_(opcode), invert_(invert) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (invert_ && inst->opcode() == opcode_) {
      EXPLAIN << "HloInstruction has opcode " << opcode_
              << ", expected anything else";
      return false;
    }
    if (!invert_ && inst->opcode() != opcode_) {
      EXPLAIN << "HloInstruction doesn't have opcode " << opcode_;
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    if (!invert_) {
      *os << "with opcode " << opcode_;
    } else {
      *os << "with any opcode other than " << opcode_;
    }
  }
 private:
  HloOpcode opcode_;
  bool invert_;
};
class HloInstructionCustomCallTargetImpl {
 public:
  explicit HloInstructionCustomCallTargetImpl(
      absl::Span<const absl::string_view> custom_call_targets)
      : custom_call_targets_(custom_call_targets.begin(),
                             custom_call_targets.end()) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kCustomCall ||
        !absl::c_linear_search(custom_call_targets_,
                               inst->custom_call_target())) {
      if (custom_call_targets_.size() == 1) {
        EXPLAIN << "HloInstruction is not a custom call with a target '"
                << custom_call_targets_.front() << "'";
      } else {
        EXPLAIN << "HloInstruction is not a custom call with a target in {"
                << absl::StrJoin(custom_call_targets_, ", ") << "}";
      }
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    if (custom_call_targets_.size() == 1) {
      *os << "custom call with target '" << custom_call_targets_.front() << "'";
    } else {
      *os << "custom call with target in {"
          << absl::StrJoin(custom_call_targets_, ", ") << "}";
    }
  }
 private:
  absl::InlinedVector<std::string, 1> custom_call_targets_;
};
class HloInstructionPatternNumOperandsImpl {
 public:
  explicit constexpr HloInstructionPatternNumOperandsImpl(int64_t num_operands)
      : num_operands_(num_operands) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->operand_count() != num_operands_) {
      EXPLAIN << "HloInstruction doesn't have " << num_operands_ << " operands";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with " << num_operands_ << " operand"
        << (num_operands_ != 1 ? "s" : "");
  }
 private:
  int64_t num_operands_;
};
template <typename ShapeType, typename ShapeImpl>
class HloInstructionPatternShapeImpl {
 public:
  explicit constexpr HloInstructionPatternShapeImpl(
      const ShapePattern<ShapeType, ShapeImpl>& shape)
      : shape_(shape) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (!shape_.Match(&inst->shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    if (!shape_.Match(inst->mutable_shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "outputting";
    Indent(os, indent + kIndentInc);
    shape_.DescribeTo(os, indent + kIndentInc);
  }
 private:
  ShapePattern<ShapeType, ShapeImpl> shape_;
};
template <typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandImpl {
 public:
  explicit constexpr HloInstructionPatternOperandImpl(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : operand_index_(operand_index), operand_(operand) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with operand " << operand_index_ << " which is:";
    Indent(os, indent + kIndentInc);
    operand_.DescribeTo(os, indent + kIndentInc);
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (operand_index_ >= inst->operand_count()) {
      EXPLAIN << "desired operand index " << operand_index_
              << " is out of bounds";
      return false;
    }
    if (!operand_.Match(HloOperand(inst, operand_index_), option)) {
      EXPLAIN << "\nin operand " << operand_index_;
      return false;
    }
    if (option.single_user_only &&
        inst->operand(operand_index_)->user_count() != 1) {
      EXPLAIN << "Operand " << operand_index_ << " of HloInstruction has "
              << inst->operand(operand_index_)->user_count()
              << " users. Expected 1.";
      return false;
    }
    return true;
  }
  int64_t operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};
template <typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandIfPresentImpl {
 public:
  explicit constexpr HloInstructionPatternOperandIfPresentImpl(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : operand_index_(operand_index), operand_(operand) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "either with fewer than " << operand_index_ + 1 << " operand"
        << (operand_index_ + 1 != 1 ? "s" : "") << ", or with an operand "
        << operand_index_ << " which is:";
    Indent(os, indent + kIndentInc);
    operand_.DescribeTo(os, indent + kIndentInc);
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (operand_index_ >= inst->operand_count()) {
      return true;
    }
    if (!operand_.Match(HloOperand(inst, operand_index_), option)) {
      EXPLAIN << "\nin operand " << operand_index_;
      return false;
    }
    return true;
  }
  int64_t operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};
template <typename OperandType1, typename OperandImpl1, typename OperandType2,
          typename OperandImpl2>
class HloInstructionPatternBinaryOperandsAnyOrderImpl {
 public:
  explicit constexpr HloInstructionPatternBinaryOperandsAnyOrderImpl(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2)
      : op1_(op1), op2_(op2) {}
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with two operands in either order:";
    Indent(os, indent);
    *os << " - ";
    op1_.DescribeTo(os, indent + 3);
    Indent(os, indent);
    *os << " - ";
    op2_.DescribeTo(os, indent + 3);
  }
 private:
  HloInstruction* operand(HloInstruction* inst, int64_t idx) const {
    return inst->mutable_operand(idx);
  }
  const HloInstruction* operand(const HloInstruction* inst, int64_t idx) const {
    return inst->operand(idx);
  }
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->operand_count() != 2) {
      EXPLAIN << "HloInstruction did not have two operands";
      return false;
    }
    if (option.single_user_only) {
      for (int i = 0; i < 2; ++i) {
        if (inst->operand(i)->user_count() != 1) {
          EXPLAIN << "Operand " << i << " of HloInstruction has "
                  << inst->operand(i)->user_count() << " users. Expected 1.";
          return false;
        }
      }
    }
    if (!option.explain_os) {
      auto try_match = [&](int64_t idx1, int64_t idx2) {
        MatchOption new_option = option;
        new_option.capture = false;
        if (op1_.Match(operand(inst, idx1), new_option) &&
            op2_.Match(operand(inst, idx2), new_option)) {
          if (option.capture) {
            bool matched = op1_.Match(operand(inst, idx1), option) &&
                           op2_.Match(operand(inst, idx2), option);
            DCHECK(matched);
          }
          return true;
        }
        return false;
      };
      return try_match(0, 1) || try_match(1, 0);
    }
    bool matches[ 2][ 2];
    std::stringstream explanations[ 2][ 2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        MatchOption new_option = option;
        new_option.capture = false;
        new_option.explain_os = &explanations[i][j];
        matches[i][j] = i == 0 ? op1_.Match(operand(inst, j), new_option)
                               : op2_.Match(operand(inst, j), new_option);
      }
    }
    for (int i = 0; i < 2; ++i) {
      if (matches[0][i] && matches[1][(i + 1) % 2]) {
        if (option.capture) {
          auto* operand1 = operand(inst, i);
          auto* operand2 = operand(inst, (i + 1) % 2);
          bool matched =
              op1_.Match(operand1, option) && op2_.Match(operand2, option);
          DCHECK(matched);
        }
        return true;
      }
    }
    auto describe_matcher = [&](int matcher_idx) {
      EXPLAIN << "\n - ";
      if (matcher_idx == 0) {
        op1_.DescribeTo(option.explain_os, 3);
      } else {
        CHECK_EQ(matcher_idx, 1);
        op2_.DescribeTo(option.explain_os, 3);
      }
      for (int i = 0; i < 2; ++i) {
        if (matches[matcher_idx][ i]) {
          continue;
        }
        EXPLAIN << "\ndoes not match " << (i == 0 ? "LHS" : "RHS") << ":\n";
        EXPLAIN << " - ";
        EXPLAIN << absl::StrReplaceAll(
            explanations[matcher_idx][ i].str(), {{"\n", "\n   "}});
      }
    };
    bool wrote_explanation = false;
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (!matches[i][0] && !matches[i][1]) {
        EXPLAIN << "HloInstruction's operands (ignoring order) did not match "
                << (i == 0 ? "first" : "second") << " matcher. Specifically,";
        describe_matcher(i);
        wrote_explanation = true;
      }
    }
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (matches[ 0][ i] &&
          matches[ 1][ i]) {
        CHECK(!matches[0][(i + 1) % 2]);
        CHECK(!matches[1][(i + 1) % 2]);
        CHECK(!wrote_explanation);
        EXPLAIN << "HloInstruction's " << (i == 1 ? "LHS" : "RHS")
                << " operand did not match either of the two matchers. "
                   "Specifically,";
        describe_matcher(0);
        EXPLAIN << "\nand";
        describe_matcher(1);
        wrote_explanation = true;
      }
    }
    CHECK(wrote_explanation);
    return false;
  }
  HloInstructionPattern<OperandType1, OperandImpl1> op1_;
  HloInstructionPattern<OperandType2, OperandImpl2> op2_;
};
class HloInstructionPatternFusionKindImpl {
 public:
  explicit constexpr HloInstructionPatternFusionKindImpl(
      ::xla::HloInstruction::FusionKind kind)
      : kind_(kind) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with fusion kind " << ToString(kind_);
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kFusion) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_)
              << "; it's not a fusion";
      return false;
    }
    if (inst->fusion_kind() != kind_) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_);
      return false;
    }
    return true;
  }
  ::xla::HloInstruction::FusionKind kind_;
};
class HloInstructionPatternTupleIndexImpl {
 public:
  explicit constexpr HloInstructionPatternTupleIndexImpl(int64_t tuple_index)
      : tuple_index_(tuple_index) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which is a GTE with index " << tuple_index_;
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kGetTupleElement) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_
              << "; it's not a GTE at all";
      return false;
    }
    if (inst->tuple_index() != tuple_index_) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_;
      return false;
    }
    return true;
  }
  int64_t tuple_index_;
};
class HloInstructionPatternParameterNumImpl {
 public:
  explicit constexpr HloInstructionPatternParameterNumImpl(
      int64_t parameter_num)
      : parameter_num_(parameter_num) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which is parameter " << parameter_num_;
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kParameter ||
        inst->parameter_number() != parameter_num_) {
      EXPLAIN << "HloInstruction is not parameter " << parameter_num_;
      return false;
    }
    return true;
  }
  int64_t parameter_num_;
};
class HloInstructionPatternOneUseOrUserImpl {
 protected:
  bool MatchOneUser(const HloInstruction* inst, MatchOption option) const {
    if (inst->user_count() != 1) {
      EXPLAIN << "HloInstruction has " << inst->user_count()
              << " users, but expected exactly one.";
      if (inst->user_count() > 1) {
        EXPLAIN << "\nAll users:";
        for (const HloInstruction* user : inst->users()) {
          EXPLAIN << "\n - " << InstToString(user);
        }
      }
      return false;
    }
    return true;
  }
};
class HloInstructionPatternOneUseImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (!MatchOneUser(inst, option)) {
      return false;
    }
    int64_t use_count = absl::c_count_if(
        inst->users()[0]->operands(),
        [&](const HloInstruction* operand) { return operand == inst; });
    if (use_count != 1) {
      EXPLAIN << "HloInstruction is used " << use_count
              << " times by its user, but is expected to be used just once: "
              << InstToString(inst->users()[0]);
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has exactly one use";
  }
};
class HloInstructionPatternOneUserImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchOneUser(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has exactly one user (but possibly is used multiple times by "
           "that instruction)";
  }
};
class HloInstructionPatternNumUserImpl {
 public:
  explicit constexpr HloInstructionPatternNumUserImpl(int64_t user_num)
      : user_num_(user_num) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->user_count() != user_num_) {
      EXPLAIN << "HloInstruction has " << inst->user_count()
              << " users, but expected exactly " << user_num_ << " users.";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has exactly " << user_num_
        << " users (but possibly is used multiple times by "
           "same instruction)";
  }
 private:
  int64_t user_num_;
};
class HloInstructionPatternAtMostNumUserImpl {
 public:
  explicit constexpr HloInstructionPatternAtMostNumUserImpl(int64_t user_num)
      : user_num_(user_num) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (inst->user_count() > user_num_) {
      EXPLAIN << "HloInstruction has " << inst->user_count()
              << " users, but expected less than or equal " << user_num_
              << " users.";
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has less than or equal " << user_num_
        << " users (but possibly is used multiple times by "
           "same instruction)";
  }
 private:
  int64_t user_num_;
};
class HloInstructionPatternComparisonDirectionImpl {
 public:
  explicit constexpr HloInstructionPatternComparisonDirectionImpl(
      ComparisonDirection direction)
      : direction_(direction) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has comparison direction "
        << ComparisonDirectionToString(direction_);
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kCompare ||
        inst->comparison_direction() != direction_) {
      EXPLAIN << "HloInstruction is not comparison "
              << ComparisonDirectionToString(direction_);
      return false;
    }
    return true;
  }
  ComparisonDirection direction_;
};
class HloInstructionPatternConvDnumsImpl {
 public:
  explicit HloInstructionPatternConvDnumsImpl(absl::string_view dnums)
      : HloInstructionPatternConvDnumsImpl(
            ParseConvolutionDimensionNumbers(dnums).value()) {}
  explicit HloInstructionPatternConvDnumsImpl(ConvolutionDimensionNumbers dnums)
      : dnums_(std::move(dnums)) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which has convolution dimension numbers "
        << ConvolutionDimensionNumbersToString(dnums_);
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kConvolution &&
        inst->opcode() != HloOpcode::kCustomCall) {
      EXPLAIN << "HloInstruction is not convolution or custom-call and so "
                 "can't have convolution_dimension_numbers";
      return false;
    }
    const ConvolutionDimensionNumbers& actual_dnums =
        inst->convolution_dimension_numbers();
    if (!tsl::protobuf::util::MessageDifferencer::Equals(dnums_,
                                                         actual_dnums)) {
      EXPLAIN << "convolution_dimension_numbers "
              << ConvolutionDimensionNumbersToString(actual_dnums)
              << " don't match expected "
              << ConvolutionDimensionNumbersToString(dnums_);
      return false;
    }
    return true;
  }
  ConvolutionDimensionNumbers dnums_;
};
class HloInstructionPredicateImpl {
 public:
  explicit HloInstructionPredicateImpl(HloPredicate fn) : fn_(std::move(fn)) {}
  bool Match(const HloInstruction* inst, MatchOption option) const {
    bool match = fn_(inst);
    if (!match) {
      EXPLAIN << "HloInstruction does not match user-specified predicate";
    }
    return match;
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which matches a user-specified predicate";
  }
 private:
  HloPredicate fn_;
};
class HloInstructionContractingDimsImpl {
 public:
  explicit HloInstructionContractingDimsImpl(
      absl::Span<const int64_t> lhs_contracting_dims,
      absl::Span<const int64_t> rhs_contracting_dims)
      : lhs_contracting_dims_(lhs_contracting_dims.begin(),
                              lhs_contracting_dims.end()),
        rhs_contracting_dims_(rhs_contracting_dims.begin(),
                              rhs_contracting_dims.end()) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "with lhs_contracting_dims {"
        << absl::StrJoin(lhs_contracting_dims_, ",")
        << "} and rhs_contracting_dims {"
        << absl::StrJoin(rhs_contracting_dims_, ",") << "}";
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (inst->opcode() != HloOpcode::kDot) {
      EXPLAIN << "HloInstruction is not dot so "
                 "can't have dot_dimension_numbers";
      return false;
    }
    const DotDimensionNumbers& dnums = inst->dot_dimension_numbers();
    if (absl::MakeSpan(dnums.lhs_contracting_dimensions()) !=
        lhs_contracting_dims_) {
      EXPLAIN << "lhs_contracting_dimensions {"
              << absl::StrJoin(dnums.lhs_contracting_dimensions(), ",")
              << "} don't match expected {"
              << absl::StrJoin(lhs_contracting_dims_, ",") << "}";
      return false;
    }
    if (absl::MakeSpan(dnums.rhs_contracting_dimensions()) !=
        rhs_contracting_dims_) {
      EXPLAIN << "rhs_contracting_dimensions {"
              << absl::StrJoin(dnums.rhs_contracting_dimensions(), ",")
              << "} don't match expected {"
              << absl::StrJoin(rhs_contracting_dims_, ",") << "}";
      return false;
    }
    return true;
  }
  absl::InlinedVector<int64_t, 8> lhs_contracting_dims_;
  absl::InlinedVector<int64_t, 8> rhs_contracting_dims_;
};
class HloInstructionReplicaGroupsImpl {
 public:
  explicit HloInstructionReplicaGroupsImpl(
      std::vector<std::vector<int64_t>> replica_groups)
      : replica_groups_(std::move(replica_groups)) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    std::vector<std::string> replica_group_strs;
    replica_group_strs.reserve(replica_groups_.size());
    for (const std::vector<int64_t>& replica_group : replica_groups_) {
      replica_group_strs.push_back(
          absl::StrCat("{", absl::StrJoin(replica_group, ","), "}"));
    }
    *os << "with replica_group {" << absl::StrJoin(replica_group_strs, ",")
        << "}";
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    const HloCollectiveInstruction* collective =
        DynCast<HloCollectiveInstruction>(inst);
    if (!collective) {
      EXPLAIN << "HloInstruction is not a collective";
      return false;
    }
    if (absl::c_equal(collective->replica_groups(), replica_groups_,
                      [](const ReplicaGroup& a, const std::vector<int64_t>& b) {
                        return absl::c_equal(a.replica_ids(), b);
                      })) {
      return true;
    }
    std::ostringstream desc_stream;
    DescribeTo(&desc_stream);
    std::vector<std::string> replica_group_strs;
    replica_group_strs.reserve(replica_groups_.size());
    for (const ReplicaGroup& replica_group : collective->replica_groups()) {
      replica_group_strs.push_back(absl::StrCat(
          "{", absl::StrJoin(replica_group.replica_ids(), ","), "}"));
    }
    EXPLAIN << "replica_group {" << absl::StrJoin(replica_group_strs, ",")
            << "} don't match expected " << desc_stream.str();
    return false;
  }
  std::vector<std::vector<int64_t>> replica_groups_;
};
class HloInstructionShardingImpl {
 public:
  explicit HloInstructionShardingImpl(
      const std::optional<HloSharding>& sharding)
      : sharding_(sharding) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    if (sharding_.has_value()) {
      *os << "with sharding " << sharding_->ToString();
    } else {
      *os << "with no sharding";
    }
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    if (!sharding_.has_value()) {
      if (!inst->has_sharding()) {
        return true;
      }
      EXPLAIN << "HloInstruction is expected to have no sharding.";
      return false;
    }
    if (inst->has_sharding()) {
      if (inst->sharding() == sharding_.value()) {
        return true;
      }
      EXPLAIN << "sharding " << inst->sharding().ToString()
              << " don't match expected " << sharding_->ToString();
      return false;
    } else {
      EXPLAIN << "HloInstruction has no sharding. Expected: "
              << sharding_->ToString();
      return false;
    }
  }
  std::optional<HloSharding> sharding_;
};
class HloInstructionControlDepsImpl {
 public:
  explicit HloInstructionControlDepsImpl(
      absl::Span<HloInstruction* const> preds,
      absl::Span<HloInstruction* const> succs)
      : preds_(preds), succs_(succs) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    auto print_deps = [os](absl::Span<HloInstruction* const> deps,
                           absl::string_view type) {
      if (deps.empty()) {
        *os << "no control " << type;
      } else {
        *os << "control " << type << " {" << absl::StrJoin(deps, ",", fmt)
            << "}";
      }
    };
    *os << "with ";
    print_deps(preds_, "predecessors");
    *os << " and ";
    print_deps(succs_, "successors");
  }
 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
    auto match_deps = [&](absl::Span<HloInstruction* const> expected_deps,
                          const PtrVec<HloInstruction*>& actual_deps,
                          absl::string_view type) {
      if (!absl::c_equal(expected_deps, actual_deps)) {
        EXPLAIN << "HloInstruction expected to have control " << type << " {"
                << absl::StrJoin(expected_deps, ",", fmt) << "} but has {"
                << absl::StrJoin(actual_deps, ",", fmt) << "}";
        return false;
      }
      return true;
    };
    return match_deps(preds_, inst->control_predecessors(), "predecessors") &&
           match_deps(succs_, inst->control_successors(), "successors");
  }
  static void fmt(std::string* out, const HloInstruction* inst) {
    absl::StrAppend(out, inst->name());
  };
  absl::Span<HloInstruction* const> preds_, succs_;
};
template <typename ScalarTy>
class HloConstantScalarImpl {
 public:
  explicit constexpr HloConstantScalarImpl(bool match_effective_scalar)
      : val_(std::nullopt), match_effective_scalar_(match_effective_scalar) {}
  constexpr HloConstantScalarImpl(ScalarTy val, bool match_effective_scalar)
      : val_(val), match_effective_scalar_(match_effective_scalar) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
    return MatchImpl(inst, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    *os << "which is a constant "
        << (match_effective_scalar_ ? "effective " : "") << "scalar";
    if (val_.has_value()) {
      *os << " with value " << *val_;
    }
  }
 private:
  template <typename InstTy>
  bool MatchImpl(InstTy* inst, MatchOption option) const {
    const auto* const_inst = DynCast<HloConstantInstruction>(inst);
    if (!const_inst) {
      EXPLAIN << "HloInstruction is not a constant";
      return false;
    }
    if (match_effective_scalar_ &&
        !ShapeUtil::IsEffectiveScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not an effective scalar";
      return false;
    }
    if (!match_effective_scalar_ && !ShapeUtil::IsScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not a scalar";
      return false;
    }
    if (!val_.has_value()) {
      return true;
    }
    auto const_inst_scalar_or = const_inst->literal().Reshape({});
    if (!const_inst_scalar_or.ok()) {
      EXPLAIN << "could not convert matched literal to effective scalar";
      return false;
    }
    Literal const_inst_scalar = std::move(const_inst_scalar_or).value();
    if (!const_inst_scalar.IsEqualAt({}, *val_)) {
      EXPLAIN << "HloInstruction's constant value "
              << const_inst_scalar.ToStringWithoutShape()
              << " did not match expected value " << *val_;
      return false;
    }
    return true;
  }
  std::optional<ScalarTy> val_;
  bool match_effective_scalar_;
};
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
    auto new_allof = AllOf<::xla::HloInstruction>(impl_, std::move(new_impl));
    return HloInstructionPattern<HloInstructionType, decltype(new_allof)>(
        std::move(new_allof), matched_inst_);
  }
 public:
  explicit constexpr HloInstructionPattern(const Impl& impl,
                                           HloInstructionType** matched_inst)
      : impl_(impl), matched_inst_(matched_inst) {}
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    if (inst != nullptr) {
      EXPLAIN << "\nin " << InstToString(inst);
    }
    return false;
  }
  bool Match(::xla::HloInstruction* inst, MatchOption option,
             bool explain_instruction = true) const {
    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    if (explain_instruction) {
      EXPLAIN << "\nin " << InstToString(inst);
    }
    return false;
  }
  auto WithName(absl::string_view name) const {
    return AppendImpl(HloInstructionPatternNameImpl(name));
  }
  auto WithOpcode(HloOpcode opcode) const {
    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, false));
  }
  auto WithCustomCallTarget(
      absl::Span<const absl::string_view> custom_call_targets) const {
    return AppendImpl(HloInstructionCustomCallTargetImpl(custom_call_targets));
  }
  auto WithNumOperands(int64_t num_operands) const {
    return AppendImpl(HloInstructionPatternNumOperandsImpl(num_operands));
  }
  auto WithoutOpcode(HloOpcode opcode) const {
    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, true));
  }
  constexpr auto Is(const HloInstruction* instr) const {
    return AppendImpl(HloInstructionIsImpl(instr));
  }
  constexpr auto IsConstant() const { return WithOpcode(HloOpcode::kConstant); }
  constexpr auto IsConstantScalar() const {
    return AppendImpl(
        HloConstantScalarImpl< int>(false));
  }
  template <typename ScalarTy>
  constexpr auto IsConstantScalar(const ScalarTy& val) const {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, false));
  }
  constexpr auto IsConstantEffectiveScalar() const {
    return AppendImpl(
        HloConstantScalarImpl< int>(true));
  }
  template <typename ScalarTy>
  constexpr auto IsConstantEffectiveScalar(const ScalarTy& val) const {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, true));
  }
  constexpr auto IsNonConstant() const {
    return WithoutOpcode(HloOpcode::kConstant);
  }
  template <typename ShapeType, typename ShapeImpl>
  constexpr auto WithShape(
      const ShapePattern<ShapeType, ShapeImpl>& shape) const {
    return AppendImpl(
        HloInstructionPatternShapeImpl<ShapeType, ShapeImpl>(shape));
  }
  constexpr auto WithShape(PrimitiveType ty, absl::Span<const int64_t> dims) {
    return WithShape(Shape().WithElementType(ty).WithDims(dims));
  }
  constexpr auto WithShape(PrimitiveType ty, absl::Span<const int64_t> dims,
                           absl::Span<const int64_t> minor_to_major) {
    return WithShape(
        Shape().WithElementType(ty).WithDims(dims).WithLayout(minor_to_major));
  }
  template <typename Dummy = void>
  constexpr auto WithShapeEqualTo(const ::xla::Shape* shape) const {
    return WithShape(Shape().EqualTo(shape));
  }
  template <typename Dummy = void>
  constexpr auto WithShapeCompatibleTo(const ::xla::Shape* shape) const {
    return WithShape(Shape().CompatibleTo(shape));
  }
  constexpr auto WithElementType(PrimitiveType ty) {
    return WithShape(Shape().WithElementType(ty));
  }
  template <typename OperandType, typename OperandImpl>
  constexpr auto WithOperand(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const {
    return AppendImpl(
        HloInstructionPatternOperandImpl<OperandType, OperandImpl>(
            operand_index, operand));
  }
  template <typename OperandType, typename OperandImpl>
  constexpr auto WithOperandIfPresent(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const {
    return AppendImpl(
        HloInstructionPatternOperandIfPresentImpl<OperandType, OperandImpl>(
            operand_index, operand));
  }
  template <typename OperandType1, typename OperandImpl1, typename OperandType2,
            typename OperandImpl2>
  constexpr auto WithBinaryOperandsAnyOrder(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2) const {
    return AppendImpl(
        HloInstructionPatternBinaryOperandsAnyOrderImpl<
            OperandType1, OperandImpl1, OperandType2, OperandImpl2>(op1, op2));
  }
  constexpr auto WithFusionKind(HloInstruction::FusionKind kind) const {
    return AppendImpl(HloInstructionPatternFusionKindImpl(kind));
  }
  constexpr auto WithTupleIndex(int64_t tuple_index) const {
    return AppendImpl(HloInstructionPatternTupleIndexImpl(tuple_index));
  }
  constexpr auto WithParameterNum(int64_t parameter_num) const {
    return AppendImpl(HloInstructionPatternParameterNumImpl(parameter_num));
  }
  constexpr auto WithOneUse() const {
    return AppendImpl(HloInstructionPatternOneUseImpl());
  }
  constexpr auto WithOneUser() const {
    return AppendImpl(HloInstructionPatternOneUserImpl());
  }
  constexpr auto WithNumUser(int64_t user_num) const {
    return AppendImpl(HloInstructionPatternNumUserImpl(user_num));
  }
  constexpr auto WithAtMostNumUser(int64_t user_num) const {
    return AppendImpl(HloInstructionPatternAtMostNumUserImpl(user_num));
  }
  auto WithComparisonDirection(ComparisonDirection direction) const {
    return AppendImpl(HloInstructionPatternComparisonDirectionImpl(direction));
  }
  auto WithConvDnums(absl::string_view dnums) const {
    return AppendImpl(HloInstructionPatternConvDnumsImpl(dnums));
  }
  auto WithConvDnums(ConvolutionDimensionNumbers dnums) const {
    return AppendImpl(HloInstructionPatternConvDnumsImpl(dnums));
  }
  auto WithPredicate(HloPredicate fn) const {
    return AppendImpl(HloInstructionPredicateImpl(std::move(fn)));
  }
  auto WithContractingDims(
      absl::Span<const int64_t> lhs_contracting_dims,
      absl::Span<const int64_t> rhs_contracting_dims) const {
    return AppendImpl(HloInstructionContractingDimsImpl(lhs_contracting_dims,
                                                        rhs_contracting_dims));
  }
  auto WithReplicaGroups(
      std::vector<std::vector<int64_t>> replica_groups) const {
    return AppendImpl(
        HloInstructionReplicaGroupsImpl(std::move(replica_groups)));
  }
  auto WithSharding(absl::string_view sharding) const {
    return AppendImpl(
        HloInstructionShardingImpl(ParseSharding(sharding).value()));
  }
  auto WithControlDeps(absl::Span<HloInstruction* const> preds,
                       absl::Span<HloInstruction* const> succs) {
    return AppendImpl(HloInstructionControlDepsImpl(preds, succs));
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    impl_.DescribeTo(os, indent);
  }
 private:
  Impl impl_;
  HloInstructionType** matched_inst_;
};
template <typename Item, typename... Patterns>
struct AnyOfImpl {
  auto operator()(const Patterns&... patterns) const {
    return AnyOfPattern<typename std::remove_const<Item>::type, Patterns...>(
        patterns...);
  }
};
template <typename... Patterns>
struct AnyOfImpl<HloInstruction, Patterns...> {
  auto operator()(const Patterns&... patterns) const {
    auto any_of = AnyOfPattern<HloInstruction, Patterns...>(patterns...);
    return HloInstructionPattern<HloInstruction, decltype(any_of)>(
        std::move(any_of), nullptr);
  }
};
}  
template <typename Item, typename... Patterns>
auto AnyOf(const Patterns&... patterns) {
  return detail::AnyOfImpl<Item, Patterns...>()(patterns...);
}
inline constexpr auto Op(const ::xla::HloInstruction** matched_inst = nullptr) {
  return detail::HloInstructionPattern<const ::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}
inline constexpr auto Op(::xla::HloInstruction** matched_inst) {
  return detail::HloInstructionPattern<::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}
#define XLA_NULLOP_PATTERN(NAME)                                     \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); } \
                                                                     \
  template <typename HloInstructionType>                             \
  inline auto NAME(HloInstructionType** matched_inst) {              \
    return Op(matched_inst).WithOpcode(HloOpcode::k##NAME);          \
  }
XLA_NULLOP_PATTERN(Constant)
XLA_NULLOP_PATTERN(Parameter)
XLA_NULLOP_PATTERN(Iota)
XLA_NULLOP_PATTERN(Rng)
XLA_NULLOP_PATTERN(PartitionId)
XLA_NULLOP_PATTERN(ReplicaId)
#undef XLA_NULLOP_PATTERN
#define XLA_UNOP_PATTERN(NAME)                                       \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); } \
                                                                     \
  template <typename HloInstructionType>                             \
  inline auto NAME(HloInstructionType** matched_inst) {              \
    return Op(matched_inst).WithOpcode(HloOpcode::k##NAME);          \
  }                                                                  \
                                                                     \
  template <typename Arg>                                            \
  inline auto NAME(Arg&& arg) {                                      \
    return Op()                                                      \
        .WithOpcode(HloOpcode::k##NAME)                              \
        .WithOperand(0, std::forward<Arg>(arg));                     \
  }                                                                  \
                                                                     \
  template <typename HloInstructionType, typename Arg>               \
  inline auto NAME(HloInstructionType** matched_inst, Arg&& arg) {   \
    return Op(matched_inst)                                          \
        .WithOpcode(HloOpcode::k##NAME)                              \
        .WithOperand(0, std::forward<Arg>(arg));                     \
  }
XLA_UNOP_PATTERN(Abs)
XLA_UNOP_PATTERN(RoundNearestAfz)
XLA_UNOP_PATTERN(Bitcast)
XLA_UNOP_PATTERN(BitcastConvert)
XLA_UNOP_PATTERN(Broadcast)
XLA_UNOP_PATTERN(Cbrt)
XLA_UNOP_PATTERN(Ceil)
XLA_UNOP_PATTERN(Convert)
XLA_UNOP_PATTERN(Copy)
XLA_UNOP_PATTERN(Cos)
XLA_UNOP_PATTERN(AllReduceStart)
XLA_UNOP_PATTERN(AllReduceDone)
XLA_UNOP_PATTERN(AllToAll)
XLA_UNOP_PATTERN(AsyncDone)
XLA_UNOP_PATTERN(CollectiveBroadcast)
XLA_UNOP_PATTERN(CollectivePermute)
XLA_UNOP_PATTERN(CollectivePermuteStart)
XLA_UNOP_PATTERN(CollectivePermuteDone)
XLA_UNOP_PATTERN(Domain)
XLA_UNOP_PATTERN(Erf)
XLA_UNOP_PATTERN(Exp)
XLA_UNOP_PATTERN(Expm1)
XLA_UNOP_PATTERN(Fft)
XLA_UNOP_PATTERN(Floor)
XLA_UNOP_PATTERN(GetTupleElement)
XLA_UNOP_PATTERN(Imag)
XLA_UNOP_PATTERN(Infeed)
XLA_UNOP_PATTERN(IsFinite)
XLA_UNOP_PATTERN(Log)
XLA_UNOP_PATTERN(Logistic)
XLA_UNOP_PATTERN(Not)
XLA_UNOP_PATTERN(Negate)
XLA_UNOP_PATTERN(OptimizationBarrier)
XLA_UNOP_PATTERN(Real)
XLA_UNOP_PATTERN(Recv)
XLA_UNOP_PATTERN(RecvDone)
XLA_UNOP_PATTERN(ReducePrecision)
XLA_UNOP_PATTERN(Reshape)
XLA_UNOP_PATTERN(Reverse)
XLA_UNOP_PATTERN(Rsqrt)
XLA_UNOP_PATTERN(SendDone)
XLA_UNOP_PATTERN(Sign)
XLA_UNOP_PATTERN(Sin)
XLA_UNOP_PATTERN(Slice)
XLA_UNOP_PATTERN(Sqrt)
XLA_UNOP_PATTERN(Tan)
XLA_UNOP_PATTERN(Tanh)
XLA_UNOP_PATTERN(Transpose)
XLA_UNOP_PATTERN(While)
#undef XLA_UNOP_PATTERN
#define XLA_BINOP_PATTERN(NAME)                                               \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }          \
                                                                              \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return Op()                                                               \
        .WithOpcode(HloOpcode::k##NAME)                                       \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs));                              \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return Op(matched_inst)                                                   \
        .WithOpcode(HloOpcode::k##NAME)                                       \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs));                              \
  }
#define XLA_COMMUTATIVE_BINOP_PATTERN(NAME)                                \
  XLA_BINOP_PATTERN(NAME)                                                  \
                                                                           \
  template <typename HloInstructionType, typename Lhs, typename Rhs>       \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs, \
                             Rhs&& rhs) {                                  \
    return Op(matched_inst)                                                \
        .WithOpcode(HloOpcode::k##NAME)                                    \
        .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),                \
                                    std::forward<Rhs>(rhs));               \
  }                                                                        \
  template <typename Lhs, typename Rhs>                                    \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs) {                       \
    return NAME##AnyOrder<const HloInstruction>(                           \
        nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));          \
  }
XLA_COMMUTATIVE_BINOP_PATTERN(Add)
XLA_BINOP_PATTERN(Atan2)
XLA_BINOP_PATTERN(Divide)
XLA_BINOP_PATTERN(Complex)
XLA_BINOP_PATTERN(Compare)
XLA_BINOP_PATTERN(Convolution)
XLA_BINOP_PATTERN(Dot)
XLA_BINOP_PATTERN(Gather)
XLA_COMMUTATIVE_BINOP_PATTERN(Maximum)
XLA_COMMUTATIVE_BINOP_PATTERN(Minimum)
XLA_COMMUTATIVE_BINOP_PATTERN(Multiply)
XLA_BINOP_PATTERN(Outfeed)
XLA_BINOP_PATTERN(Pad)
XLA_BINOP_PATTERN(Power)
XLA_BINOP_PATTERN(Remainder)
XLA_BINOP_PATTERN(Send)
XLA_BINOP_PATTERN(Subtract)
XLA_COMMUTATIVE_BINOP_PATTERN(And)
XLA_COMMUTATIVE_BINOP_PATTERN(Or)
XLA_BINOP_PATTERN(ShiftLeft)
XLA_BINOP_PATTERN(ShiftRightArithmetic)
XLA_BINOP_PATTERN(ShiftRightLogical)
XLA_COMMUTATIVE_BINOP_PATTERN(Xor)
#undef XLA_COMMUTATIVE_BINOP_PATTERN
#undef XLA_BINOP_PATTERN
#define XLA_TERNOP_PATTERN(NAME)                                       \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }   \
                                                                       \
  template <typename Arg0, typename Arg1, typename Arg2>               \
  inline auto NAME(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {            \
    return Op()                                                        \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }                                                                    \
                                                                       \
  template <typename HloInstructionType, typename Arg0, typename Arg1, \
            typename Arg2>                                             \
  inline auto NAME(HloInstructionType** matched_inst, Arg0&& arg0,     \
                   Arg1&& arg1, Arg2&& arg2) {                         \
    return Op(matched_inst)                                            \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }
XLA_TERNOP_PATTERN(Clamp);
XLA_TERNOP_PATTERN(Select);
XLA_TERNOP_PATTERN(SelectAndScatter);
#undef XLA_TERNOP_PATTERN
namespace detail {
template <typename Matcher, typename FirstArg>
inline auto WithOperands(Matcher&& m, int64_t operand_num,
                         FirstArg&& first_arg) {
  return m.WithOperand(operand_num, std::forward<FirstArg>(first_arg));
}
template <typename Matcher, typename FirstArg, typename... Args>
inline auto WithOperands(Matcher&& m, int64_t operand_num, FirstArg&& first_arg,
                         Args&&... args) {
  return WithOperands(
      m.WithOperand(operand_num, std::forward<FirstArg>(first_arg)),
      operand_num + 1, std::forward<Args>(args)...);
}
}  
#define XLA_VARIADIC_OP_PATTERN(NAME)                                         \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }          \
                                                                              \
  template <typename... Args>                                                 \
  inline auto NAME(Args&&... args) {                                          \
    return detail::WithOperands(                                              \
        Op().WithOpcode(HloOpcode::k##NAME).WithNumOperands(sizeof...(Args)), \
        0, std::forward<Args>(args)...);                      \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename... Args>                    \
  inline auto NAME(HloInstructionType** matched_inst, Args&&... args) {       \
    return detail::WithOperands(Op(matched_inst)                              \
                                    .WithOpcode(HloOpcode::k##NAME)           \
                                    .WithNumOperands(sizeof...(Args)),        \
                                0,                            \
                                std::forward<Args>(args)...);                 \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType>                                      \
  inline auto NAME(HloInstructionType** matched_inst) {                       \
    return Op(matched_inst).WithOpcode(HloOpcode::k##NAME);                   \
  }
XLA_VARIADIC_OP_PATTERN(AfterAll);
XLA_VARIADIC_OP_PATTERN(AllGather)
XLA_VARIADIC_OP_PATTERN(AllReduce)
XLA_VARIADIC_OP_PATTERN(AsyncStart)
XLA_VARIADIC_OP_PATTERN(Concatenate);
XLA_VARIADIC_OP_PATTERN(Conditional);
XLA_VARIADIC_OP_PATTERN(DynamicSlice)
XLA_VARIADIC_OP_PATTERN(DynamicUpdateSlice)
XLA_VARIADIC_OP_PATTERN(Fusion);
XLA_VARIADIC_OP_PATTERN(Map)
XLA_VARIADIC_OP_PATTERN(Reduce);
XLA_VARIADIC_OP_PATTERN(ReduceScatter)
XLA_VARIADIC_OP_PATTERN(ReduceWindow)
XLA_VARIADIC_OP_PATTERN(Scatter);
XLA_VARIADIC_OP_PATTERN(Sort);
XLA_VARIADIC_OP_PATTERN(Tuple);
XLA_VARIADIC_OP_PATTERN(Call);
inline auto CustomCall() { return Op().WithOpcode(HloOpcode::kCustomCall); }
template <typename HloInstructionType>
auto CustomCall(HloInstructionType** matched_inst) {
  return Op(matched_inst).WithOpcode(HloOpcode::kCustomCall);
}
template <
    typename Arg0, typename... Args,
    typename std::enable_if<
        !std::is_convertible<Arg0, absl::string_view>::value &&
        !std::is_convertible<Arg0, HloInstruction**>::value &&
        !std::is_convertible<Arg0, const HloInstruction**>::value>::type* =
        nullptr>
auto CustomCall(Arg0&& arg0, Args&&... args) {
  return detail::WithOperands(CustomCall().WithNumOperands(sizeof...(Args) + 1),
                              0, std::forward<Arg0>(arg0),
                              std::forward<Args>(args)...);
}
template <typename... Args>
auto CustomCall(absl::Span<const absl::string_view> custom_call_targets,
                Args&&... args) {
  return CustomCall(std::forward<Args>(args)...)
      .WithCustomCallTarget(custom_call_targets);
}
template <typename HloInstructionType, typename Arg0, typename... Args,
          typename std::enable_if<!std::is_convertible<
              Arg0, absl::string_view>::value>::type* = nullptr>
auto CustomCall(HloInstructionType** matched_inst, Arg0&& arg0,
                Args&&... args) {
  return detail::WithOperands(
      CustomCall(matched_inst).WithNumOperands(sizeof...(Args) + 1),
      0, std::forward<Arg0>(arg0), std::forward<Args>(args)...);
}
template <typename HloInstructionType, typename... Args>
auto CustomCall(HloInstructionType** matched_inst,
                absl::Span<const absl::string_view> custom_call_targets,
                Args&&... args) {
  return CustomCall(matched_inst, std::forward<Args>(args)...)
      .WithCustomCallTarget(custom_call_targets);
}
#define XLA_COMPARE_PATTERN(NAME)                                             \
  inline auto NAME() {                                                        \
    return Op()                                                               \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }                                                                           \
                                                                              \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return Op()                                                               \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs))                               \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return Op(matched_inst)                                                   \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs))                               \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }
#define XLA_COMMUTATIVE_COMPARE_PATTERN(NAME)                              \
  XLA_COMPARE_PATTERN(NAME)                                                \
                                                                           \
  template <typename HloInstructionType, typename Lhs, typename Rhs>       \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs, \
                             Rhs&& rhs) {                                  \
    return Op(matched_inst)                                                \
        .WithOpcode(HloOpcode::kCompare)                                   \
        .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),                \
                                    std::forward<Rhs>(rhs));               \
  }                                                                        \
  template <typename Lhs, typename Rhs>                                    \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs) {                       \
    return NAME##AnyOrder<const HloInstruction>(                           \
        nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));          \
  }
XLA_COMMUTATIVE_COMPARE_PATTERN(Eq);
XLA_COMMUTATIVE_COMPARE_PATTERN(Ne);
XLA_COMPARE_PATTERN(Ge);
XLA_COMPARE_PATTERN(Gt);
XLA_COMPARE_PATTERN(Le);
XLA_COMPARE_PATTERN(Lt);
inline auto NonConstant() { return Op().IsNonConstant(); }
template <typename HloInstructionType>
inline auto NonConstant(HloInstructionType** matched_inst) {
  return Op(matched_inst).IsNonConstant();
}
template <typename Arg>
inline auto GetTupleElement(Arg&& arg, int64_t tuple_index) {
  return Op()
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}
template <typename HloInstructionType, typename Arg>
inline auto GetTupleElement(HloInstructionType** matched_inst, Arg&& arg,
                            int64_t tuple_index) {
  return Op(matched_inst)
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}
inline auto Parameter(int64_t parameter_num) {
  return Op().WithOpcode(HloOpcode::kParameter).WithParameterNum(parameter_num);
}
template <typename HloInstructionType>
inline auto Parameter(HloInstructionType** matched_inst,
                      int64_t parameter_num) {
  return Op(matched_inst)
      .WithOpcode(HloOpcode::kParameter)
      .WithParameterNum(parameter_num);
}
inline auto ConstantScalar() { return Op().IsConstantScalar(); }
template <typename HloInstructionType>
inline auto ConstantScalar(HloInstructionType** matched_inst) {
  return Op(matched_inst).IsConstantScalar();
}
template <typename ScalarTy>
inline auto ConstantScalar(ScalarTy val) {
  return Op().IsConstantScalar(val);
}
template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantScalar(HloInstructionType** matched_inst, ScalarTy val) {
  return Op(matched_inst).IsConstantScalar(val);
}
inline auto ConstantEffectiveScalar() {
  return Op().IsConstantEffectiveScalar();
}
template <typename HloInstructionType>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst) {
  return Op(matched_inst).IsConstantEffectiveScalar();
}
template <typename ScalarTy>
inline auto ConstantEffectiveScalar(ScalarTy val) {
  return Op().IsConstantEffectiveScalar(val);
}
template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst,
                                    ScalarTy val) {
  return Op(matched_inst).IsConstantEffectiveScalar(val);
}
namespace detail {
class InstructionPatternInterface {
 public:
  virtual ~InstructionPatternInterface() = default;
  virtual bool Match(::xla::HloInstruction* instr,
                     MatchOption option) const = 0;
  virtual void DescribeTo(std::ostream* os, int64_t indent) const = 0;
};
template <typename Pattern>
class TypedInstructionPattern : public InstructionPatternInterface {
 public:
  explicit TypedInstructionPattern(Pattern pattern)
      : pattern_(std::move(pattern)) {}
  bool Match(::xla::HloInstruction* instr, MatchOption option) const override {
    return pattern_.Match(instr, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent) const override {
    pattern_.DescribeTo(os, indent);
  }
 private:
  Pattern pattern_;
};
class HloInstructionPatternSharedImpl {
 public:
  template <typename Pattern>
  explicit HloInstructionPatternSharedImpl(Pattern pattern)
      : pattern_(std::make_shared<TypedInstructionPattern<Pattern>>(
            std::move(pattern))) {}
  bool Match(::xla::HloInstruction* instr, MatchOption option) const {
    return pattern_->Match(instr, option);
  }
  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
    pattern_->DescribeTo(os, indent);
  }
 private:
  std::shared_ptr<InstructionPatternInterface> pattern_;
};
}  
template <typename HloInstructionType, typename OriginalImpl>
inline auto SharedSubpattern(
    detail::HloInstructionPattern<HloInstructionType, OriginalImpl> pattern) {
  auto impl = detail::HloInstructionPatternSharedImpl(std::move(pattern));
  return detail::HloInstructionPattern<HloInstructionType, decltype(impl)>(
      std::move(impl), nullptr);
}
}  
}  
#undef EXPLAIN
#pragma pop_macro("EXPLAIN")
#endif  