#include "xla/service/hlo_graph_dumper.h"
#include <cstdint>
#include <unordered_map>
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/thread_annotations.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/env.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/status.h"
namespace xla {
namespace {
using absl::StrAppend;
using absl::StrCat;
using absl::StrFormat;
using absl::StrJoin;
using std::nullopt;
using std::optional;
enum NodeFilterResult {
  kNormalNode,
  kHideNode,
  kHighlightNode,
  kSomeOperandsOmitted,
  kOmitNodeOperands,
  kSomeUsersOmitted,
};
class NodeFilter {
 public:
  NodeFilter() : filter_([](const HloInstruction*) { return kNormalNode; }) {}
  explicit NodeFilter(
      std::function<NodeFilterResult(const HloInstruction* instr)> filter,
      std::optional<int> num_rendered = std::nullopt)
      : filter_(std::move(filter)), num_rendered_(num_rendered) {}
  bool Show(const HloInstruction* instr) const {
    return filter_(instr) != kHideNode;
  }
  bool Highlight(const HloInstruction* instr) const {
    return filter_(instr) == kHighlightNode;
  }
  bool OmitOperands(const HloInstruction* instr) const {
    return filter_(instr) == kOmitNodeOperands;
  }
  bool SomeOrAllOperandsOmitted(const HloInstruction* instr) const {
    auto result = filter_(instr);
    return result == kOmitNodeOperands || result == kSomeOperandsOmitted;
  }
  bool Deemphasized(const HloInstruction* instr) const {
    auto result = filter_(instr);
    return result == kOmitNodeOperands || result == kSomeOperandsOmitted ||
           result == kSomeUsersOmitted;
  }
  std::optional<int> GetNumRendered() const { return num_rendered_; }
 private:
  std::function<NodeFilterResult(const HloInstruction* instr)> filter_;
  std::optional<int> num_rendered_;
};
bool IsSmall(const HloInstruction* instr) {
  if (ShapeUtil::HasPrimitiveType(instr->shape(), OPAQUE_TYPE) ||
      ShapeUtil::HasPrimitiveType(instr->shape(), TOKEN)) {
    return true;
  }
  return ShapeUtil::ElementsInRecursive(instr->shape()) < 4096;
}
enum ColorScheme {
  kBlue,
  kBrown,
  kDarkBlue,
  kDarkGreen,
  kDarkOrange,
  kDarkRed,
  kGray,
  kGreen,
  kOrange,
  kPurple,
  kRed,
  kWhite,
  kYellow,
  kDashedBorder,
};
struct NodeColors {
  std::string style;
  std::string fill_color;
  std::string stroke_color;
  std::string font_color;
};
NodeColors NodeColorsForScheme(ColorScheme color) {
  switch (color) {
    case kBlue:
      return NodeColors{"filled", "#bbdefb", "#8aacc8", "black"};
    case kBrown:
      return NodeColors{"filled", "#bcaaa4", "#8c7b75", "black"};
    case kDarkBlue:
      return NodeColors{"filled", "#1565c0", "#003c8f", "white"};
    case kDarkGreen:
      return NodeColors{"filled", "#2e7d32", "#005005", "white"};
    case kDarkOrange:
      return NodeColors{"filled", "#ffb74d", "#c88719", "black"};
    case kDarkRed:
      return NodeColors{"filled", "#b71c1c", "#7f0000", "white"};
    case kGray:
      return NodeColors{"filled", "#cfd8dc", "#9ea7aa", "black"};
    case kGreen:
      return NodeColors{"filled", "#c8e6c9", "#97b498", "black"};
    case kOrange:
      return NodeColors{"filled", "#ffe0b2", "#cbae82", "black"};
    case kPurple:
      return NodeColors{"filled", "#e1bee7", "#af8eb5", "black"};
    case kRed:
      return NodeColors{"filled", "#ffcdd2", "#cb9ca1", "black"};
    case kWhite:
      return NodeColors{"filled", "white", "#9e9e9e", "black"};
    case kYellow:
      return NodeColors{"filled", "#fff9c4", "#cbc693", "black"};
    case kDashedBorder:
      return NodeColors{"filled,dashed", "white", "#757575", "#757575"};
  }
}
std::string NodeFillColorForStatistic(const Statistic& statistic) {
  auto stat_val = statistic.stat_val();
  if (stat_val == 0) {
    return "#f5f5f5";
  } else if (stat_val < 10) {
    return "#f7d4cc";
  } else if (stat_val < 20) {
    return "#f8b2a3";
  } else if (stat_val < 30) {
    return "#f9a28f";
  } else if (stat_val < 40) {
    return "#fa917b";
  } else if (stat_val < 50) {
    return "#fb8066";
  } else if (stat_val < 60) {
    return "#fc7052";
  } else if (stat_val < 70) {
    return "#fd5f3d";
  } else if (stat_val < 80) {
    return "#fd4e29";
  } else if (stat_val < 90) {
    return "#fe3e14";
  } else {
    return "#ff2d00";
  }
}
std::string NodeFontColorForStatistic(const Statistic& statistic) {
  if (statistic.stat_val() < 60) {
    return "black";
  } else {
    return "white";
  }
}
std::string NodeColorAttributes(ColorScheme color) {
  NodeColors node_colors = NodeColorsForScheme(color);
  return StrFormat(R"(style="%s", fontcolor="%s", color="%s", fillcolor="%s")",
                   node_colors.style, node_colors.font_color,
                   node_colors.stroke_color, node_colors.fill_color);
}
std::string HtmlLikeStringSanitize(absl::string_view s) {
  return absl::StrReplaceAll(s,
                             {{"<", "&lt;"}, {">", "&gt;"}, {"\"", "&quot;"}});
}
bool IsFusedBroadcastOfConstantEffectiveScalar(const HloInstruction* instr) {
  namespace m = match;
  return instr->parent()->IsFusionComputation() &&
         Match(instr, m::Broadcast(m::ConstantEffectiveScalar()));
}
optional<std::string> MatchTrivialComputation(
    const HloComputation* computation) {
  namespace m = match;
  if (computation->instruction_count() != 3) {
    return nullopt;
  }
  HloInstruction* root = computation->root_instruction();
  const HloInstruction *param0, *param1;
  if (!Match(root, m::Op()
                       .WithNumOperands(2)
                       .WithShape(m::Shape().IsEffectiveScalar())
                       .WithBinaryOperandsAnyOrder(
                           m::Parameter(&param0, 0)
                               .WithShape(m::Shape().IsEffectiveScalar()),
                           m::Parameter(&param1, 1)
                               .WithShape(m::Shape().IsEffectiveScalar())))) {
    return nullopt;
  }
  if (root->operand(0) == param1) {
    CHECK_EQ(root->operand(1), param0);
    if (root->opcode() == HloOpcode()) {
      switch (root->comparison_direction()) {
        case ComparisonDirection::kLe:
        case ComparisonDirection::kGe:
        case ComparisonDirection::kGt:
        case ComparisonDirection::kLt:
          return nullopt;
        default:
          break;
      }
    }
  }
  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return "add";
    case HloOpcode::kMultiply:
      return "multiply";
    case HloOpcode::kMinimum:
      return "min";
    case HloOpcode::kMaximum:
      return "max";
    case HloOpcode::kXor:
      return "xor";
    case HloOpcode::kAnd:
      return "and";
    case HloOpcode::kOr:
      return "or";
    case HloOpcode::kCompare: {
      switch (root->comparison_direction()) {
        case ComparisonDirection::kLe:
          return "less-or-equal";
        case ComparisonDirection::kGe:
          return "greater-or-equal";
        case ComparisonDirection::kGt:
          return "greater-than";
        case ComparisonDirection::kLt:
          return "less-than";
        case ComparisonDirection::kEq:
          return "equal-to";
        case ComparisonDirection::kNe:
          return "not-equal-to";
      }
    }
    default:
      return nullopt;
  }
}
class HloDotDumper {
 public:
  HloDotDumper(
      const HloComputation* computation, absl::string_view label,
      const DebugOptions& debug_options, HloRenderOptions hlo_render_options,
      NodeFilter filter,
      std::optional<absl::flat_hash_map<const HloInstruction*, ColorStats>>
          color_map = std::nullopt)
      : computation_(computation),
        label_(label),
        debug_options_(debug_options),
        hlo_render_options_(hlo_render_options),
        filter_(std::move(filter)),
        color_map_(color_map) {}
  std::string Dump();
  std::optional<std::string> CssIdForInstruction(const HloInstruction& instr) {
    if (instr.opcode() == HloOpcode::kFusion) {
      auto it = cluster_ids_.find(instr.called_computations()[0]);
      if (it == cluster_ids_.end()) {
        return std::nullopt;
      }
      return StrCat("#a_clust", it->second, " path");
    }
    auto it = node_ids_.find(&instr);
    if (it == node_ids_.end()) {
      return std::nullopt;
    }
    return StrCat("#node", it->second, " polygon");
  }
 private:
  std::string InstructionId(const HloInstruction* instruction) {
    return StrCat(reinterpret_cast<uint64_t>(instruction));
  }
  std::string SubcomputationId(const HloComputation* computation) {
    return StrCat("cluster_", reinterpret_cast<uint64_t>(computation));
  }
  std::string Header();
  std::string Footer();
  bool ShouldShowSubcomputation(const HloComputation* subcomp);
  bool ShouldShowFusionSubcomputation(const HloInstruction* instr);
  bool ShouldMergeIntoUsers(const HloInstruction* instr) const;
  std::string DumpSubcomputation(const HloComputation* subcomp,
                                 const HloInstruction* parent_instr);
  std::string DumpComputation(const HloComputation* comp);
  std::string DumpRootTag();
  std::string DumpInstruction(const HloInstruction* instr);
  ColorScheme GetInstructionColor(const HloInstruction* instr);
  std::string GetInstructionNodeShape(const HloInstruction* instr);
  std::string GetInstructionNodeLabel(const HloInstruction* instr);
  std::string GetInstructionNodeMetadata(const HloInstruction* instr);
  std::string GetInstructionNodeBackendConfig(const HloInstruction* instr);
  std::string GetInstructionNodeExtraInfo(const HloInstruction* instr);
  std::string GetInstructionNodeInlinedOperands(const HloInstruction* instr);
  void AddInstructionIncomingEdges(const HloInstruction* instr);
  const HloInstruction* GetNodeForEdge(const HloInstruction* instr);
  std::string GetInstructionTrivialComputationStr(const HloInstruction* instr);
  const HloComputation* computation_;  
  const std::string label_;            
  const DebugOptions& debug_options_;
  const HloRenderOptions hlo_render_options_;
  const NodeFilter filter_;
  const std::optional<absl::flat_hash_map<const HloInstruction*, ColorStats>>
      color_map_;
  int64_t next_node_id_ = 1;
  absl::flat_hash_map<const HloInstruction*, int64_t> node_ids_;
  int64_t root_node_id_;
  int64_t next_edge_id_ = 1;
  std::unordered_multimap<
      std::pair<const HloInstruction*, const HloInstruction*>, int64_t,
      absl::Hash<std::pair<const HloInstruction*, const HloInstruction*>>>
      edge_ids_;
  int64_t next_cluster_id_ = 1;
  absl::flat_hash_map<const HloComputation*, int64_t> cluster_ids_;
  std::vector<std::string> edges_;
  absl::flat_hash_map<HloSharding, ColorScheme> sharding_colors_;
  int64_t next_shard_color_ = 0;
};
std::string HloDotDumper::Dump() {
  std::string body;
  StrAppend(&body, DumpComputation(computation_));
  StrAppend(&body, DumpRootTag());
  std::string g = Header();
  StrAppend(&g, body);
  StrAppend(&g, Footer());
  return g;
}
std::string HloDotDumper::Header() {
  constexpr char fmt[] = R"(digraph G {
rankdir = TB;
compound = true;
label = <<b>%s</b>>;
labelloc = t;
tooltip = " ";
stylesheet=<
  data:text/css,
  @import url(https:
  svg text {
    font-family: 'Roboto';
    font-size: 12px;
  }
  %s
>
)";
  VLOG(3) << "Generating Header";
  std::string graph_label =
      StrCat(label_, "<br/>Computation ", computation_->name());
  if (computation_->IsFusionComputation()) {
    StrAppend(&graph_label, " (in fusion instruction ",
              computation_->FusionInstruction()->name(), ")");
  }
  std::vector<std::string> edge_css_rules;
  std::string kBlue = "#1976d2";
  std::string kRed = "#d32f2f";
  for (const auto& kv : edge_ids_) {
    const HloInstruction* from_node = kv.first.first;
    const HloInstruction* to_node = kv.first.second;
    int64_t edge_id = kv.second;
    auto add_hover_css_rule = [&](std::string elem_type, int64_t elem_id,
                                  std::string color) {
      edge_css_rules.push_back(
          StrFormat("  #%s%d:hover ~ #edge%d text { fill: %s; }\n"
                    "  #%s%d:hover ~ #edge%d path { "
                    "stroke: %s; stroke-width: .2em; }\n"
                    "  #%s%d:hover ~ #edge%d polygon { "
                    "fill: %s; stroke: %s; stroke-width: .2em; }\n",
                    elem_type, elem_id, edge_id, color,  
                    elem_type, elem_id, edge_id, color,  
                    elem_type, elem_id, edge_id, color, color));
    };
    int64_t from_node_id = tsl::gtl::FindWithDefault(node_ids_, from_node, -1);
    if (from_node_id == -1) {
      LOG(FATAL) << from_node->name() << " was added to edges but not to nodes";
    }
    int64_t to_node_id = to_node
                             ? tsl::gtl::FindWithDefault(node_ids_, to_node, -1)
                             : root_node_id_;
    if (to_node != nullptr && to_node_id == -1) {
      LOG(FATAL) << to_node->name() << " was added to edges but not to nodes";
    }
    add_hover_css_rule("node", from_node_id, kBlue);
    add_hover_css_rule("node", to_node_id, kRed);
    if (to_node) {
      VLOG(3) << "Adding css for edge " << edge_id << " from node "
              << from_node->name() << " to node " << to_node->name();
    } else {
      VLOG(3) << "Adding css for edge " << edge_id << " from node "
              << from_node->name() << " to root tag";
    }
    if (to_node) {
      if (from_node->IsFused() &&
          from_node->parent()->root_instruction() == from_node) {
        int64_t cluster_id = cluster_ids_.at(from_node->parent());
        add_hover_css_rule("clust", cluster_id, kBlue);
      }
      if (to_node->IsFused() && to_node->opcode() == HloOpcode::kParameter) {
        int64_t cluster_id = cluster_ids_.at(to_node->parent());
        add_hover_css_rule("clust", cluster_id, kRed);
      }
    }
  }
  return StrFormat(
      fmt, graph_label,
      absl::StrReplaceAll(StrJoin(edge_css_rules, "\n"), {{"#", "%23"}}));
}
std::string HloDotDumper::Footer() {
  return StrCat(StrJoin(edges_, "\n"), "\n}");
}
bool HloDotDumper::ShouldShowFusionSubcomputation(const HloInstruction* instr) {
  CHECK_EQ(instr->opcode(), HloOpcode::kFusion);
  return ShouldShowSubcomputation(instr->fused_instructions_computation());
}
bool HloDotDumper::ShouldShowSubcomputation(const HloComputation* subcomp) {
  if (subcomp->IsFusionComputation()) {
    const HloInstruction* fusion = subcomp->FusionInstruction();
    if (!filter_.Show(fusion) || filter_.SomeOrAllOperandsOmitted(fusion) ||
        !hlo_render_options_.show_fusion_subcomputations) {
      return false;
    }
  }
  if (!subcomp->IsFusionComputation() && MatchTrivialComputation(subcomp)) {
    return false;
  }
  if (subcomp->WhileCallInstruction() != nullptr &&
      !hlo_render_options_.show_while_subcomputations) {
    return false;
  }
  return absl::c_any_of(
      subcomp->instructions(),
      [&](const HloInstruction* instr) { return filter_.Show(instr); });
}
std::string HloDotDumper::DumpSubcomputation(
    const HloComputation* subcomp, const HloInstruction* parent_instr) {
  VLOG(2) << "Dumping subcomputation " << subcomp->name();
  if (parent_instr->opcode() != HloOpcode::kFusion) {
    const HloInstruction* from = GetNodeForEdge(subcomp->root_instruction());
    VLOG(2) << "Edge: from " << from->name() << " to " << parent_instr->name()
            << " as " << next_edge_id_;
    edge_ids_.insert({{from, parent_instr}, next_edge_id_++});
    constexpr char edge_fmt[] =
        R"(%s -> %s [ltail="%s", style="dashed" tooltip="%s -> %s"];)";
    edges_.push_back(StrFormat(
        edge_fmt, InstructionId(from), InstructionId(parent_instr),
        SubcomputationId(subcomp), subcomp->name(), parent_instr->name()));
  }
  if (cluster_ids_.find(subcomp) != cluster_ids_.end()) {
    return "";
  }
  cluster_ids_[subcomp] = next_cluster_id_++;
  std::string id = SubcomputationId(subcomp);
  std::string subcomp_label, style;
  if (parent_instr->opcode() == HloOpcode::kFusion) {
    subcomp_label =
        StrFormat("Fused expression for <b>%s</b><br/>%s",
                  HtmlLikeStringSanitize(parent_instr->name()),
                  HtmlLikeStringSanitize(parent_instr->ToCategory()));
    std::string extra_info = GetInstructionNodeExtraInfo(parent_instr);
    if (!extra_info.empty()) {
      StrAppend(&subcomp_label, "<br/>", extra_info);
    }
    std::string node_backend_config =
        GetInstructionNodeBackendConfig(parent_instr);
    if (!node_backend_config.empty()) {
      StrAppend(&subcomp_label, "<br/>", node_backend_config);
    }
    bool highlight = filter_.Highlight(parent_instr);
    std::string fillcolor;
    std::string strokecolor;
    if (!highlight && (parent_instr->module_has_statistics() ||
                       parent_instr->has_statistics())) {
      fillcolor = parent_instr->has_statistics()
                      ? NodeFillColorForStatistic(
                            parent_instr->statistic_to_visualize())
                      : "#f5f5f5";
      strokecolor = "#c2c2c2";
    } else if (debug_options_.xla_hlo_graph_sharding_color() && !highlight) {
      NodeColors node_colors =
          NodeColorsForScheme(GetInstructionColor(parent_instr));
      fillcolor = node_colors.fill_color;
      strokecolor = node_colors.stroke_color;
    } else {
      fillcolor = highlight ? "#ffcdd2" : "#f5f5f5";
      strokecolor = highlight ? "#b71c1c" : "#c2c2c2";
    }
    style =
        StrFormat(R"(style="rounded,filled,bold"; fillcolor="%s"; color="%s;")",
                  fillcolor, strokecolor);
  } else {
    subcomp_label = StrFormat("Subcomputation for <b>%s</b><br/>%s",
                              HtmlLikeStringSanitize(parent_instr->name()),
                              HtmlLikeStringSanitize(subcomp->name()));
    style = "style=rounded; color=black;";
  }
  std::string comp_body = DumpComputation(subcomp);
  constexpr char computation_fmt[] = R"(subgraph %s {
%s
label = <%s>;
labelloc = t;
tooltip = " ";
%s
}  
)";
  return StrFormat(computation_fmt, id, style, subcomp_label, comp_body, id);
}
std::string HloDotDumper::DumpComputation(const HloComputation* comp) {
  std::string g;
  for (const auto* instr : comp->instructions()) {
    if (!filter_.Show(instr)) {
      continue;
    }
    for (const HloComputation* subcomp : instr->called_computations()) {
      if (ShouldShowSubcomputation(subcomp)) {
        StrAppend(&g, DumpSubcomputation(subcomp, instr));
      }
    }
    StrAppend(&g, DumpInstruction(instr));
  }
  return g;
}
std::string HloDotDumper::DumpRootTag() {
  const HloInstruction* from = GetNodeForEdge(computation_->root_instruction());
  if (!filter_.Show(from) || from->opcode() == HloOpcode::kConstant ||
      IsFusedBroadcastOfConstantEffectiveScalar(from)) {
    return "";
  }
  auto from_id = InstructionId(from);
  HloInstruction* to = nullptr;
  auto to_id = SubcomputationId(computation_);
  std::string node_body = "ROOT";
  std::string node_shape = "circle";
  ColorScheme color = kBrown;
  VLOG(2) << "Adding root tag as node " << next_node_id_;
  root_node_id_ = next_node_id_++;
  VLOG(2) << "Adding edge from " << from->name() << " to root tag as "
          << next_edge_id_;
  edge_ids_.insert({{from, to}, next_edge_id_++});
  edges_.push_back(StrFormat(R"(%s -> %s [tooltip=" "];)", from_id, to_id));
  return StrFormat(R"(%s [label=<%s>, shape=%s, tooltip=" ", %s];)"
                   "\n",
                   to_id, node_body, node_shape, NodeColorAttributes(color));
}
static const HloConstantInstruction* TryGetFusionParameterConstant(
    const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kParameter || !instr->IsFused()) {
    return nullptr;
  }
  const HloInstruction* fusion = instr->parent()->FusionInstruction();
  const HloInstruction* operand = fusion->operand(instr->parameter_number());
  return DynCast<HloConstantInstruction>(operand);
}
bool HloDotDumper::ShouldMergeIntoUsers(const HloInstruction* instr) const {
  if ((instr->opcode() == HloOpcode::kGetTupleElement &&
       instr != instr->parent()->root_instruction()) ||
      TryGetFusionParameterConstant(instr) != nullptr) {
    return true;
  }
  const int kMinUsersToOmit = 3;
  return instr->opcode() == HloOpcode::kParameter && instr->shape().IsTuple() &&
         !instr->IsFused() &&
         absl::c_count_if(instr->users(),
                          [&](const HloInstruction* user) {
                            return filter_.Show(user);
                          }) > kMinUsersToOmit &&
         absl::c_all_of(instr->users(), [&](const HloInstruction* user) {
           return !filter_.Show(user) ||
                  user->opcode() == HloOpcode::kGetTupleElement;
         });
}
std::string HloDotDumper::DumpInstruction(const HloInstruction* instr) {
  if ((instr->opcode() == HloOpcode::kConstant ||
       IsFusedBroadcastOfConstantEffectiveScalar(instr)) &&
      instr != instr->parent()->root_instruction()) {
    return "";
  }
  if (ShouldMergeIntoUsers(instr)) {
    return "";
  }
  if (instr->opcode() == HloOpcode::kFusion &&
      ShouldShowFusionSubcomputation(instr)) {
    return "";
  }
  VLOG(2) << "Adding node " << instr->name() << " as " << next_node_id_;
  node_ids_[instr] = next_node_id_++;
  std::string node_shape = GetInstructionNodeShape(instr);
  std::string node_label = GetInstructionNodeLabel(instr);
  std::string node_metadata = GetInstructionNodeMetadata(instr);
  std::string node_backend_config = GetInstructionNodeBackendConfig(instr);
  std::string extra_info = GetInstructionNodeExtraInfo(instr);
  std::string inlined_constants = GetInstructionNodeInlinedOperands(instr);
  std::string trivial_subcomputation =
      GetInstructionTrivialComputationStr(instr);
  AddInstructionIncomingEdges(instr);
  NodeColors node_colors;
  std::string node_style;
  std::string node_attributes;
  if (hlo_render_options_.override_node_colors && color_map_.has_value()) {
    if (color_map_->contains(instr)) {
      node_colors.fill_color = color_map_->at(instr).color;
      node_attributes = color_map_->at(instr).stats;
    } else {
      VLOG(2) << "color_map_ for instruction:" << instr->name() << "is empty"
              << "\n";
      node_colors.fill_color = "#808080";
    }
    node_colors.style = "filled";
    node_colors.font_color = "black";
    node_colors.stroke_color = "#c2c2c2";
    node_style =
        StrFormat(R"(style="%s", fontcolor="%s", color="%s", fillcolor="%s")",
                  node_colors.style, node_colors.font_color,
                  node_colors.stroke_color, node_colors.fill_color);
  } else {
    ColorScheme color = GetInstructionColor(instr);
    if (!debug_options_.xla_hlo_graph_sharding_color()) {
      if (filter_.Deemphasized(instr)) {
        color = kDashedBorder;
      }
      if (filter_.Highlight(instr)) {
        node_shape = "diamond";
        color = kDarkRed;
      }
    }
    node_colors = NodeColorsForScheme(color);
    if (instr->has_statistics()) {
      const auto& statistic_to_visualize = instr->statistic_to_visualize();
      node_colors.fill_color =
          NodeFillColorForStatistic(statistic_to_visualize);
      node_colors.stroke_color = "#c2c2c2";
      node_colors.font_color =
          NodeFontColorForStatistic(statistic_to_visualize);
    } else if (instr->module_has_statistics()) {
      node_colors.fill_color = "#f5f5f5";
      node_colors.stroke_color = "#c2c2c2";
      node_colors.font_color = "black";
    }
    node_style =
        StrFormat(R"(style="%s", fontcolor="%s", color="%s", fillcolor="%s")",
                  node_colors.style, node_colors.font_color,
                  node_colors.stroke_color, node_colors.fill_color);
  }
  std::string node_body = node_label;
  for (const std::string& s :
       {trivial_subcomputation, extra_info, inlined_constants,
        node_backend_config, node_attributes}) {
    if (!s.empty()) {
      StrAppend(&node_body, "<br/>", s);
    }
  }
  return StrFormat(R"(%s [label=<%s>, shape=%s, tooltip="%s", %s];)"
                   "\n",
                   InstructionId(instr), node_body, node_shape, node_metadata,
                   node_style);
}
std::string HloDotDumper::GetInstructionNodeInlinedOperands(
    const HloInstruction* instr) {
  auto stringify_constant = [](const HloConstantInstruction* constant,
                               const Shape& shape) {
    if (ShapeUtil::IsZeroElementArray(shape)) {
      return StrFormat("{} (%s)", ShapeUtil::HumanString(constant->shape()));
    }
    optional<int64_t> elem_count;
    if (shape.IsArray()) {
      elem_count = ShapeUtil::ElementsIn(constant->shape());
    }
    if (elem_count.has_value() && *elem_count <= 8 && constant->HasLiteral()) {
      std::string literal_str = constant->literal().ToStringWithoutShape();
      if (literal_str.size() <= 64) {
        return StrFormat("%s %s", shape.ToString(), literal_str);
      }
    }
    std::string constant_name;
    if (absl::StartsWith(constant->name(), "constant")) {
      constant_name = std::string(constant->name());
    } else {
      constant_name = StrCat("constant ", constant->name());
    }
    return StrFormat("%s %s", constant_name, ShapeUtil::HumanString(shape));
  };
  std::vector<std::string> lines;
  constexpr int64_t kMaxOperandsShown = 32;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    optional<std::string> operand_str;
    if (const auto* constant_operand =
            DynCast<HloConstantInstruction>(operand)) {
      operand_str =
          stringify_constant(constant_operand, constant_operand->shape());
    } else if (IsFusedBroadcastOfConstantEffectiveScalar(operand)) {
      operand_str = stringify_constant(
          Cast<HloConstantInstruction>(operand->operand(0)), operand->shape());
    } else if (ShouldMergeIntoUsers(operand)) {
      if (operand->opcode() == HloOpcode::kParameter) {
        if (const HloConstantInstruction* constant =
                TryGetFusionParameterConstant(operand)) {
          operand_str = stringify_constant(constant, constant->shape());
        } else {
          operand_str = StrFormat("Parameter %d", operand->parameter_number());
        }
      } else if (operand->opcode() == HloOpcode::kGetTupleElement) {
        operand_str =
            StrFormat("tuple-element %d of %s %s", operand->tuple_index(),
                      operand->operand(0)->name(),
                      ShapeUtil::HumanStringWithLayout(operand->shape()));
      } else {
        operand_str = std::string(operand->name());
      }
    }
    if (operand_str) {
      if (instr->operand_count() > 1) {
        lines.push_back(StrFormat("<b>operand %d</b> = %s", i, *operand_str));
      } else {
        lines.push_back(StrFormat("<b>operand</b> = %s", *operand_str));
      }
    }
    if (lines.size() == kMaxOperandsShown && i < instr->operand_count() - 1) {
      lines.push_back("...");
      break;
    }
  }
  if (instr->opcode() == HloOpcode::kParameter && instr->IsFused()) {
    const HloInstruction* param_input =
        instr->parent()->FusionInstruction()->operand(
            instr->parameter_number());
    if (param_input->opcode() == HloOpcode::kGetTupleElement) {
      lines.push_back(
          StrFormat("tuple-element %d of %s %s", param_input->tuple_index(),
                    param_input->operand(0)->name(),
                    ShapeUtil::HumanStringWithLayout(param_input->shape())));
    }
  }
  return StrJoin(lines, "<br/>");
}
ColorScheme HloDotDumper::GetInstructionColor(const HloInstruction* instr) {
  if (debug_options_.xla_hlo_graph_sharding_color()) {
    if (!instr->has_sharding()) {
      return kDashedBorder;
    }
    auto it = sharding_colors_.find(instr->sharding());
    if (it != sharding_colors_.end()) {
      return it->second;
    }
    ColorScheme color = static_cast<ColorScheme>(
        kBlue + (next_shard_color_++ % (kDashedBorder - kBlue)));
    sharding_colors_.emplace(instr->sharding(), color);
    return color;
  }
  auto parameter_color = IsSmall(instr) ? kOrange : kDarkOrange;
  if (absl::c_any_of(instr->operands(), [&](const HloInstruction* operand) {
        return operand->opcode() == HloOpcode::kParameter &&
               ShouldMergeIntoUsers(operand) &&
               TryGetFusionParameterConstant(operand) == nullptr;
      })) {
    return parameter_color;
  }
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRemainder:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
      return kWhite;
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPad:
    case HloOpcode::kTuple:
      return kWhite;
    case HloOpcode::kConstant:
      return kWhite;
    case HloOpcode::kBroadcast:
    case HloOpcode::kDynamicUpdateSlice:
      return kYellow;
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return kGreen;
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
      return kGreen;
    case HloOpcode::kBitcast:
      if (!instr->IsFused()) {
        return kWhite;
      }
      return kGreen;
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return GetInstructionColor(instr->async_wrapped_instruction());
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kFft:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      return kDarkBlue;
    case HloOpcode::kParameter:
      return parameter_color;
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:  
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kGather:  
      return kPurple;
    case HloOpcode::kDomain:
    case HloOpcode::kFusion:
    case HloOpcode::kMap:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return kGray;
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPartitionId:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kReplicaId:
      return kBrown;
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
      return kDarkGreen;
  }
}
std::string HloDotDumper::GetInstructionNodeShape(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kWhile:
      return "ellipse";
    default:
      return "rect";
  }
}
std::string HloDotDumper::GetInstructionNodeLabel(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kParameter) {
    return StrFormat("<b>Parameter %d</b>", instr->parameter_number());
  }
  if (absl::StartsWith(instr->name(), HloOpcodeString(instr->opcode()))) {
    return StrFormat("<b>%s</b>", HtmlLikeStringSanitize(instr->name()));
  }
  std::string extended_opcode =
      StrCat(HloOpcodeString(instr->opcode()),
             instr->opcode() != HloOpcode::kFusion
                 ? ""
                 : StrCat(":", xla::ToString(instr->fusion_kind())));
  return StrFormat("<b>%s</b><br/>%s", HtmlLikeStringSanitize(instr->name()),
                   HtmlLikeStringSanitize(extended_opcode));
}
std::string HloDotDumper::GetInstructionNodeMetadata(
    const HloInstruction* instr) {
  std::vector<std::string> lines;
  if (!instr->metadata().op_name().empty()) {
    lines.push_back(HtmlLikeStringSanitize(instr->metadata().op_name()));
  }
  if (!instr->metadata().op_type().empty()) {
    lines.push_back(StrFormat(
        "op_type: %s", HtmlLikeStringSanitize(instr->metadata().op_type())));
  }
  if (!instr->metadata().source_file().empty() &&
      instr->metadata().source_line() != 0) {
    lines.push_back(StrFormat("source: %s:%d", instr->metadata().source_file(),
                              instr->metadata().source_line()));
  }
  if (instr->metadata().stack_frame_id() != 0) {
    auto hlo_module = instr->parent()->parent();
    int frame_id = instr->metadata().stack_frame_id();
    while (frame_id != 0) {
      HloModule::StackFrame frame = hlo_module->get_stack_frame(frame_id);
      if (frame.empty()) {
        break;
      }
      frame_id = frame.parent_frame_id;
      lines.push_back(StrFormat(
          "%s:%s:%d%s", frame.file_name, frame.function_name, frame.line,
          frame.column == 0 ? "" : StrFormat(":%d", frame.column)));
    }
  }
  return StrJoin(lines, "\n");
}
static std::vector<std::pair<std::string, std::string>>
ExtractCudnnConvBackendConfigProps(const gpu::CudnnConvBackendConfig& config) {
  std::vector<std::pair<std::string, std::string>> props;
  if (config.conv_result_scale() != 1) {
    props.emplace_back("conv_result_scale", StrCat(config.conv_result_scale()));
  }
  if (config.side_input_scale() != 0 && config.side_input_scale() != 1) {
    props.emplace_back("side_input_scale", StrCat(config.side_input_scale()));
  }
  if (config.activation_mode() == se::dnn::ActivationMode::kLeakyRelu) {
    props.emplace_back("leakyrelu_alpha", StrCat(config.leakyrelu_alpha()));
  }
  props.emplace_back(
      "activation_mode",
      se::dnn::ActivationModeString(
          static_cast<se::dnn::ActivationMode>(config.activation_mode())));
  props.emplace_back("algo",
                     se::dnn::AlgorithmDesc(config.algorithm()).ToString());
  return props;
}
static std::vector<std::pair<std::string, std::string>>
ExtractGemmBackendConfigProps(const gpu::GemmBackendConfig& config,
                              const HloInstruction* instr) {
  std::vector<std::pair<std::string, std::string>> props;
  if (primitive_util::IsComplexType(instr->shape().element_type())) {
    if (config.alpha_real() != 1 || config.alpha_imag() != 1) {
      props.emplace_back("alpha_real", StrCat(config.alpha_real()));
      props.emplace_back("alpha_imag", StrCat(config.alpha_real()));
    }
  } else {
    if (config.alpha_real() != 1) {
      props.emplace_back("alpha", StrCat(config.alpha_real()));
    }
  }
  if (config.beta() != 0 && config.beta() != 1) {
    props.emplace_back("beta", StrCat(config.beta()));
  }
  props.emplace_back(
      "", absl::StrReplaceAll(
              DotDimensionNumbersToString(config.dot_dimension_numbers()),
              {{", ", "<br/>"}}));
  if (config.algorithm_case() == gpu::GemmBackendConfig::kSelectedAlgorithm) {
    props.emplace_back("algorithm", StrCat(config.selected_algorithm()));
  }
  if (config.epilogue() != gpu::GemmBackendConfig::DEFAULT) {
    props.emplace_back(
        "epilogue", gpu::GemmBackendConfig::Epilogue_Name(config.epilogue()));
  }
  return props;
}
std::string HloDotDumper::GetInstructionNodeBackendConfig(
    const HloInstruction* instr) {
  std::vector<std::pair<std::string, std::string>> props;
  if (gpu::IsCustomCallToDnnConvolution(*instr)) {
    absl::StatusOr<gpu::GpuBackendConfig> config =
        instr->backend_config<gpu::GpuBackendConfig>();
    if (config.ok()) {
      props = ExtractCudnnConvBackendConfigProps(
          config->cudnn_conv_backend_config());
    }
  } else if (gpu::IsCublasGemm(*instr)) {
    absl::StatusOr<gpu::GpuBackendConfig> config =
        instr->backend_config<gpu::GpuBackendConfig>();
    if (config.ok()) {
      props =
          ExtractGemmBackendConfigProps(config->gemm_backend_config(), instr);
    }
  }
  if (!props.empty()) {
    return StrCat((props.size() > 1 ? "<br/>" : ""),
                  StrJoin(props, "<br/>",
                          [](std::string* out,
                             const std::pair<std::string, std::string>& kv) {
                            if (!kv.first.empty()) {
                              return StrAppend(out, kv.first, "=", kv.second);
                            }
                            StrAppend(out, kv.second);
                          }));
  }
  if (!hlo_render_options_.show_backend_config ||
      instr->raw_backend_config_string().empty()) {
    return "";
  }
  return StrCat("backend_config=\"", instr->raw_backend_config_string(), "\"");
}
std::string HloDotDumper::GetInstructionNodeExtraInfo(
    const HloInstruction* instr) {
  std::vector<std::string> lines;
  for (const auto& line : instr->ExtraAttributesToString(
           HloPrintOptions().set_print_subcomputation_mode(
               HloPrintOptions::PrintSubcomputationMode::kOff))) {
    constexpr int kMaxDeviceIdFieldLen = 128;
    if ((absl::StartsWith(line, "replica_groups=") ||
         absl::StartsWith(line, "source_target_pairs=") ||
         absl::StartsWith(line, "control-predecessors=")) &&
        line.length() > kMaxDeviceIdFieldLen) {
      lines.push_back(HtmlLikeStringSanitize(
          StrCat(line.substr(0, kMaxDeviceIdFieldLen - 3), "...")));
    } else if (absl::StartsWith(line, "feature_group_count=")) {
      lines.push_back(StrFormat("<b>%s</b>", HtmlLikeStringSanitize(line)));
    } else {
      lines.push_back(HtmlLikeStringSanitize(line));
    }
  }
  if (instr->opcode() != HloOpcode::kFusion ||
      !ShouldShowFusionSubcomputation(instr)) {
    bool shape_is_multidim = false;
    ShapeUtil::ForEachSubshape(instr->shape(),
                               [&](const Shape& s, const ShapeIndex&) {
                                 shape_is_multidim |= s.dimensions_size() > 1;
                               });
    std::string instr_shape;
    if (instr->opcode() != HloOpcode::kTuple && shape_is_multidim) {
      instr_shape = ShapeUtil::HumanStringWithLayout(instr->shape());
    } else {
      instr_shape = ShapeUtil::HumanString(instr->shape());
    }
    constexpr int kMaxShapeLen = 64;
    if (instr_shape.length() > kMaxShapeLen) {
      instr_shape = StrCat(
          absl::string_view(instr_shape).substr(0, kMaxShapeLen - 3), "...");
    }
    lines.push_back(HtmlLikeStringSanitize(instr_shape));
  }
  if (debug_options_.xla_hlo_graph_addresses()) {
    lines.push_back(StrFormat("[%p]", instr));
  }
  return StrJoin(lines, "<br/>");
}
void HloDotDumper::AddInstructionIncomingEdges(const HloInstruction* instr) {
  constexpr int kMaxEdgesBetweenTwoNodes = 64;
  auto add_edge = [&](const HloInstruction* from, const HloInstruction* to,
                      int64_t operand_num, bool control_edge = false) {
    if (edge_ids_.count({from, to}) > kMaxEdgesBetweenTwoNodes) {
      return;
    }
    from = GetNodeForEdge(from);
    if (!filter_.Show(from) || from->opcode() == HloOpcode::kConstant ||
        IsFusedBroadcastOfConstantEffectiveScalar(from) ||
        ShouldMergeIntoUsers(from)) {
      return;
    }
    VLOG(2) << "Adding edge from " << from->name() << " to " << to->name()
            << " as " << next_edge_id_;
    edge_ids_.insert({{from, to}, next_edge_id_++});
    std::string edge_label;
    if (control_edge) {
      edge_label = "style=\"dotted\" color=\"gray\" label=\"ctrl\"";
    } else if (instr->operand_count() > 1) {
      edge_label =
          StrFormat(R"( headlabel="%d", labeldistance=2)", operand_num);
    }
    constexpr char kEdgeFmt[] =
        R"(%s -> %s [arrowhead=%s tooltip="%s -> %s" %s];)";
    edges_.push_back(StrFormat(kEdgeFmt, InstructionId(from), InstructionId(to),
                               (IsSmall(from) ? "empty" : "normal"),
                               from->name(), to->name(), edge_label));
  };
  if (instr->opcode() == HloOpcode::kParameter && instr->IsFused()) {
    if (instr->parent() != computation_) {
      const HloInstruction* fusion = instr->parent()->FusionInstruction();
      add_edge(fusion->operand(instr->parameter_number()), instr,
               0);
    }
  } else {
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      add_edge(instr->operand(i), instr, i);
    }
    for (const HloInstruction* pred : instr->control_predecessors()) {
      add_edge(pred, instr, 0, true);
    }
  }
}
std::string HloDotDumper::GetInstructionTrivialComputationStr(
    const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kFusion) {
    return "";
  }
  std::vector<std::string> lines;
  for (int64_t i = 0; i < instr->called_computations().size(); ++i) {
    optional<std::string> computation_type =
        MatchTrivialComputation(instr->called_computations()[i]);
    if (!computation_type) {
      continue;
    }
    if (instr->called_computations().size() == 1) {
      lines.push_back(StrFormat("Subcomputation: <b>%s</b>",
                                HtmlLikeStringSanitize(*computation_type)));
    } else {
      lines.push_back(StrFormat("Subcomputation %d: <b>%s</b>", i,
                                HtmlLikeStringSanitize(*computation_type)));
    }
  }
  return StrJoin(lines, "<br/>");
}
const HloInstruction* HloDotDumper::GetNodeForEdge(
    const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kGetTupleElement) {
    instr = instr->operand(0);
  }
  while (instr->opcode() == HloOpcode::kFusion &&
         ShouldShowFusionSubcomputation(instr)) {
    instr = instr->fused_expression_root();
  }
  return instr;
}
NodeFilter MakeNodeRadiusAroundFilter(
    const HloInstruction* root, int64_t radius,
    const absl::flat_hash_set<const HloInstruction*>& boundary) {
  absl::flat_hash_map<const HloInstruction*, NodeFilterResult> nodes;
  std::deque<std::pair<const HloInstruction*,  int64_t>> worklist;
  worklist.push_back({root, 0});
  while (!worklist.empty()) {
    const HloInstruction* instr;
    int64_t depth;
    std::tie(instr, depth) = worklist.front();
    worklist.pop_front();
    nodes[instr] = kNormalNode;
    if (depth == radius) {
      continue;
    }
    if (boundary.contains(instr)) {
      continue;
    }
    if (instr == root || instr->opcode() != HloOpcode::kTuple) {
      for (const HloInstruction* operand : instr->operands()) {
        if (!nodes.contains(operand)) {
          int new_depth = (operand->opcode() == HloOpcode::kBitcast ||
                           instr->opcode() == HloOpcode::kBitcast)
                              ? depth
                              : depth + 1;
          worklist.push_back({operand, new_depth});
        }
      }
    }
    for (const HloComputation* computation : instr->called_computations()) {
      worklist.push_back({computation->root_instruction(), depth + 1});
    }
    if (instr->opcode() == HloOpcode::kConstant) {
      continue;
    }
    constexpr int kMaxUsersToRender = 16;
    if (instr->user_count() > kMaxUsersToRender) {
      nodes[instr] = kSomeUsersOmitted;
      continue;
    }
    for (const HloInstruction* user : instr->users()) {
      if (!nodes.contains(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }
  auto is_displayed = [&](const HloInstruction* instr) {
    return nodes.contains(instr) || instr->opcode() == HloOpcode::kConstant ||
           instr->parent() != root->parent();
  };
  for (auto& kv : nodes) {
    const HloInstruction* instr = kv.first;
    NodeFilterResult& filter_result = kv.second;
    const auto& operands = instr->operands();
    if (absl::c_any_of(operands, is_displayed) &&
        !absl::c_all_of(operands, is_displayed)) {
      filter_result = kSomeOperandsOmitted;
    } else if (!operands.empty() && absl::c_none_of(operands, is_displayed)) {
      filter_result = kOmitNodeOperands;
    }
    if (filter_result == kSomeUsersOmitted &&
        absl::c_all_of(instr->users(), is_displayed)) {
      filter_result = kNormalNode;
    }
  }
  nodes[root] = kHighlightNode;
  return NodeFilter(
      [=](const HloInstruction* instr) {
        auto it = nodes.find(instr);
        if (it != nodes.end()) {
          return it->second;
        }
        if (instr->parent() != root->parent()) {
          return kNormalNode;
        }
        return kHideNode;
      },
      nodes.size());
}
NodeFilter MakeNodeFromToFilter(const HloInstruction* from,
                                const HloInstruction* to, int64_t max_nodes,
                                bool* hit_limit) {
  *hit_limit = false;
  std::deque<std::vector<const HloInstruction*>> queue;
  queue.push_front({from});
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::flat_hash_set<const HloInstruction*> to_display = {from, to};
  while (!queue.empty() && to_display.size() < max_nodes) {
    std::vector<const HloInstruction*> path = std::move(queue.front());
    queue.pop_front();
    if (!visited.insert(path.back()).second) {
      continue;
    }
    for (const auto* user : path.back()->users()) {
      if (user == to) {
        auto it = path.begin();
        for (; it != path.end() && to_display.size() < max_nodes; ++it) {
          to_display.insert(*it);
        }
        if (it != path.end()) {
          *hit_limit = true;
        }
      } else if (!visited.count(user)) {
        auto new_path = path;
        new_path.push_back(user);
        queue.push_back(std::move(new_path));
      }
    }
  }
  return NodeFilter([=](const HloInstruction* instr) {
    if (instr == from || instr == to) {
      return kHighlightNode;
    }
    return to_display.count(instr) ? kNormalNode : kHideNode;
  });
}
absl::Mutex url_renderer_mu(absl::kConstInit);
std::function<absl::StatusOr<std::string>(absl::string_view)>* url_renderer
    ABSL_GUARDED_BY(url_renderer_mu) = nullptr;
absl::Mutex fusion_visualizer_state_mu(absl::kConstInit);
namespace {
struct FusionVisualizerProgress {
  void AddState(absl::string_view dot, absl::string_view explanation,
                std::optional<std::string> to_highlight) {
    if (dot_graphs.empty() || dot_graphs.back() != dot) {
      dot_graphs.push_back(std::string(dot));
    }
    frames.push_back({static_cast<int>(dot_graphs.size() - 1),
                      std::string(explanation), to_highlight.value_or("")});
  }
  std::vector<std::string> dot_graphs;
  struct FusionFrame {
    int dot_graph;
    std::string label;
    std::string to_highlight;
  };
  std::vector<FusionFrame> frames;
};
}  
static auto& fusion_visualizer_states
    TF_GUARDED_BY(fusion_visualizer_state_mu) = *new absl::flat_hash_map<
        std::pair<int64_t, int64_t>, FusionVisualizerProgress>();
static std::pair<int, int> FusionVisualizerStateKey(
    const HloComputation& computation) {
  return std::make_pair(computation.parent()->unique_id(),
                        computation.unique_id());
}
}  
static absl::StatusOr<std::string> CompressAndEncode(absl::string_view input) {
  class WritableStringFile : public tsl::WritableFile {
   public:
    explicit WritableStringFile(std::string* data) : data_(data){};
    ~WritableStringFile() override = default;
    absl::Status Append(absl::string_view data) override {
      absl::StrAppend(data_, data);
      return absl::OkStatus();
    }
    absl::Status Close() override { return absl::OkStatus(); }
    absl::Status Flush() override { return absl::OkStatus(); }
    absl::Status Sync() override { return absl::OkStatus(); }
   private:
    std::string* data_;
  };
  std::string compressed;
  WritableStringFile f(&compressed);
  auto gz_opts = tsl::io::ZlibCompressionOptions::GZIP();
  tsl::io::ZlibOutputBuffer gz_file(&f, gz_opts.input_buffer_size,
                                    gz_opts.output_buffer_size, gz_opts);
  TF_RETURN_IF_ERROR(gz_file.Init());
  TF_RETURN_IF_ERROR(gz_file.Append(input));
  TF_RETURN_IF_ERROR(gz_file.Close());
  std::string encoded;
  TF_RETURN_IF_ERROR(tsl::Base64Encode(compressed, &encoded));
  return absl::StrReplaceAll(encoded, {{"_", "/"}, {"-", "+"}});
}
static std::string EscapeJSONString(absl::string_view raw) {
  return absl::StrCat(
      "\"",
      absl::StrReplaceAll(raw, {{"\n", "\\n"}, {"\"", "\\\""}, {"\\", "\\\\"}}),
      "\"");
}
absl::StatusOr<std::string> WrapFusionExplorer(
    const FusionVisualizerProgress& visualizer_progress,
    absl::string_view graph_title) {
  if (visualizer_progress.frames.empty()) {
    return Internal("Empty");
  }
  std::string dot_graphs =
      StrFormat("[%s]", StrJoin(visualizer_progress.dot_graphs, ", ",
                                [&](std::string* out, const std::string& dot) {
                                  StrAppend(out, EscapeJSONString(dot));
                                }));
  std::string frames = StrJoin(
      visualizer_progress.frames, ", ", [&](std::string* out, const auto& p) {
        StrAppend(out, StrFormat("[%d, %s, %s]", p.dot_graph,
                                 EscapeJSONString(p.label),
                                 EscapeJSONString(p.to_highlight)));
      });
  TF_ASSIGN_OR_RETURN(std::string dot_graphs_compressed,
                      CompressAndEncode(dot_graphs));
  return absl::StrReplaceAll(
      R"wrapper(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {height: 100%; text-align: center;}
    #rendered {height: 70%; width: 80%; border:1px solid black; margin: auto; }
    #label {width: 80%; margin: auto;}
    #performance_note { font-size: small; color: gray; }
    #frames_list {
      list-style: none; text-align: left; height: 20%; overflow: scroll;
    }
    #frames_list   li { padding: 0.2em; margin: 0.2em; }
    .selected { background-color: #e0e0e0; }
    .selected a { color: black; text-decoration: none; }
    #rendered svg { height: 100% !important; width: 100% !important; }
  </style>
</head>
<body>
  <script src="https:
      integrity="sha384-LigJPbR3TOfU/Xbb+PjiN1dGJYPweLk7kiGnaMgmxnUmKWaCFKbb5tH6iLlyVhPZ"
      crossorigin="anonymous"></script>
  <script src="https:
  </script>
  <title>Fusion Explorer: $TITLE</title>
  <div id='rendered'><center>Loading...</center></div>
  <ul id='frames_list'></ul>
  <p>Use j/k for keyboard navigation.</p>
  <p id='performance_note'>Loading data...</p>
  <script>
  <!--
  const renderCache = {};
  const cssregex = new RegExp('stylesheet=<([^]*)\n>\n', 'gm');
  const hpccWasm = window["@hpcc-js/wasm"];
  const getIdFromHash = () => {
    let hash = window.location.hash;
    if (hash.indexOf('frame') == -1) {
      return 0;
    }
    return parseInt(window.location.hash.substring('#frame'.length, window.location.hash.length));
  }
  const renderCurrentFrame = () => {
    if (!window.loaded) { return; }
    const frames_list = document.getElementById('frames_list');
    const currId = getIdFromHash();
    for (let selected of frames_list.getElementsByClassName('selected')) {
        selected.classList.remove('selected');
    }
    const selected = frames_list.children[currId];
    selected.classList.add('selected');
    selected.scrollIntoView();
    const frame = frames[currId];
    const dot_ptr = frame[0];
    let dot_txt = window.dots[dot_ptr];
    const label = frame[1];
    document.getElementById('performance_note').innerText = "Rendering...";
    const results = cssregex.exec(dot_txt)
    let css_data = ''
    if (results !== null) {
        css_data = results[1].replace(/\s*data:.*\s*,/,''); 
        css_data = unescape(css_data);
        dot_txt = dot_txt.replace(cssregex, ''); 
    }
    let render_start = performance.now();
    const render_callback = svg => {
      renderCache[dot_ptr] = svg;
      var area = document.getElementById('rendered');
      area.innerHTML = `${svg}<style>${css_data}</style>`;
      var panzoom = svgPanZoom(area.children[0], {
          zoomEnabled: true, controlIconsEnabled: true, maxZoom: 200, });
      var to_highlight = frame[2].length ?
        document.querySelector(`${frame[2]}`) : null;
      if (to_highlight) {
        to_highlight.style.setProperty('fill', 'red');
      }
      document.getElementById('performance_note').innerText =
        `Rendering took ${(performance.now() - render_start).toFixed(2)}ms`;
      let text_nodes = document.getElementsByTagName("text");
      for (var el of text_nodes) {
        if (title_to_id.has(el.innerHTML)) {
          el.style.cursor = "pointer";
        }
      }
    };
    if (renderCache[dot_ptr]) {
      render_callback(renderCache[dot_ptr]);
    } else {
      hpccWasm.graphviz.layout(dot_txt, "svg", "dot").then(render_callback);
    }
  };
  const update = (delta) => {
    let currId = getIdFromHash();
    currId = (currId + delta + frames.length) % frames.length;
    window.location.hash = `#frame${currId}`
  };
  const renderFrameList = () => {
    const currId = getIdFromHash();
    const frames_list = document.getElementById('frames_list');
    for (let i=0; i<frames.length; i++) {
      const f = frames[i];
      let frame_descr = f[1];
      const rendered = document.createElement("li");
      if (frame_descr == "") {
        frame_descr = "Unnamed state";
      }
      rendered.innerHTML = `<a href="#frame${i}">${frame_descr}</a>`;
      if (i == currId) {
        rendered.classList.add('selected');
      }
      frames_list.appendChild(rendered);
    }
  };
  const decompress = async function(compressed) {
    const ds = new DecompressionStream('gzip');
    const in_fetch = await fetch(`data:application/octet-stream;base64,${compressed}`);
    const in_blob = await in_fetch.blob();
    const out_stream = in_blob.stream().pipeThrough(ds);
    const out_blob = await new Response(out_stream).blob();
    return await out_blob.text();
  }
  const dots_compressed = "$DOTS";
  const frames = [$FRAMES];
  let loaded = false;
  window.addEventListener('hashchange', () => {
    renderCurrentFrame();
  });
  window.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) {
      return;
    }
    if (event.key == "j") {
      update(1);
    } else if (event.key == "k") {
      update(-1);
    } else {
      return;
    }
    event.preventDefault();
  }, true);
  document.addEventListener("DOMContentLoaded", () => {
    decompress(dots_compressed).then(text => {
      window.dots = JSON.parse(text);
      window.loaded = true;
      renderFrameList();
      renderCurrentFrame();
    });
    window.title_to_id = new Map();
    for (let i=0; i < frames.length; i++) {
       title_to_id.set(frames[i][1], i);
     }
    document.addEventListener("click", (event) => {
      let txt = event.target.innerHTML;
      if (title_to_id.has(txt)) {
        let id = title_to_id.get(txt);
        window.location.hash = `#frame${id}`;
      }
    });
  });
  </script>
  </body>
</html>
  )wrapper",
      {{"$DOTS", dot_graphs_compressed},
       {"$FRAMES", frames},
       {"$TITLE", graph_title}});
}
static std::string GraphTitle(const HloComputation& computation) {
  return absl::StrCat(computation.parent()->name(), "_", computation.name());
}
absl::StatusOr<std::string> WrapFusionExplorer(
    const HloComputation& computation) {
  absl::MutexLock lock(&fusion_visualizer_state_mu);
  const FusionVisualizerProgress& visualizer_progress =
      fusion_visualizer_states[FusionVisualizerStateKey(computation)];
  return WrapFusionExplorer(visualizer_progress, GraphTitle(computation));
}
static absl::StatusOr<std::string> WrapDotInHtml(absl::string_view dot,
                                                 absl::string_view title) {
  FusionVisualizerProgress progress;
  progress.AddState(dot, title, std::nullopt);
  return WrapFusionExplorer(progress, title);
}
static absl::StatusOr<std::string> WrapDotInFormat(
    const HloComputation& computation, absl::string_view dot,
    RenderedGraphFormat format) ABSL_EXCLUSIVE_LOCKS_REQUIRED(url_renderer_mu) {
  switch (format) {
    case RenderedGraphFormat::kUrl:
      CHECK(url_renderer != nullptr)
          << "Should have checked url_renderer != null before calling.";
      return (*url_renderer)(dot);
    case RenderedGraphFormat::kHtml:
      return WrapDotInHtml(dot, GraphTitle(computation));
    case RenderedGraphFormat::kDot:
      return std::string(dot);
  }
}
void RegisterGraphToURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view)> renderer) {
  absl::MutexLock lock(&url_renderer_mu);
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer. Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer =
      new std::function<absl::StatusOr<std::string>(absl::string_view)>(
          std::move(renderer));
}
void RegisterFusionState(const HloComputation& computation,
                         absl::string_view label,
                         const HloInstruction& consumer,
                         const HloInstruction* producer) {
  absl::MutexLock lock(&fusion_visualizer_state_mu);
  FusionVisualizerProgress& fusion_progress =
      fusion_visualizer_states[FusionVisualizerStateKey(computation)];
  static constexpr int kRenderRadius = 4;
  absl::flat_hash_set<const HloInstruction*> render_boundary;
  for (const HloInstruction* user : consumer.users()) {
    render_boundary.insert(user);
  }
  HloDotDumper dumper(
      consumer.parent(),
      StrCat("Rendering of ", kRenderRadius, " nodes around fusion consumer"),
      consumer.GetModule()->config().debug_options(), {},
      MakeNodeRadiusAroundFilter(&consumer, kRenderRadius, render_boundary));
  std::string dot_txt = dumper.Dump();
  std::optional<std::string> producer_to_highlight;
  if (producer) {
    producer_to_highlight = dumper.CssIdForInstruction(*producer);
  }
  fusion_progress.AddState(dot_txt, label, producer_to_highlight);
}
absl::StatusOr<std::string> RenderGraph(
    const HloComputation& computation, absl::string_view label,
    const DebugOptions& debug_options, RenderedGraphFormat format,
    HloRenderOptions hlo_render_options,
    std::optional<absl::flat_hash_map<const HloInstruction*, ColorStats>>
        color_map) {
  absl::MutexLock lock(&url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return Unavailable("Can't render as URL; no URL renderer was registered.");
  }
  std::string rendered_dot =
      HloDotDumper(&computation, label, debug_options, hlo_render_options,
                   NodeFilter(), color_map)
          .Dump();
  return WrapDotInFormat(computation, rendered_dot, format);
}
absl::StatusOr<std::string> RenderAllComputationsToHtml(
    const HloModule& module) {
  FusionVisualizerProgress progress;
  std::vector<HloInstruction*> instrs =
      module.entry_computation()->MakeInstructionPostOrder();
  absl::c_reverse(instrs);
  for (const HloInstruction* instr : instrs) {
    if (absl::c_linear_search(
            std::vector<HloOpcode>{HloOpcode::kConstant,
                                   HloOpcode::kGetTupleElement},
            instr->opcode())) {
      continue;
    }
    HloRenderOptions opts;
    opts.show_fusion_subcomputations = true;
    opts.show_backend_config = true;
    opts.show_while_subcomputations = instr->opcode() == HloOpcode::kWhile;
    static constexpr int64_t max_nodes_to_render = 100;
    absl::flat_hash_set<const HloInstruction*> render_boundary;
    NodeFilter filter = MakeNodeRadiusAroundFilter(instr, 2, render_boundary);
    if (filter.GetNumRendered().value_or(1) > max_nodes_to_render) {
      filter = MakeNodeRadiusAroundFilter(instr, 1, render_boundary);
    }
    std::string dot =
        HloDotDumper(module.entry_computation(), instr->name(),
                     module.config().debug_options(), opts, filter)
            .Dump();
    progress.AddState(dot, instr->name(), std::nullopt);
  }
  return WrapFusionExplorer(progress, module.name());
}
absl::StatusOr<std::string> RenderNeighborhoodAround(
    const HloInstruction& node, int radius, RenderedGraphFormat format,
    HloRenderOptions hlo_render_options,
    const absl::flat_hash_set<const HloInstruction*>& boundary,
    std::optional<absl::flat_hash_map<const HloInstruction*, ColorStats>>
        color_map) {
  absl::MutexLock lock(&url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return FailedPrecondition(
        "Can't render as URL; no URL renderer was registered.");
  }
  std::string label =
      StrCat("Neighborhood of ", radius, " nodes around ", node.name());
  std::string rendered_dot =
      HloDotDumper(
          node.parent(), label, node.GetModule()->config().debug_options(),
          hlo_render_options,
          MakeNodeRadiusAroundFilter(&node, radius, boundary), color_map)
          .Dump();
  return WrapDotInFormat(*node.parent(), rendered_dot, format);
}
absl::StatusOr<std::string> RenderAllPathsFromTo(
    const HloInstruction& from, const HloInstruction& to, int64_t max_nodes,
    RenderedGraphFormat format, HloRenderOptions hlo_render_options) {
  absl::MutexLock lock(&url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return FailedPrecondition(
        "Can't render as URL; no URL renderer was registered.");
  }
  CHECK_EQ(from.parent(), to.parent()) << "Nodes must be in same computation!";
  auto debug_options = from.GetModule()->config().debug_options();
  bool hit_limit = false;
  NodeFilter filter = MakeNodeFromToFilter(&from, &to, max_nodes, &hit_limit);
  std::string label;
  if (!hit_limit) {
    label = StrCat("All paths from ", from.name(), " to ", to.name());
  } else {
    label = StrCat(max_nodes, " nodes on the shortest paths from ", from.name(),
                   " to ", to.name(),
                   "<br/><br/>***SHOWING ONLY A SUBSET OF ALL PATHS BETWEEN "
                   "NODES***<br/><br/>");
  }
  std::string rendered_dot = HloDotDumper(from.parent(), label, debug_options,
                                          hlo_render_options, filter)
                                 .Dump();
  return WrapDotInFormat(*from.parent(), rendered_dot, format);
}
}  