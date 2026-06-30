#include "xla/service/hlo_rematerialization.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
using ::tsl::strings::HumanReadableNumBytes;
bool IsRematerializable(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kCopy) {
    if (LayoutUtil::Equal(instruction->shape().layout(),
                          instruction->operand(0)->shape().layout())) {
      return false;
    }
  }
  if (auto collective = DynCast<HloCollectiveInstruction>(instruction)) {
    return !collective->constrain_layout();
  }
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConstant:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
      return false;
    default:
      return !instruction->HasSideEffect();
  }
}
bool CanBeRematerialized(
    const HloInstruction* instruction,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  auto it = rematerializable_map->find(instruction);
  if (it != rematerializable_map->end()) {
    return it->second;
  }
  bool rematerializable = IsRematerializable(instruction);
  (*rematerializable_map)[instruction] = rematerializable;
  return rematerializable;
}
bool IsSupportedIndirectUser(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kGetTupleElement;
}
using BufferId = int64_t;
using BufferIdList = absl::InlinedVector<BufferId, 3>;
struct RematStrategy {
  enum {
    kRecompute,
    kCompress,
    kHostOffload,
  } kind;
  Shape compact_shape;
};
struct Item {
  HloInstruction* instruction;
  bool placed = false;
  bool denylisted = false;
  BufferIdList buffers_defined;
  BufferIdList buffers_output;
  BufferIdList buffers_used;
  bool is_skip_node = false;
 private:
  friend class InstructionList;
  Item* next = nullptr;
  Item* prev = nullptr;
  Item* prev_skip_node = nullptr;
  Item* next_skip_node = nullptr;
  int64_t position;
};
struct ItemUse {
  Item* user;
  int64_t operand_number;
  std::optional<int64_t> index;
  ItemUse(Item* user, int64_t op_num, std::optional<int64_t> index)
      : user(user), operand_number(op_num), index(index) {}
  bool operator==(const ItemUse& other) const {
    return user == other.user && operand_number == other.operand_number &&
           index == other.index;
  }
};
using ItemList = absl::InlinedVector<Item*, 3>;
using UsesList = absl::InlinedVector<ItemUse, 3>;
class InstructionList {
 public:
  explicit InstructionList(const HloInstructionSequence& order) {
    int64_t position = 0;
    Item* last = nullptr;
    last_skip_node_ = nullptr;
    first_skip_node_ = nullptr;
    for (HloInstruction* inst : order.instructions()) {
      Item* item = new Item;
      item->next = nullptr;
      item->prev = last;
      if (last == nullptr) {
        first_ = item;
      } else {
        last->next = item;
      }
      last = item;
      item->instruction = inst;
      item->position = position;
      position++;
      item_map_[inst] = item;
    }
  }
  ~InstructionList() {
    for (Item* item = first_; item != nullptr;) {
      Item* next = item->next;
      delete item;
      item = next;
    }
  }
  size_t size() const { return item_map_.size(); }
  Item* first() const { return first_; }
  Item* next(Item* item) const { return item->next; }
  const Item* next(const Item* item) const { return item->next; }
  Item* prev(Item* item) const { return item->prev; }
  const Item* prev(const Item* item) const { return item->prev; }
  Item* first_skip_node() const { return first_skip_node_; }
  Item* next_skip_node(Item* item) const { return item->next_skip_node; }
  Item* CreateItem(HloInstruction* inst) {
    Item* item = new Item;
    item->instruction = inst;
    CHECK(item_map_.insert({inst, item}).second)
        << "inserting inst twice " << inst->name();
    return item;
  }
  Item* GetItem(const HloInstruction* inst) const {
    auto iter = item_map_.find(inst);
    CHECK(iter != item_map_.end()) << "Did not find " << inst->name();
    return iter->second;
  }
  void InsertBeforeInstructions(Item* to_insert,
                                absl::Span<Item* const> before_instructions) {
    VLOG(3) << "InsertBeforeInstructions: " << to_insert->instruction->name()
            << " before {"
            << absl::StrJoin(before_instructions, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << "}";
    CHECK(!before_instructions.empty());
    Item* min_position_item = nullptr;
    for (Item* item : before_instructions) {
      if (min_position_item == nullptr ||
          item->position < min_position_item->position) {
        min_position_item = item;
      }
    }
    while (min_position_item->prev != nullptr &&
           min_position_item->position == min_position_item->prev->position) {
      min_position_item = min_position_item->prev;
    }
    while (!absl::c_linear_search(before_instructions, min_position_item)) {
      min_position_item = min_position_item->next;
    }
    return InsertBefore(to_insert, min_position_item);
  }
  void PromoteNodesToSkip(absl::FunctionRef<bool(Item*)> should_promote) {
    int64_t count = 0;
    for (auto* item = first(); item != nullptr; item = next(item)) {
      if (should_promote(item)) {
        count += 1;
        if (first_skip_node_ == nullptr) {
          first_skip_node_ = item;
        }
        item->is_skip_node = true;
        item->prev_skip_node = last_skip_node_;
        if (last_skip_node_ != nullptr) {
          last_skip_node_->next_skip_node = item;
        }
        last_skip_node_ = item;
      }
    }
    VLOG(1) << " Rematerialization has " << count << " items in express lane";
  }
  void InsertAfterInstructions(Item* to_insert,
                               absl::Span<Item* const> after_instructions) {
    VLOG(3) << "InsertAfterInstructions: " << to_insert->instruction->name()
            << " after {"
            << absl::StrJoin(after_instructions, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << "}";
    CHECK(!after_instructions.empty());
    Item* max_position_item = nullptr;
    for (Item* item : after_instructions) {
      if (max_position_item == nullptr ||
          item->position > max_position_item->position) {
        max_position_item = item;
      }
    }
    CHECK(max_position_item->next != nullptr);
    InsertBeforeInstructions(to_insert, {max_position_item->next});
  }
  void Denylist(const HloInstruction* inst) {
    GetItem(inst)->denylisted = true;
  }
 private:
  void InsertBefore(Item* item, Item* before) {
    VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
            << before->instruction->name();
    item->is_skip_node = true;
    Item* cursor = before;
    while (cursor != nullptr && !cursor->is_skip_node) {
      cursor = cursor->next;
    }
    CHECK(cursor == nullptr || cursor->is_skip_node);
    if (cursor == nullptr) {
      item->prev_skip_node = last_skip_node_;
      item->next_skip_node = nullptr;
      last_skip_node_ = item;
    } else {
      CHECK(cursor->is_skip_node);
      item->prev_skip_node = cursor->prev_skip_node;
      if (item->prev_skip_node != nullptr) {
        item->prev_skip_node->next_skip_node = item;
      }
      item->next_skip_node = cursor;
      cursor->prev_skip_node = item;
    }
    if (first_skip_node_ == cursor) {
      first_skip_node_ = item;
    }
    item->prev = before->prev;
    item->next = before;
    before->prev = item;
    if (item->prev != nullptr) {
      item->prev->next = item;
    } else {
      first_ = item;
    }
    item->position = before->position;
  }
  Item* first_;
  Item* first_skip_node_;
  Item* last_skip_node_;
  absl::flat_hash_map<const HloInstruction*, Item*> item_map_;
};
UsesList GetUsers(const InstructionList& instruction_list,
                  const LogicalBuffer* logical_buffer,
                  const TuplePointsToAnalysis& points_to_analysis,
                  bool* has_indirect_users) {
  UsesList users;
  *has_indirect_users = false;
  for (const BufferAlias& buffer_alias :
       points_to_analysis.GetBufferAliases(*logical_buffer)) {
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      if (points_to_analysis.DoesNotUseOperandBuffer(
              buffer_alias.instruction(), buffer_alias.index(), user)) {
        continue;
      }
      if (buffer_alias.instruction() != logical_buffer->instruction() &&
          !IsSupportedIndirectUser(buffer_alias.instruction())) {
        *has_indirect_users = true;
      }
      Item* user_item = instruction_list.GetItem(user);
      std::optional<int64_t> user_index =
          logical_buffer->index().size() != 1
              ? std::nullopt
              : std::make_optional(logical_buffer->index().back());
      for (int64_t op_idx : user->OperandIndices(buffer_alias.instruction())) {
        if (!absl::c_linear_search(
                users,
                ItemUse{user_item, static_cast<int>(op_idx), user_index})) {
          users.push_back(
              ItemUse{user_item, static_cast<int>(op_idx), user_index});
        }
      }
    }
  }
  return users;
}
class MemoryUsageTracker {
 public:
  MemoryUsageTracker(const HloRematerialization::Options& options,
                     const HloComputation* computation,
                     const TuplePointsToAnalysis& points_to_analysis,
                     const InstructionList& instruction_list);
  absl::Status BeginInstruction(Item* item);
  int64_t RematerializationCost(const std::vector<Item*>& items,
                                int64_t memory_reduced,
                                int64_t memory_limit_bytes) const {
    bool zero_cost_move = true;
    for (auto* item : items) {
      auto* instruction = item->instruction;
      if (absl::c_any_of(
              instruction->users(),
              [this](const HloInstruction* inst) { return IsPlaced(inst); })) {
        zero_cost_move = false;
        break;
      }
    }
    if (zero_cost_move) {
      return 0;
    }
    CHECK_GT(memory_reduced, 0);
    return memory_limit_bytes / memory_reduced;
  }
  absl::Status EndInstruction();
  int64_t MemoryReducedIfCompressed(const Item* item,
                                    const Shape& compact_shape) const;
  int64_t MemoryReducedIfRematerialized(
      absl::Span<const Item* const> items) const;
  absl::Status AddCompressInstructions(Item* original_item,
                                       Item* compressed_item,
                                       Item* uncompressed_item);
  absl::Status AddRematerializedInstruction(Item* original_item,
                                            Item* remat_item,
                                            absl::Span<Item*> indirect_users);
  std::tuple<UsesList, UsesList> GetPlacedAndUnplacedUsers(
      const UsesList& uses) const;
 public:
  absl::Status AddHostOffloadCopyInstructions(Item* original_item,
                                              Item* copy_start_to_host_item,
                                              Item* copy_done_to_host_item,
                                              Item* copy_start_to_device_item,
                                              Item* copy_done_to_device_item);
  int64_t BytesUsedByBuffers(const Item* item,
                             bool only_count_unplaced_users) const;
  std::optional<int64_t> GetCostOfCompression(const Item* candidate_item,
                                              int64_t memory_limit_bytes,
                                              int64_t peak_memory_bytes);
  std::optional<int64_t> GetCostOfHostOffload(const Item* candidate_item,
                                              int64_t memory_limit_bytes) const;
  std::optional<int64_t> GetCostOfRecompute(
      const std::vector<Item*>& candidate_items,
      int64_t memory_limit_bytes) const;
  std::tuple<std::vector<Item*>, RematStrategy, int>
  PickRematerializationCandidates(
      const InstructionList& instruction_list, int64_t memory_limit_bytes,
      absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
      int min_block_size, int max_block_size, int64_t peak_memory_bytes);
  bool IsPlaced(const HloInstruction* instruction) const {
    return instruction_list_.GetItem(instruction)->placed;
  }
  bool HasUnplacedUsers(Item* item) const;
  UsesList GetItemUses(Item* item) const;
  bool IsInProgressItem(Item* item) const { return item == in_progress_item_; }
  int64_t memory_usage() const { return memory_usage_; }
  int64_t AllocatedSize(Item* item) const {
    int64_t size = 0;
    for (auto buffer_id : item->buffers_defined) {
      size += AllocatedSize(buffer_id);
    }
    return size;
  }
  const HloComputation* computation() const { return computation_; }
  const HloRematerialization::Options& options() const { return options_; }
  bool Check() const;
  std::string ToString() const;
 private:
  struct Buffer {
    const BufferId id;
    Item* defining_instruction;
    const int64_t size;
    Shape shape;
    bool live_out;
    bool has_indirect_uses;
    ShapeIndex index;
    UsesList users;
    int64_t unfinished_user_count;
    std::string ToString() const {
      return absl::StrCat("Buffer ", id, " (defined by ",
                          defining_instruction->instruction->name(), ", size ",
                          size, " bytes)");
    }
  };
  void CountAllocatedMemory(Item* item);
  absl::Status CountFreedMemory(Item* item);
  void ReplaceUsesInUsersOfBuffer(Buffer& buffer, BufferId old_id) const;
  absl::StatusOr<const Shape*> GetCompactShape(const HloInstruction* hlo);
  Buffer& CreateBufferFromLogicalBuffer(
      const LogicalBuffer* logical_buffer,
      const TuplePointsToAnalysis& points_to_analysis, bool live_out) {
    bool has_indirect_uses = false;
    UsesList users = GetUsers(instruction_list_, logical_buffer,
                              points_to_analysis, &has_indirect_uses);
    return NewBuffer(instruction_list_.GetItem(logical_buffer->instruction()),
                     logical_buffer->shape(), logical_buffer->index(),
                     std::move(users), live_out, has_indirect_uses);
  }
  Buffer& RematerializeBuffer(const Buffer& original_buffer, Item* remat_item,
                              UsesList&& rematerialized_uses) {
    CHECK(original_buffer.defining_instruction->placed)
        << original_buffer.defining_instruction->instruction->name();
    CHECK(!original_buffer.has_indirect_uses) << original_buffer.ToString();
    CHECK(!original_buffer.live_out) << original_buffer.ToString();
    for (ItemUse& use : rematerialized_uses) {
      CHECK(!use.user->placed) << use.user->instruction->name();
    }
    return NewBuffer(remat_item, original_buffer.shape, original_buffer.index,
                     std::move(rematerialized_uses), false,
                     false);
  }
  int64_t AllocatedSize(BufferId buffer_id) const {
    const Buffer& buffer = buffers_.at(buffer_id);
    HloInstruction* inst = buffer.defining_instruction->instruction;
    HloOpcode def_opcode = inst->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      if (options_.host_memory_offload_config && buffer.shape.has_layout() &&
          buffer.shape.layout().memory_space() ==
              options_.host_memory_offload_config->host_memory_space) {
        return 0;
      }
      return buffer.size;
    }
  }
  bool IsFinished(Item* item) const {
    return item->placed && item != in_progress_item_;
  }
  bool IsInUse(BufferId buffer_id) const {
    if (in_progress_item_ == nullptr) {
      return false;
    }
    const BufferIdList& in_progress_uses = in_progress_item_->buffers_used;
    return absl::c_linear_search(in_progress_uses, buffer_id);
  }
  bool IsCurrentlyLive(BufferId buffer_id) const {
    const Buffer& buffer = buffers_[buffer_id];
    return (buffer.defining_instruction->placed &&
            buffer.unfinished_user_count > 0);
  }
  bool IsInstructionCurrentlyLive(const Item* instruction) const {
    if (!IsPlaced(instruction->instruction)) {
      return false;
    }
    for (const HloInstruction* user : instruction->instruction->users()) {
      if (!IsPlaced(user)) {
        return true;
      }
    }
    return false;
  }
  Buffer& NewBuffer(Item* defining_instruction, const Shape& shape,
                    const ShapeIndex& index, UsesList&& uses, bool live_out,
                    bool has_indirect_uses) {
    int buffer_id = buffers_.size();
    auto get_num_of_unique_users = [](const UsesList& uses) -> int64_t {
      absl::flat_hash_set<Item*> users_set;
      for (const ItemUse& use : uses) {
        users_set.insert(use.user);
      }
      return users_set.size();
    };
    buffers_.push_back(Buffer{buffer_id, defining_instruction,
                              options_.hlo_cost_analysis.GetShapeSize(shape),
                              shape, live_out, has_indirect_uses, index, uses,
                              get_num_of_unique_users(uses)});
    return buffers_.back();
  }
  const HloRematerialization::Options& options_;
  const HloComputation* computation_;
  const InstructionList& instruction_list_;
  absl::flat_hash_map<const HloInstruction*, Shape> compact_shape_;
  int64_t memory_usage_ = 0;
  Item* in_progress_item_ = nullptr;
  std::vector<Buffer> buffers_;
};
MemoryUsageTracker::MemoryUsageTracker(
    const HloRematerialization::Options& options,
    const HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const InstructionList& instruction_list)
    : options_(options),
      computation_(computation),
      instruction_list_(instruction_list) {
  PointsToSet::BufferSet live_out_set =
      points_to_analysis.GetPointsToSet(computation_->root_instruction())
          .CreateFlattenedSet();
  absl::flat_hash_map<const LogicalBuffer*, BufferId>
      logical_buffer_to_buffer_id;
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* const instruction = item->instruction;
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
      Buffer* buffer;
      if (instruction->opcode() == HloOpcode::kWhile) {
        const PointsToSet& operand_points_to =
            points_to_analysis.GetPointsToSet(instruction->operand(0));
        CHECK_EQ(operand_points_to.element(logical_buffer->index()).size(), 1);
        const LogicalBuffer* source_logical_buffer =
            operand_points_to.element(logical_buffer->index())[0];
        buffer =
            &buffers_.at(logical_buffer_to_buffer_id.at(source_logical_buffer));
        buffer->has_indirect_uses = true;
        buffer->live_out =
            buffer->live_out || ContainsKey(live_out_set, logical_buffer);
        bool unused;
        for (ItemUse& user_item : GetUsers(instruction_list_, logical_buffer,
                                           points_to_analysis, &unused)) {
          auto existing_user_it = absl::c_find_if(
              buffer->users,
              [&](const ItemUse& use) { return user_item.user == use.user; });
          if (existing_user_it == buffer->users.end()) {
            buffer->unfinished_user_count++;
            user_item.user->buffers_used.push_back(buffer->id);
            buffer->users.push_back(user_item);
          }
        }
      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis,
            ContainsKey(live_out_set, logical_buffer));
        item->buffers_defined.push_back(buffer->id);
        for (ItemUse& user : buffer->users) {
          if (!absl::c_linear_search(user.user->buffers_used, buffer->id)) {
            user.user->buffers_used.push_back(buffer->id);
          }
        }
      }
      logical_buffer_to_buffer_id[logical_buffer] = buffer->id;
    }
    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetPointsToSet(instruction).CreateFlattenedSet()) {
      item->buffers_output.push_back(
          logical_buffer_to_buffer_id[logical_buffer]);
    }
  }
  XLA_VLOG_LINES(10, ToString());
  DCHECK(Check());
}
void MemoryUsageTracker::CountAllocatedMemory(Item* item) {
  for (BufferId buffer_id : item->buffers_defined) {
    VLOG(3) << "  Buffer " << buffers_.at(buffer_id).ToString()
            << " is now live.";
    memory_usage_ += AllocatedSize(buffer_id);
  }
}
absl::Status MemoryUsageTracker::CountFreedMemory(Item* item) {
  for (BufferId buffer_id : item->buffers_used) {
    Buffer& buffer = buffers_.at(buffer_id);
    buffer.unfinished_user_count--;
    TF_RET_CHECK(buffer.unfinished_user_count >= 0)
        << buffer.ToString() << " has negative unfinished user count.";
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is now dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
    }
  }
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      VLOG(3) << "  " << buffer.ToString() << " is immediately dead.";
      memory_usage_ -= AllocatedSize(buffer_id);
    }
  }
  return absl::OkStatus();
}
absl::Status MemoryUsageTracker::BeginInstruction(Item* item) {
  const HloInstruction* instruction = item->instruction;
  VLOG(3) << "BeginInstruction " << instruction->name();
  TF_RET_CHECK(in_progress_item_ == nullptr);
  in_progress_item_ = item;
  item->placed = true;
  CountAllocatedMemory(item);
  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();
  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}
absl::Status MemoryUsageTracker::EndInstruction() {
  TF_RET_CHECK(in_progress_item_ != nullptr);
  VLOG(3) << "EndInstruction " << in_progress_item_->instruction->name();
  TF_RETURN_IF_ERROR(CountFreedMemory(in_progress_item_));
  in_progress_item_ = nullptr;
  VLOG(3) << "  memory usage = " << memory_usage_;
  VLOG(10) << ToString();
  if (VLOG_IS_ON(1)) {
    DCHECK(Check());
  }
  return absl::OkStatus();
}
int64_t MemoryUsageTracker::MemoryReducedIfCompressed(
    const Item* item, const Shape& compact_shape) const {
  CHECK_NE(in_progress_item_, nullptr);
  if (!item->placed || item == in_progress_item_) {
    return 0;
  }
  int64_t memory_reduced = 0;
  CHECK_EQ(item->buffers_output.size(), 1);
  BufferId buffer_id = item->buffers_output[0];
  if (IsCurrentlyLive(buffer_id) && !IsInUse(buffer_id) &&
      IsInstructionCurrentlyLive(item)) {
    const Buffer& buffer = buffers_.at(buffer_id);
    memory_reduced += buffer.size;
    int64_t compact_shape_size =
        options_.hlo_cost_analysis.GetShapeSize(compact_shape);
    memory_reduced -= compact_shape_size;
  }
  return memory_reduced;
}
int64_t MemoryUsageTracker::MemoryReducedIfRematerialized(
    absl::Span<const Item* const> items) const {
  CHECK_NE(in_progress_item_, nullptr);
  int64_t memory_reduced = 0;
  absl::flat_hash_set<const Item*> remat_candidates;
  for (const Item* item : items) {
    if (!item->placed || item == in_progress_item_) {
      LOG(WARNING) << "Unplaced item or in progress item being checked for "
                      "rematerialization.";
      return 0;
    }
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffers_.at(buffer_id);
      if (buffer.has_indirect_uses || buffer.live_out ||
          buffer.index.size() > 1) {
        return 0;
      }
      if (IsInUse(buffer_id)) {
        return 0;
      }
      if (IsCurrentlyLive(buffer_id)) {
        memory_reduced += AllocatedSize(buffer_id);
      }
    }
    for (BufferId buffer_id : item->buffers_used) {
      if (!IsCurrentlyLive(buffer_id)) {
        Item* defining_instruction =
            buffers_.at(buffer_id).defining_instruction;
        if (!remat_candidates.contains(defining_instruction)) {
          memory_reduced -= AllocatedSize(buffer_id);
        }
      }
    }
    remat_candidates.insert(item);
  }
  return memory_reduced;
}
std::tuple<UsesList, UsesList> MemoryUsageTracker::GetPlacedAndUnplacedUsers(
    const UsesList& uses) const {
  UsesList placed_users, unplaced_users;
  for (const ItemUse& use : uses) {
    if (use.user->placed) {
      DCHECK(IsFinished(use.user)) << use.user->instruction->name();
      placed_users.push_back(use);
    } else {
      unplaced_users.push_back(use);
    }
  }
  return {placed_users, unplaced_users};
}
void MemoryUsageTracker::ReplaceUsesInUsersOfBuffer(Buffer& buffer,
                                                    BufferId old_id) const {
  for (ItemUse& use : buffer.users) {
    BufferIdList& buffers_used = use.user->buffers_used;
    absl::c_replace(buffers_used, old_id, buffer.id);
  }
}
absl::Status MemoryUsageTracker::AddCompressInstructions(
    Item* original_item, Item* compressed_item, Item* uncompressed_item) {
  CHECK(original_item->placed)
      << "Compressing instruction, but the original is not yet placed.";
  CHECK_EQ(original_item->buffers_output.size(), 1)
      << "Only compressing items which have a single output buffer";
  memory_usage_ -= options_.hlo_cost_analysis.GetShapeSize(
      original_item->instruction->shape());
  memory_usage_ += options_.hlo_cost_analysis.GetShapeSize(
      compressed_item->instruction->shape());
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);
  original_buffer.users = std::move(placed_users);
  original_buffer.unfinished_user_count = 0;
  original_buffer.users.push_back(ItemUse{compressed_item, 0, std::nullopt});
  ShapeIndex copied_index = original_buffer.index;
  Buffer& compressed_buffer =
      NewBuffer(compressed_item, compressed_item->instruction->shape(),
                copied_index, {ItemUse{uncompressed_item, 0, std::nullopt}},
                false,
                false);
  compressed_item->buffers_used = original_item->buffers_output;
  compressed_item->buffers_output = {compressed_buffer.id};
  compressed_item->buffers_defined.push_back(compressed_buffer.id);
  Buffer& uncompressed_buffer =
      NewBuffer(uncompressed_item, uncompressed_item->instruction->shape(),
                copied_index, std::move(unplaced_users), false,
                false);
  uncompressed_item->buffers_used = {compressed_item->buffers_output[0]};
  uncompressed_item->buffers_output = {uncompressed_buffer.id};
  uncompressed_item->buffers_defined = {uncompressed_buffer.id};
  ReplaceUsesInUsersOfBuffer(uncompressed_buffer, original_buffer_id);
  return absl::OkStatus();
}
absl::Status MemoryUsageTracker::AddRematerializedInstruction(
    Item* original_item, Item* remat_item, absl::Span<Item*> indirect_users) {
  VLOG(3) << "AddRematerializedInstruction: original_instruction = "
          << original_item->instruction->name()
          << ", remat_instruction = " << remat_item->instruction->name();
  TF_RET_CHECK(in_progress_item_ != nullptr);
  TF_RET_CHECK(original_item->placed) << original_item->instruction->name();
  TF_RET_CHECK(!remat_item->placed) << remat_item->instruction->name();
  remat_item->buffers_used = original_item->buffers_used;
  for (BufferId buffer_id : original_item->buffers_used) {
    Buffer& buffer = buffers_.at(buffer_id);
    if (buffer.unfinished_user_count == 0) {
      memory_usage_ += AllocatedSize(buffer.id);
    }
    buffer.unfinished_user_count++;
    absl::InlinedVector<ItemUse, 2> filtered_users;
    std::copy_if(buffer.users.begin(), buffer.users.end(),
                 std::back_inserter(filtered_users),
                 [&](const ItemUse& iu) { return iu.user == original_item; });
    for (ItemUse& u : filtered_users) {
      buffer.users.push_back(ItemUse{remat_item, u.operand_number, u.index});
    }
  }
  const absl::flat_hash_set<Item*> indirect_users_set(indirect_users.begin(),
                                                      indirect_users.end());
  for (BufferId old_buffer_id : original_item->buffers_defined) {
    Buffer& old_buffer = buffers_.at(old_buffer_id);
    UsesList placed_users;
    UsesList unplaced_users;
    for (ItemUse& user : old_buffer.users) {
      if (user.user->placed) {
        placed_users.push_back(user);
      } else {
        if (!IsSupportedIndirectUser(user.user->instruction) ||
            indirect_users_set.contains(user.user)) {
          unplaced_users.push_back(user);
        } else {
          CHECK(user.user->buffers_defined.empty())
              << "Buffers defined expected to be empty for use passthrough "
                 "instructions";
          user.user->buffers_output.clear();
          user.user->buffers_used.clear();
        }
      }
    }
    old_buffer.users = std::move(placed_users);
    old_buffer.unfinished_user_count = 0;
    memory_usage_ -= AllocatedSize(old_buffer.id);
    Buffer& new_buffer =
        RematerializeBuffer(old_buffer, remat_item, std::move(unplaced_users));
    remat_item->buffers_defined.push_back(new_buffer.id);
    remat_item->buffers_output.push_back(new_buffer.id);
    auto update_buffers = [old_buffer_id, new_buffer_id = new_buffer.id](
                              BufferIdList& to_update) {
      std::replace(to_update.begin(), to_update.end(), old_buffer_id,
                   new_buffer_id);
    };
    for (ItemUse& user : new_buffer.users) {
      update_buffers(user.user->buffers_used);
      update_buffers(user.user->buffers_output);
    }
  }
  for (Item* indirect_user : indirect_users) {
    const Item* source_item =
        instruction_list_.GetItem(indirect_user->instruction->operand(0));
    switch (indirect_user->instruction->opcode()) {
      case HloOpcode::kBitcast: {
        if (IsSupportedIndirectUser(source_item->instruction)) {
          indirect_user->buffers_used = source_item->buffers_output;
          indirect_user->buffers_output = source_item->buffers_output;
        } else {
          indirect_user->buffers_used = source_item->buffers_defined;
          indirect_user->buffers_output = source_item->buffers_defined;
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(indirect_user->instruction);
        for (BufferId buffer_id : source_item->buffers_defined) {
          const Buffer& def_buffer = buffers_.at(buffer_id);
          if (def_buffer.index == ShapeIndex{gte->tuple_index()}) {
            indirect_user->buffers_output.push_back(buffer_id);
          }
          if (def_buffer.index.empty()) {
            indirect_user->buffers_used.push_back(buffer_id);
          }
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported indirect instruction with opcode "
                   << indirect_user->instruction->opcode();
        break;
      }
    }
    for (BufferId buffer_id : indirect_user->buffers_used) {
      Buffer& buffer = buffers_.at(buffer_id);
      buffer.unfinished_user_count++;
      buffer.users.push_back(ItemUse{indirect_user, 0, std::nullopt});
    }
  }
  VLOG(3) << "  memory usage = " << memory_usage_;
  XLA_VLOG_LINES(10, ToString());
  DCHECK(Check());
  return absl::OkStatus();
}
absl::Status MemoryUsageTracker::AddHostOffloadCopyInstructions(
    Item* original_item, Item* copy_start_to_host_item,
    Item* copy_done_to_host_item, Item* copy_start_to_device_item,
    Item* copy_done_to_device_item) {
  CHECK_EQ(original_item->buffers_defined.size(), 1);
  CHECK_EQ(original_item->buffers_output.size(), 1);
  BufferId original_buffer_id = original_item->buffers_output[0];
  Buffer& original_buffer = buffers_.at(original_buffer_id);
  auto [placed_users, unplaced_users] =
      GetPlacedAndUnplacedUsers(original_buffer.users);
  original_buffer.users = std::move(placed_users);
  original_buffer.users.emplace_back(copy_start_to_host_item, 0, std::nullopt);
  original_buffer.unfinished_user_count = 1;
  CHECK_EQ(copy_start_to_host_item->instruction->shape().tuple_shapes_size(), 3)
      << "copy_start_to_host_item's shape is "
      << copy_start_to_host_item->instruction->shape().ToString();
  CHECK_EQ(copy_start_to_device_item->instruction->shape().tuple_shapes_size(),
           3)
      << "copy_start_to_device_item's shape is "
      << copy_start_to_device_item->instruction->shape().ToString();
  BufferId copy_start_to_host_device_buffer_id =
      NewBuffer(copy_start_to_host_item,
                copy_start_to_host_item->instruction->shape().tuple_shapes(1),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_host_item, 0, std::nullopt}},
                false, false)
          .id;
  BufferId copy_start_to_host_context_buffer_id =
      NewBuffer(copy_start_to_host_item,
                copy_start_to_host_item->instruction->shape().tuple_shapes(2),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_host_item, 0, std::nullopt}},
                false, false)
          .id;
  BufferId copy_start_to_device_device_buffer_id =
      NewBuffer(copy_start_to_device_item,
                copy_start_to_device_item->instruction->shape().tuple_shapes(0),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_device_item, 0, std::nullopt}},
                false, false)
          .id;
  BufferId copy_start_to_device_context_buffer_id =
      NewBuffer(copy_start_to_device_item,
                copy_start_to_device_item->instruction->shape().tuple_shapes(2),
                ShapeIndex(),
                UsesList{ItemUse{copy_done_to_device_item, 0, std::nullopt}},
                false, false)
          .id;
  BufferId copy_done_to_device_buffer_id =
      NewBuffer(copy_done_to_device_item,
                copy_done_to_device_item->instruction->shape(), ShapeIndex(),
                std::move(unplaced_users), false,
                false)
          .id;
  copy_start_to_host_item->buffers_used = original_item->buffers_output;
  copy_start_to_host_item->buffers_output = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};
  copy_start_to_host_item->buffers_defined = {
      copy_start_to_host_device_buffer_id,
      copy_start_to_host_context_buffer_id};
  copy_done_to_host_item->buffers_used =
      copy_start_to_host_item->buffers_output;
  copy_done_to_host_item->buffers_output = {};
  copy_done_to_host_item->buffers_defined = {};
  copy_start_to_device_item->buffers_used =
      copy_done_to_host_item->buffers_output;
  copy_start_to_device_item->buffers_output = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};
  copy_start_to_device_item->buffers_defined = {
      copy_start_to_device_device_buffer_id,
      copy_start_to_device_context_buffer_id};
  copy_done_to_device_item->buffers_used =
      copy_start_to_device_item->buffers_output;
  copy_done_to_device_item->buffers_output = {copy_done_to_device_buffer_id};
  copy_done_to_device_item->buffers_defined = {copy_done_to_device_buffer_id};
  Buffer& copy_done_to_device_buffer =
      buffers_.at(copy_done_to_device_buffer_id);
  ReplaceUsesInUsersOfBuffer(copy_done_to_device_buffer, original_buffer_id);
  if (copy_start_to_host_item->placed) {
    CountAllocatedMemory(copy_start_to_host_item);
    TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_host_item));
    if (copy_done_to_host_item->placed) {
      CountAllocatedMemory(copy_done_to_host_item);
      TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_host_item));
      if (copy_start_to_device_item->placed) {
        CountAllocatedMemory(copy_start_to_device_item);
        TF_RETURN_IF_ERROR(CountFreedMemory(copy_start_to_device_item));
        if (copy_done_to_device_item->placed) {
          CountAllocatedMemory(copy_done_to_device_item);
          TF_RETURN_IF_ERROR(CountFreedMemory(copy_done_to_device_item));
        }
      }
    }
  }
  return absl::OkStatus();
}
std::string MemoryUsageTracker::ToString() const {
  std::string output =
      absl::StrCat("MemoryUsageTracker for ", computation_->name(), "\n");
  absl::StrAppend(&output,
                  "Memory usage: ", HumanReadableNumBytes(memory_usage()), " (",
                  memory_usage(), " bytes)");
  for (auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    const HloInstruction* instruction = item->instruction;
    absl::string_view inprogress =
        item == in_progress_item_ ? " in-progress" : "";
    absl::string_view placed = item->placed ? " placed" : "";
    absl::StrAppend(&output, "  ", instruction->name(), inprogress, placed,
                    "\n    Defines:\n");
    for (BufferId buffer_id : item->buffers_defined) {
      const Buffer& buffer = buffers_[buffer_id];
      absl::string_view live = IsCurrentlyLive(buffer_id) ? " live" : "";
      absl::StrAppend(&output, "      ", buffer.ToString(), live, ", ",
                      buffer.unfinished_user_count, " unfinished uses\n");
    }
    absl::StrAppend(&output, "    Outputs:\n");
    for (BufferId buffer_id : item->buffers_output) {
      absl::StrAppend(&output, "      ", buffers_[buffer_id].ToString(), "\n");
    }
    absl::StrAppend(&output, "    Uses:\n");
    for (BufferId buffer_id : item->buffers_used) {
      absl::StrAppend(&output, "      ", buffers_[buffer_id].ToString(), "\n");
    }
  }
  return output;
}
absl::StatusOr<const Shape*> MemoryUsageTracker::GetCompactShape(
    const HloInstruction* hlo) {
  auto it = compact_shape_.find(hlo);
  if (it != compact_shape_.end()) {
    return &it->second;
  }
  const Shape& original_shape = hlo->shape();
  TF_ASSIGN_OR_RETURN(Shape min_shape,
                      options_.compact_shape_function(original_shape));
  return &compact_shape_.emplace(hlo, min_shape).first->second;
}
bool MemoryUsageTracker::Check() const {
  auto elements_are_unique = [](const BufferIdList& vec) {
    return vec.size() == std::set<BufferId>(vec.begin(), vec.end()).size();
  };
  for (auto* instruction : computation_->instructions()) {
    const BufferIdList& defined_buffers =
        instruction_list_.GetItem(instruction)->buffers_defined;
    CHECK(elements_are_unique(defined_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique defined buffers: "
        << absl::StrJoin(defined_buffers, ", ",
                         [this](std::string* out, BufferId buffer_id) {
                           absl::StrAppend(out,
                                           buffers_.at(buffer_id).ToString());
                         });
    for (const Buffer& buffer : buffers_) {
      if (buffer.defining_instruction->instruction == instruction) {
        CHECK(absl::c_linear_search(defined_buffers, buffer.id))
            << "Instruction " << instruction->name()
            << " defined buffers is missing: " << buffer.ToString();
      }
    }
  }
  for (auto* instruction : computation_->instructions()) {
    const BufferIdList& used_buffers =
        instruction_list_.GetItem(instruction)->buffers_used;
    CHECK(elements_are_unique(used_buffers))
        << "Instruction " << instruction->name()
        << " does not have unique used buffers: "
        << absl::StrJoin(used_buffers, ", ",
                         [this](std::string* out, BufferId buffer_id) {
                           absl::StrAppend(out,
                                           buffers_.at(buffer_id).ToString());
                         });
  }
  for (const Buffer& buffer : buffers_) {
    int64_t unfinished_uses = 0;
    absl::flat_hash_set<Item*> already_counted_user;
    for (const ItemUse& user : buffer.users) {
      const BufferIdList& used_buffers = user.user->buffers_used;
      CHECK(absl::c_linear_search(used_buffers, buffer.id))
          << "Instruction " << user.user->instruction->name()
          << " used buffers is missing " << buffer.ToString();
      if (!IsFinished(user.user) &&
          already_counted_user.insert(user.user).second) {
        unfinished_uses++;
      }
    }
    CHECK_EQ(buffer.unfinished_user_count, unfinished_uses)
        << "Incorrect unplaced use count for " << buffer.ToString();
  }
  return true;
}
std::vector<Item*> GetInitialBlock(const InstructionList& instruction_list,
                                   const MemoryUsageTracker& tracker,
                                   Item* start_item, int min_block_size) {
  std::vector<Item*> item_block;
  Item* curr_item = start_item;
  for (int i = 0; i < min_block_size; ++i) {
    if (curr_item == nullptr || !curr_item->placed ||
        tracker.IsInProgressItem(curr_item)) {
      break;
    }
    item_block.push_back(curr_item);
    curr_item = instruction_list.next(curr_item);
  }
  return item_block;
}
bool AnyDenylistedOrNonRematerializable(
    const std::vector<Item*>& block,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map) {
  for (auto* item : block) {
    if (item->denylisted) {
      return true;
    }
    if (!CanBeRematerialized(item->instruction, rematerializable_map)) {
      return true;
    }
  }
  return false;
}
int64_t MemoryUsageTracker::BytesUsedByBuffers(
    const Item* item, bool only_count_unplaced_users) const {
  int64_t bytes_used_by_buffers = 0;
  for (const auto& buffer_id : item->buffers_defined) {
    VLOG(3) << "  buffer " << buffer_id << "'s users are "
            << absl::StrJoin(buffers_.at(buffer_id).users, ", ",
                             [](std::string* str, const auto& use) {
                               str->append(use.user->instruction->name());
                             });
    for (const auto& use : buffers_.at(buffer_id).users) {
      if (!only_count_unplaced_users || !use.user->placed) {
        bytes_used_by_buffers += AllocatedSize(buffer_id);
        break;
      }
    }
  }
  return bytes_used_by_buffers;
}
std::optional<int64_t> MemoryUsageTracker::GetCostOfCompression(
    const Item* candidate_item, int64_t memory_limit_bytes,
    int64_t peak_memory_bytes) {
  CHECK(candidate_item != nullptr);
  if (candidate_item->buffers_output.size() != 1) {
    HloInstruction* candidate_instruction = candidate_item->instruction;
    VLOG(2) << "  " << candidate_instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return {};
  }
  const Buffer& output_buffer = buffers_.at(candidate_item->buffers_output[0]);
  if (!candidate_item->placed || candidate_item == in_progress_item_ ||
      output_buffer.live_out) {
    return {};
  }
  const Shape& original_shape = candidate_item->instruction->shape();
  if (!original_shape.IsArray()) {
    return {};
  }
  const Shape* compact_shape =
      GetCompactShape(candidate_item->instruction).value();
  const int64_t memory_reduced =
      MemoryReducedIfCompressed(candidate_item, *compact_shape);
  const int64_t size = options_.hlo_cost_analysis.GetShapeSize(
      candidate_item->instruction->shape());
  const int64_t reduced_size =
      options_.hlo_cost_analysis.GetShapeSize(*compact_shape);
  if (memory_reduced > 0 && size + reduced_size < peak_memory_bytes) {
    return memory_limit_bytes / memory_reduced;
  } else {
    return {};
  }
}
std::optional<int64_t> MemoryUsageTracker::GetCostOfHostOffload(
    const Item* candidate_item, int64_t memory_limit_bytes) const {
  CHECK(candidate_item != nullptr);
  HloInstruction* candidate_instruction = candidate_item->instruction;
  VLOG(2)
      << "Considering host offload as an option for remat. looking at instr "
      << candidate_instruction->name();
  if (candidate_item->buffers_output.size() != 1) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " has more than one output buffer; cannot offload to host.";
    return {};
  }
  for (auto buffer_id : candidate_item->buffers_defined) {
    for (auto use : buffers_.at(buffer_id).users) {
      if (use.user->instruction->opcode() == HloOpcode::kBitcast) {
        VLOG(3) << "  " << candidate_item->instruction->name()
                << " has a user which is a bitcast instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      } else if (use.user->instruction->opcode() == HloOpcode::kTuple) {
        VLOG(3) << "  " << candidate_item->instruction->name()
                << " has a user which is a tuple instruction("
                << use.user->instruction->name()
                << "); cannot offload "
                   "to host.";
        return {};
      }
    }
  }
  const Buffer& output_buffer = buffers_.at(candidate_item->buffers_output[0]);
  if (!candidate_item->placed || candidate_item == in_progress_item_ ||
      output_buffer.live_out) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " is not yet placed, is in progress, or is \"live_out\"; cannot "
               "offload to host.";
    return {};
  }
  const bool current_instruction_uses_this_item = [&]() {
    if (in_progress_item_ == nullptr) {
      return false;
    }
    const auto& output_buffer_ids = candidate_item->buffers_output;
    for (const auto& output_buffer_id : output_buffer_ids) {
      const Buffer& output_buffer = buffers_.at(output_buffer_id);
      for (const auto& use : output_buffer.users) {
        if (use.user == in_progress_item_) {
          return true;
        }
      }
    }
    return false;
  }();
  if (current_instruction_uses_this_item) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " is used by the current instruction in mem tracker ("
            << in_progress_item_->instruction->name()
            << "); cannot offload to host.";
    return {};
  }
  const int64_t bytes_used_by_buffers =
      BytesUsedByBuffers(candidate_item, true);
  if (bytes_used_by_buffers == 0) {
    VLOG(2) << "  " << candidate_instruction->name()
            << " consumes no memory; no point in offloading.";
    return {};
  }
  const auto [placed_uses, unplaced_uses] =
      GetPlacedAndUnplacedUsers(output_buffer.users);
  const Item* last_placed_user = nullptr;
  const Item* first_unplaced_user = nullptr;
  for (const auto* item = instruction_list_.first(); item != nullptr;
       item = instruction_list_.next(item)) {
    if (absl::c_find_if(placed_uses, [&](const auto& use) {
          return use.user == item;
        }) != placed_uses.end()) {
      last_placed_user = item;
    }
    if (first_unplaced_user == nullptr &&
        absl::c_find_if(unplaced_uses, [&](const auto& use) {
          return use.user == item;
        }) != unplaced_uses.end()) {
      first_unplaced_user = item;
      break;
    }
  }
  if (last_placed_user == nullptr) {
    VLOG(3) << "  " << candidate_instruction->name()
            << " has no placed users, starting search at self.";
    last_placed_user = candidate_item;
  }
  CHECK(first_unplaced_user != nullptr)
      << "Didn't find any unplaced user for instruction \""
      << candidate_instruction->name()
      << "\". There must be a "
         "bug in how we calculate how much memory this item uses.";
  float time_spent_before_next_use = 0.0;
  for (auto* item = last_placed_user; item != first_unplaced_user;
       item = instruction_list_.next(item)) {
    time_spent_before_next_use += std::max(
        0.0f, options_.hlo_cost_analysis.optimal_seconds(*item->instruction));
  }
  if (time_spent_before_next_use <= 0.0) {
    return {};
  }
  const float time_spent_on_copies =
      bytes_used_by_buffers / options_.host_memory_offload_config
                                  ->bandwidth_to_host_bytes_per_second +
      bytes_used_by_buffers / options_.host_memory_offload_config
                                  ->bandwidth_from_host_bytes_per_second;
  if (time_spent_before_next_use < time_spent_on_copies) {
    return {};
  }
  VLOG(3) << "  " << candidate_instruction->name() << " has enough time ("
          << time_spent_before_next_use
          << ") between itself and next use. The memcpy out and back will take "
          << time_spent_on_copies << "s";
  return memory_limit_bytes / bytes_used_by_buffers;
}
std::optional<int64_t> MemoryUsageTracker::GetCostOfRecompute(
    const std::vector<Item*>& candidate_items,
    int64_t memory_limit_bytes) const {
  for (auto* item : candidate_items) {
    HloInstruction* candidate = item->instruction;
    if (std::any_of(
            candidate->control_successors().begin(),
            candidate->control_successors().end(),
            [this](const HloInstruction* inst) { return IsPlaced(inst); })) {
      return {};
    }
  }
  VLOG(5) << "Block contains:";
  for (auto* hlo : candidate_items) {
    VLOG(5) << hlo->instruction->name();
  }
  const int64_t memory_reduced = MemoryReducedIfRematerialized(candidate_items);
  if (memory_reduced <= 0) {
    return {};
  }
  return RematerializationCost(candidate_items, memory_reduced,
                               memory_limit_bytes);
}
std::tuple<std::vector<Item*>, RematStrategy, int>
MemoryUsageTracker::PickRematerializationCandidates(
    const InstructionList& instruction_list, int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    int min_block_size, int max_block_size, int64_t peak_memory_bytes) {
  std::vector<Item*> best_items;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  RematStrategy best_strategy;
  int effort = 0;
  VLOG(5) << "Picking candidate block with size in [" << min_block_size << ", "
          << max_block_size << "]";
  for (auto* start_item = instruction_list.first_skip_node();
       start_item != nullptr;
       start_item = instruction_list.next_skip_node(start_item)) {
    std::vector<Item*> block =
        GetInitialBlock(instruction_list, *this, start_item, min_block_size);
    if (block.size() < min_block_size) {
      break;
    }
    if (AnyDenylistedOrNonRematerializable(block, rematerializable_map)) {
      continue;
    }
    if (options_.remat_mode_config.compress && block.size() == 1) {
      auto cost =
          GetCostOfCompression(block[0], memory_limit_bytes, peak_memory_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kCompress on block of size " << block.size();
        best_strategy.kind = RematStrategy::kCompress;
        best_strategy.compact_shape =
            *GetCompactShape(block[0]->instruction).value();
        best_items = block;
        best_cost = *cost;
      }
    }
    if (options_.remat_mode_config.host_offload && block.size() == 1) {
      auto cost = GetCostOfHostOffload(block[0], memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kHostOffload on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kHostOffload;
        best_items = block;
        best_cost = *cost;
      }
    }
    if (!options_.remat_mode_config.recompute) {
      continue;
    }
    while (block.size() <= max_block_size) {
      auto cost = GetCostOfRecompute(block, memory_limit_bytes);
      ++effort;
      if (cost && *cost < best_cost) {
        VLOG(1) << "Found new best cost; from " << best_cost << " to " << *cost
                << " with strategy kRecompute on block of size "
                << block.size();
        best_strategy.kind = RematStrategy::kRecompute;
        best_items = block;
        best_cost = *cost;
      }
      auto* last_item = block[block.size() - 1];
      auto* next_item = instruction_list.next(last_item);
      if (next_item == nullptr || next_item->denylisted || !next_item->placed ||
          next_item == in_progress_item_ ||
          !CanBeRematerialized(next_item->instruction, rematerializable_map)) {
        break;
      }
      block.push_back(next_item);
    }
  }
  return {best_items, best_strategy, effort};
}
bool MemoryUsageTracker::HasUnplacedUsers(Item* item) const {
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    for (const ItemUse& user : buffer.users) {
      if (!user.user->placed) {
        return true;
      }
    }
  }
  return false;
}
UsesList MemoryUsageTracker::GetItemUses(Item* item) const {
  UsesList combined_users;
  for (BufferId buffer_id : item->buffers_defined) {
    const Buffer& buffer = buffers_.at(buffer_id);
    for (const ItemUse& user : buffer.users) {
      combined_users.push_back(user);
    }
  }
  return combined_users;
}
absl::StatusOr<int64_t> RematerializeInstructions(
    MemoryUsageTracker* memory_tracker, std::vector<Item*>* best_items,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    InstructionList* instruction_list, HloSchedule* schedule,
    HloRematerialization* rematerialization) {
  int64_t net_instructions_added = 0;
  std::vector<std::string> instruction_names(best_items->size());
  for (int i = best_items->size() - 1; i >= 0; --i) {
    Item* best_item = (*best_items)[i];
    HloInstruction* best = best_item->instruction;
    instruction_names[i] = best->name();
    HloComputation* computation = best->parent();
    if (!memory_tracker->HasUnplacedUsers(best_item)) {
      continue;
    }
    HloCloneContext context(computation->parent());
    HloInstruction* remat =
        computation->AddInstruction(best->Clone("remat", &context));
    for (auto& cloned_computation_pair : context.cloned_computations()) {
      if (!schedule->is_computation_scheduled(cloned_computation_pair.first)) {
        continue;
      }
      HloInstructionSequence& sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.second);
      HloInstructionSequence& old_sequence =
          schedule->GetOrCreateSequence(cloned_computation_pair.first);
      for (HloInstruction* instr : old_sequence.instructions()) {
        sequence.push_back(instr);
      }
    }
    if (DynCast<HloChannelInstruction>(best) &&
        DynCast<HloChannelInstruction>(best)->channel_id()) {
      remat->set_channel_id(rematerialization->NextChannelId());
    }
    TF_RETURN_IF_ERROR(remat->CopyAllControlDepsFrom(best));
    Item* remat_item = instruction_list->CreateItem(remat);
    absl::InlinedVector<Item*, 4> indirect_users;
    absl::flat_hash_map<int64_t, HloInstruction*> gte_cache;
    for (auto& user : memory_tracker->GetItemUses(best_item)) {
      if (!memory_tracker->IsPlaced(user.user->instruction)) {
        VLOG(2) << "  Replacing use of " << best->name() << " in "
                << user.user->instruction->name() << " with " << remat->name();
        HloInstruction* remat_use = remat;
        HloInstruction* const user_operand =
            user.user->instruction->mutable_operand(user.operand_number);
        if (remat_use == user_operand) {
          continue;
        }
        if (user.index && remat_use->shape() != user_operand->shape()) {
          auto cached_gte = gte_cache.find(*user.index);
          if (cached_gte == gte_cache.end()) {
            remat_use = computation->AddInstruction(
                HloInstruction::CreateGetTupleElement(
                    ShapeUtil::GetTupleElementShape(remat_use->shape(),
                                                    *user.index),
                    remat_use, *user.index),
                "gte.remat");
            indirect_users.push_back(instruction_list->CreateItem(remat_use));
            gte_cache[*user.index] = remat_use;
          } else {
            remat_use = cached_gte->second;
          }
        }
        if (user_operand->shape() != remat_use->shape()) {
          remat_use = computation->AddInstruction(
              HloInstruction::CreateBitcast(user_operand->shape(), remat_use),
              "bitcast.remat");
          indirect_users.push_back(instruction_list->CreateItem(remat_use));
        }
        TF_RETURN_IF_ERROR(user.user->instruction->ReplaceOperandWith(
            user.operand_number, remat_use));
      }
    }
    TF_RETURN_IF_ERROR(memory_tracker->AddRematerializedInstruction(
        best_item, remat_item, absl::MakeSpan(indirect_users)));
    ItemList place_before;
    const absl::flat_hash_set<Item*> indirect_users_set(indirect_users.begin(),
                                                        indirect_users.end());
    for (auto user : remat->users()) {
      if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
        place_before.push_back(instruction_list->GetItem(user));
      }
    }
    for (auto* indirect_user : indirect_users) {
      for (auto user : indirect_user->instruction->users()) {
        if (!indirect_users_set.contains(instruction_list->GetItem(user))) {
          place_before.push_back(instruction_list->GetItem(user));
        }
      }
    }
    for (auto* operand : remat->operands()) {
      for (auto* operand_user : operand->users()) {
        if (operand_user != remat) {
          Item* operand_user_item = instruction_list->GetItem(operand_user);
          if (!operand_user_item->placed) {
            place_before.push_back(operand_user_item);
          }
        }
      }
    }
    for (auto successor : remat->control_successors()) {
      Item* successor_item = instruction_list->GetItem(successor);
      CHECK(!successor_item->placed) << successor_item->instruction->name();
      place_before.push_back(successor_item);
    }
    instruction_list->InsertBeforeInstructions(remat_item, place_before);
    for (auto* bitcast : indirect_users) {
      instruction_list->InsertBeforeInstructions(bitcast, place_before);
    }
    std::function<bool(HloInstruction*)> uses_empty = [&](HloInstruction* i) {
      for (auto* u : i->users()) {
        if (!IsSupportedIndirectUser(u) || !uses_empty(u)) {
          return false;
        }
      }
      return true;
    };
    if (uses_empty(best)) {
      VLOG(2) << best->name() << " is now dead";
      if (ContainsKey(*remat_move_instructions, best)) {
        instruction_list->Denylist(remat);
      }
      remat_move_instructions->insert(remat);
      net_instructions_added += indirect_users.size();
    } else {
      net_instructions_added += indirect_users.size() + 1;
    }
    for (auto* indirect_user : indirect_users) {
      instruction_list->Denylist(indirect_user->instruction);
    }
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(best->opcode()) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(best->opcode())) {
      VLOG(2) << "The old instruction " << best->name()
              << " is an async op. Removing to maintain one start to one done "
                 "invariant to keep the HLO valid.";
      TF_RETURN_IF_ERROR(best->DropAllControlDeps());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(best));
    }
  }
  return net_instructions_added;
}
absl::StatusOr<int64_t> CompressInstruction(MemoryUsageTracker* memory_tracker,
                                            Item* best_item,
                                            const Shape& compact_shape,
                                            InstructionList* instruction_list) {
  HloInstruction* best = best_item->instruction;
  VLOG(5) << "Transposing instruction " << best->name() << " (saving "
          << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                 best_item, compact_shape))
          << ") to" << compact_shape.ToString(true);
  HloComputation* computation = best->parent();
  HloInstruction* compressed = computation->AddInstruction(
      HloInstruction::CreateUnary(compact_shape, HloOpcode::kCopy, best),
      absl::StrCat(best->name(), ".remat_compressed"));
  HloInstruction* uncompressed = computation->AddInstruction(
      HloInstruction::CreateUnary(best->shape(), HloOpcode::kCopy, compressed),
      absl::StrCat(best->name(), ".remat_uncompressed"));
  Item* compressed_item = instruction_list->CreateItem(compressed);
  compressed_item->placed = true;
  Item* uncompressed_item = instruction_list->CreateItem(uncompressed);
  std::vector<HloInstruction*> best_users_copy = best->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsPlaced(user)) {
      VLOG(5) << "  Replacing use of " << best->name() << " in " << user->name()
              << " with " << uncompressed->name();
      TF_RETURN_IF_ERROR(best->ReplaceUseWith(user, uncompressed));
    }
  }
  TF_RETURN_IF_ERROR(memory_tracker->AddCompressInstructions(
      best_item, compressed_item, uncompressed_item));
  ItemList place_before;
  for (auto user : uncompressed->users()) {
    place_before.push_back(instruction_list->GetItem(user));
  }
  instruction_list->Denylist(compressed_item->instruction);
  instruction_list->Denylist(uncompressed_item->instruction);
  instruction_list->InsertBeforeInstructions(uncompressed_item, place_before);
  instruction_list->InsertAfterInstructions(compressed_item, {best_item});
  return 2;
}
absl::StatusOr<int64_t> OffloadInstruction(MemoryUsageTracker* memory_tracker,
                                           Item* best_item,
                                           InstructionList* instruction_list) {
  HloInstruction* best_instruction = best_item->instruction;
  HloComputation* computation = best_instruction->parent();
  VLOG(2) << "Best_instruction's users: "
          << absl::StrJoin(best_instruction->users(), ", ",
                           [](std::string* str, const auto* x) {
                             return str->append(x->name());
                           });
  Shape instruction_shape_device = best_instruction->shape();
  Shape instruction_shape_host = best_instruction->shape();
  instruction_shape_host.mutable_layout()->set_memory_space(
      memory_tracker->options().host_memory_offload_config->host_memory_space);
  Shape context_shape = ShapeUtil::MakeShape(U32, {});
  HloInstruction* copy_start_to_host =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_host,
                                     instruction_shape_device, context_shape}),
          best_instruction));
  HloInstruction* copy_done_to_host =
      computation->AddInstruction(HloInstruction::CreateUnary(
          instruction_shape_host, HloOpcode::kCopyDone, copy_start_to_host));
  HloInstruction* copy_start_to_device =
      computation->AddInstruction(HloInstruction::CreateCopyStart(
          ShapeUtil::MakeTupleShape({instruction_shape_device,
                                     instruction_shape_host, context_shape}),
          copy_done_to_host));
  HloInstruction* copy_done_to_device = computation->AddInstruction(
      HloInstruction::CreateUnary(instruction_shape_device,
                                  HloOpcode::kCopyDone, copy_start_to_device));
  VLOG(3) << "Created copy_start_to_host instr: "
          << copy_start_to_host->ToString();
  VLOG(3) << "Created copy_done_to_host instr: "
          << copy_done_to_host->ToString();
  VLOG(3) << "Created copy_start_to_device instr: "
          << copy_start_to_device->ToString();
  VLOG(3) << "Created copy_done_to_device instr: "
          << copy_done_to_device->ToString();
  TF_RETURN_IF_ERROR(
      copy_start_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_host->Visit(&memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(copy_start_to_device->Visit(
      &memory_tracker->options().hlo_cost_analysis));
  TF_RETURN_IF_ERROR(
      copy_done_to_device->Visit(&memory_tracker->options().hlo_cost_analysis));
  Item* copy_start_to_host_item =
      instruction_list->CreateItem(copy_start_to_host);
  Item* copy_done_to_host_item =
      instruction_list->CreateItem(copy_done_to_host);
  Item* copy_start_to_device_item =
      instruction_list->CreateItem(copy_start_to_device);
  Item* copy_done_to_device_item =
      instruction_list->CreateItem(copy_done_to_device);
  instruction_list->Denylist(copy_start_to_host);
  instruction_list->Denylist(copy_done_to_host);
  instruction_list->Denylist(copy_start_to_device);
  instruction_list->Denylist(copy_done_to_device);
  Item* place_before{nullptr};
  {
    ItemList place_before_list;
    for (auto user : best_instruction->users()) {
      if (user == copy_start_to_host) {
        continue;
      }
      auto item_of_user = instruction_list->GetItem(user);
      if (item_of_user->placed) {
        continue;
      }
      place_before_list.push_back(item_of_user);
    }
    CHECK(!place_before_list.empty()) << "Have nothing to place this before!";
    for (auto* item = instruction_list->first(); item != nullptr;
         item = instruction_list->next(item)) {
      if (absl::c_linear_search(place_before_list, item)) {
        place_before = item;
        break;
      }
    }
  }
  CHECK_NE(place_before, nullptr)
      << "Could not find an item to place this before.";
  auto get_first_item_after_compute_time = [&](Item* start_item, Item* end_item,
                                               auto successor_func,
                                               float time_spent_on_copy) {
    float time_so_far = 0.0;
    auto* current_item = start_item;
    while (time_so_far < time_spent_on_copy) {
      auto next_item = successor_func(current_item);
      if (next_item == end_item) {
        LOG(WARNING) << "Didn't find enough computation before end of window";
        break;
      }
      current_item = next_item;
      CHECK_NE(current_item, nullptr) << "current_item is null";
      CHECK_NE(current_item->instruction, nullptr)
          << "current_item's instruction is null";
      time_so_far += std::max(
          0.0f, memory_tracker->options().hlo_cost_analysis.optimal_seconds(
                    *current_item->instruction));
    }
    return current_item;
  };
  const int64_t bytes_used_by_buffers = memory_tracker->BytesUsedByBuffers(
      best_item, false);
  const float copy_to_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_to_host_bytes_per_second;
  const float copy_from_host_time_seconds =
      bytes_used_by_buffers /
      memory_tracker->options()
          .host_memory_offload_config->bandwidth_from_host_bytes_per_second;
  VLOG(2) << "Item uses " << bytes_used_by_buffers << "B and will take "
          << copy_to_host_time_seconds << "s to copy to host and "
          << copy_from_host_time_seconds << "s to copy from host.";
  VLOG(2) << "Inserting " << copy_start_to_host_item->instruction->name()
          << " immediately after " << best_item->instruction->name();
  instruction_list->InsertAfterInstructions(copy_start_to_host_item,
                                            {best_item});
  VLOG(2) << "Inserting " << copy_done_to_device_item->instruction->name()
          << " immediately before " << place_before->instruction->name();
  instruction_list->InsertBeforeInstructions(copy_done_to_device_item,
                                             {place_before});
  auto first_item_after_to_host_copy = get_first_item_after_compute_time(
      copy_start_to_host_item, copy_done_to_device_item,
      [&instruction_list](Item* item) { return instruction_list->next(item); },
      copy_to_host_time_seconds);
  VLOG(2) << "Inserting " << copy_done_to_host_item->instruction->name()
          << " immediately after "
          << first_item_after_to_host_copy->instruction->name();
  instruction_list->InsertAfterInstructions(copy_done_to_host_item,
                                            {first_item_after_to_host_copy});
  auto first_item_before_from_host_copy = get_first_item_after_compute_time(
      copy_done_to_device_item, copy_done_to_host_item,
      [&instruction_list](Item* item) { return instruction_list->prev(item); },
      copy_from_host_time_seconds);
  VLOG(2) << "Inserting " << copy_start_to_device_item->instruction->name()
          << " immediately before "
          << first_item_before_from_host_copy->instruction->name();
  instruction_list->InsertBeforeInstructions(
      copy_start_to_device_item, {first_item_before_from_host_copy});
  {
    auto item = instruction_list->first();
    while (item != nullptr) {
      if (item == copy_start_to_host_item || item == copy_done_to_host_item ||
          item == copy_start_to_device_item ||
          item == copy_done_to_device_item) {
        item->placed = true;
      } else if (memory_tracker->IsInProgressItem(item)) {
        break;
      }
      item = instruction_list->next(item);
    }
  }
  std::vector<HloInstruction*> best_users_copy = best_instruction->users();
  for (HloInstruction* user : best_users_copy) {
    if (!memory_tracker->IsPlaced(user)) {
      VLOG(3) << "  Replacing use of " << best_instruction->name() << " in "
              << user->name() << " with " << copy_done_to_device->name();
      TF_RETURN_IF_ERROR(
          best_instruction->ReplaceUseWith(user, copy_done_to_device));
    } else {
      VLOG(3) << user->name() << " is placed, not going to update";
    }
  }
  TF_RETURN_IF_ERROR(memory_tracker->AddHostOffloadCopyInstructions(
      best_item, copy_start_to_host_item, copy_done_to_host_item,
      copy_start_to_device_item, copy_done_to_device_item));
  return 4;
}
struct InstructionsAdded {
  int remat_count;
  int net_instructions_added;
  int effort;
};
absl::StatusOr<InstructionsAdded> RematerializeBestBlock(
    int min_block_size, int max_block_size, MemoryUsageTracker* memory_tracker,
    InstructionList* instruction_list, HloSchedule* schedule,
    int64_t memory_limit_bytes,
    absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
    absl::flat_hash_set<const HloInstruction*>* remat_move_instructions,
    HloRematerialization* rematerialization) {
  CHECK(min_block_size > 0) << "Negative block size.";
  std::vector<Item*> best_items;
  RematStrategy best_strategy;
  int effort;
  std::tie(best_items, best_strategy, effort) =
      memory_tracker->PickRematerializationCandidates(
          *instruction_list, memory_limit_bytes, rematerializable_map,
          min_block_size, max_block_size,
          rematerialization->ComputationPeakMemory(
              memory_tracker->computation()));
  InstructionsAdded num_instructions_added;
  num_instructions_added.remat_count = best_items.size();
  num_instructions_added.effort = effort;
  if (best_items.empty()) {
    num_instructions_added.net_instructions_added = 0;
    return num_instructions_added;
  }
  if (best_strategy.kind == RematStrategy::kCompress) {
    CHECK(best_items.size() == 1)
        << "More than one instruction compressed simultaneously.";
    HloInstruction* best = best_items[0]->instruction;
    VLOG(1) << "Remat via compression: " << best->name() << " (saving "
            << HumanReadableNumBytes(memory_tracker->MemoryReducedIfCompressed(
                   best_items[0], best_strategy.compact_shape))
            << ")";
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        CompressInstruction(memory_tracker, best_items[0],
                            best_strategy.compact_shape, instruction_list));
  } else if (best_strategy.kind == RematStrategy::kHostOffload) {
    CHECK_EQ(best_items.size(), 1)
        << "More than one buffer offloaded simultaneously.";
    VLOG(1) << "Remat via offload: " << best_items[0]->instruction->name();
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        OffloadInstruction(memory_tracker, best_items[0], instruction_list));
    VLOG(4) << "Offload done, hlo computation:\n"
            << memory_tracker->computation()->ToString();
    VLOG(6) << "Memory tracker:\n" << memory_tracker->ToString();
  } else {
    CHECK_EQ(best_strategy.kind, RematStrategy::kRecompute)
        << "Expecting strategy to be Recompute";
    VLOG(1) << "Remat via recomputation: {"
            << absl::StrJoin(best_items, ", ",
                             [](std::string* out, Item* item) {
                               absl::StrAppend(out, item->instruction->name());
                             })
            << '}';
    TF_ASSIGN_OR_RETURN(
        num_instructions_added.net_instructions_added,
        RematerializeInstructions(memory_tracker, &best_items,
                                  remat_move_instructions, instruction_list,
                                  schedule, rematerialization));
  }
  return num_instructions_added;
}
}  
absl::StatusOr<int64_t> HloRematerialization::ComputePeakMemory(
    const HloComputation* computation, const HloInstructionSequence& order,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  InstructionList instruction_list(order);
  MemoryUsageTracker tracker(options_, computation, *points_to_analysis_,
                             instruction_list);
  int64_t peak_memory = tracker.memory_usage();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_RETURN_IF_ERROR(tracker.BeginInstruction(item));
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    peak_memory =
        std::max<int64_t>(peak_memory, tracker.memory_usage() + callee_usage);
    TF_RETURN_IF_ERROR(tracker.EndInstruction());
  }
  VLOG(1) << "Peak memory for " << computation->name() << ": "
          << HumanReadableNumBytes(peak_memory);
  return peak_memory;
}
absl::StatusOr<int64_t> HloRematerialization::CalledComputationsMemoryUsage(
    const HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kEmbedded) {
    return 0;
  }
  int64_t callee_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads)) {
      continue;
    }
    TF_RET_CHECK(ContainsKey(computation_peak_memory_, computation));
    callee_usage += computation_peak_memory_.at(computation);
  }
  return callee_usage;
}
absl::StatusOr<bool> HloRematerialization::RematerializeComputation(
    HloComputation* computation, HloSchedule* schedule,
    int64_t memory_limit_bytes, int64_t min_remat_size,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto peak_memory_usage = computation_peak_memory_.at(computation);
  if (peak_memory_usage <= memory_limit_bytes) {
    VLOG(1) << "Asked to rematerialize computation of size "
            << peak_memory_usage
            << " but it already fits within the given memory limit ("
            << memory_limit_bytes << ")";
    return false;
  }
  VLOG(1) << "Rematerializing computation " << computation->name()
          << " with limit " << HumanReadableNumBytes(memory_limit_bytes);
  VLOG(1) << "peak memory usage is "
          << HumanReadableNumBytes(peak_memory_usage);
  CHECK(!ContainsKey(rematerialized_computations_, computation));
  InstructionList instruction_list(schedule->sequence(computation));
  MemoryUsageTracker memory_tracker(options_, computation, *points_to_analysis_,
                                    instruction_list);
  instruction_list.PromoteNodesToSkip([&](Item* item) {
    return memory_tracker.AllocatedSize(item) >= min_remat_size;
  });
  bool changed = false;
  absl::flat_hash_set<const HloInstruction*> remat_move_instructions;
  absl::flat_hash_map<const HloInstruction*, bool> rematerializable_map;
  int64_t peak_memory = memory_tracker.memory_usage();
  int64_t remat_count = 0;
  int64_t net_instructions_added = 0;
  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
  int64_t instruction_index = 0;
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(
        int64_t callee_usage,
        CalledComputationsMemoryUsage(instruction, execution_threads));
    TF_RETURN_IF_ERROR(memory_tracker.BeginInstruction(item));
    VLOG(2) << "Program point at " << instruction->name()
            << ", memory usage = " << memory_tracker.memory_usage()
            << ", callee usage = " << callee_usage << ", [" << instruction_index
            << "/" << instruction_list.size() << "]";
    instruction_index++;
    int min_block_size = 1;
    int max_block_size = 1;
    if (memory_tracker.AllocatedSize(item) + callee_usage > 0) {
      bool is_first_phase = true;
      int64_t first_phase_effort = 0;
      int64_t second_phase_effort = 0;
      while (memory_tracker.memory_usage() + callee_usage >
             memory_limit_bytes) {
        VLOG(2) << "Over memory limit at instruction " << instruction->name()
                << ", using "
                << HumanReadableNumBytes(memory_tracker.memory_usage() +
                                         callee_usage)
                << ", limit is " << HumanReadableNumBytes(memory_limit_bytes);
        TF_ASSIGN_OR_RETURN(
            InstructionsAdded instructions_added,
            RematerializeBestBlock(min_block_size, max_block_size,
                                   &memory_tracker, &instruction_list, schedule,
                                   memory_limit_bytes, &rematerializable_map,
                                   &remat_move_instructions, this));
        net_instructions_added += instructions_added.net_instructions_added;
        remat_count += instructions_added.remat_count;
        if (is_first_phase) {
          first_phase_effort += instructions_added.effort;
        } else {
          second_phase_effort += instructions_added.effort;
        }
        if (instructions_added.net_instructions_added > 0) {
          VLOG(1) << "memory_usage after rematerialization = "
                  << HumanReadableNumBytes(memory_tracker.memory_usage());
        }
        if (instructions_added.remat_count == 0) {
          min_block_size = max_block_size + 1;
          max_block_size = 2 * max_block_size;
          is_first_phase = false;
        } else {
          max_rematerialized_block_size_ =
              std::max(max_rematerialized_block_size_, max_block_size);
          changed = true;
          min_block_size = 1;
          max_block_size = 1;
        }
        if (max_block_size > options_.block_size_limit ||
            second_phase_effort >
                options_.block_rematerialization_factor * first_phase_effort) {
          break;
        }
      }
    }
    const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    if (callsite != nullptr &&
        callsite->context() == CallContext::kControlFlow &&
        memory_tracker.memory_usage() + callee_usage > memory_limit_bytes) {
      VLOG(1) << "Memory usage still over the limit ("
              << (memory_tracker.memory_usage() + callee_usage) << " > "
              << memory_limit_bytes
              << "). Rematerializing computations called by "
              << instruction->name();
      for (HloComputation* called_computation :
           callsite->called_computations()) {
        if (!ContainsKey(rematerialized_computations_, called_computation) &&
            HloInstruction::IsThreadIncluded(
                called_computation->execution_thread(), execution_threads)) {
          int64_t subcomputation_memory_limit_bytes = std::max<int64_t>(
              0, memory_limit_bytes - memory_tracker.memory_usage());
          TF_ASSIGN_OR_RETURN(
              bool subcomputation_changed,
              RematerializeComputation(called_computation, schedule,
                                       subcomputation_memory_limit_bytes,
                                       min_remat_size, execution_threads));
          changed |= subcomputation_changed;
        }
      }
      TF_ASSIGN_OR_RETURN(callee_usage, CalledComputationsMemoryUsage(
                                            instruction, execution_threads));
    }
    peak_memory = std::max<int64_t>(
        peak_memory, memory_tracker.memory_usage() + callee_usage);
    VLOG(3) << "peak memory usage = " << HumanReadableNumBytes(peak_memory);
    TF_RETURN_IF_ERROR(memory_tracker.EndInstruction());
  }
  for (auto* instruction : computation->instructions()) {
    CHECK(memory_tracker.IsPlaced(instruction)) << instruction->name();
  }
  VLOG(1) << "In computation " << computation->name() << " rematerialized "
          << remat_count << " instructions; " << net_instructions_added
          << " net instructions added";
  VLOG(1) << "  peak memory usage now " << HumanReadableNumBytes(peak_memory)
          << " (was "
          << HumanReadableNumBytes(computation_peak_memory_.at(computation))
          << ")";
  computation_peak_memory_.at(computation) = peak_memory;
  HloInstructionSequence& sequence = schedule->GetOrCreateSequence(computation);
  sequence.clear();
  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    HloInstruction* instruction = item->instruction;
    sequence.push_back(instruction);
  }
  rematerialized_computations_.insert(computation);
  instructions_rematerialized_ += remat_count;
  net_instructions_added_ += net_instructions_added;
  return changed;
}
absl::StatusOr<bool> HloRematerialization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (options_.remat_mode_config.host_offload) {
    CHECK(options_.host_memory_offload_config.has_value())
        << "Host memory config is required when host memory offload strategy "
           "is specified";
  }
  VLOG(1) << "HloRematerialization() with memory limit of "
          << HumanReadableNumBytes(options_.memory_limit_bytes);
  if (!options_.remat_mode_config.compress &&
      !options_.remat_mode_config.recompute &&
      !options_.remat_mode_config.host_offload) {
    VLOG(1) << "All rematerialization strategies are disabled. Skipping.";
    return false;
  }
  VLOG(2) << "HloRemat mode: compress: " << options_.remat_mode_config.compress
          << ", host_offload: " << options_.remat_mode_config.host_offload
          << ", recompute: " << options_.remat_mode_config.recompute;
  XLA_VLOG_LINES(3, "Before HloRematerialization:\n" + module->ToString());
  computation_peak_memory_.clear();
  rematerialized_computations_.clear();
  instructions_rematerialized_ = 0;
  net_instructions_added_ = 0;
  TF_RET_CHECK(module->has_schedule());
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));
  next_channel_id_ = hlo_query::NextChannelId(*module);
  int64_t module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&module_output_size, this](const Shape& subshape,
                                  const ShapeIndex& output_index) {
        module_output_size += options_.hlo_cost_analysis.GetShapeSize(subshape);
      });
  int64_t adjusted_memory_limit_bytes =
      std::max<int64_t>(0, options_.memory_limit_bytes - module_output_size);
  VLOG(1) << "Adjusted memory limit accounting for output ("
          << HumanReadableNumBytes(module_output_size)
          << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);
  call_graph_ = CallGraph::Build(module);
  int64_t total_async_peak_memory = 0;
  if (!options_.async_computation_parallelism.empty()) {
    absl::flat_hash_set<std::string_view> async_threads;
    for (const auto& [computation, _] :
         options_.async_computation_parallelism) {
      async_threads.insert(computation->execution_thread());
    }
    TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
        [this, module,
         &async_threads](const CallGraphNode& node) -> absl::Status {
          auto callee_thread = node.computation()->execution_thread();
          if (node.context() == CallContext::kControlFlow &&
              HloInstruction::IsThreadIncluded(callee_thread, async_threads)) {
            TF_ASSIGN_OR_RETURN(computation_peak_memory_[node.computation()],
                                ComputePeakMemory(node.computation(),
                                                  module->schedule().sequence(
                                                      node.computation()),
                                                  {callee_thread}));
          }
          return absl::OkStatus();
        },
        false));
    int64_t async_peak_memory = 0;
    for (const auto [entry_computation, parallel_threads] :
         options_.async_computation_parallelism) {
      const int64_t peak_memory =
          computation_peak_memory_.at(entry_computation);
      const int64_t parallel_peak_memory = peak_memory * parallel_threads;
      async_peak_memory = std::max(async_peak_memory, parallel_peak_memory);
    }
    adjusted_memory_limit_bytes =
        std::max<int64_t>(0, adjusted_memory_limit_bytes - async_peak_memory);
    total_async_peak_memory += async_peak_memory;
    VLOG(1) << "Adjusted memory limit accounting for async computations ("
            << HumanReadableNumBytes(async_peak_memory)
            << "): " << HumanReadableNumBytes(adjusted_memory_limit_bytes);
    computation_peak_memory_.clear();
  }
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, module,
       &execution_threads](const CallGraphNode& node) -> absl::Status {
        if (node.context() == CallContext::kControlFlow &&
            HloInstruction::IsThreadIncluded(
                node.computation()->execution_thread(), execution_threads)) {
          TF_ASSIGN_OR_RETURN(
              computation_peak_memory_[node.computation()],
              ComputePeakMemory(node.computation(),
                                module->schedule().sequence(node.computation()),
                                execution_threads));
        }
        return absl::OkStatus();
      },
      false));
  const int64_t before_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size + total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&options_.hlo_cost_analysis));
  }
  TF_ASSIGN_OR_RETURN(
      bool changed,
      RematerializeComputation(module->entry_computation(), &module->schedule(),
                               adjusted_memory_limit_bytes,
                               options_.min_remat_size, execution_threads));
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();
  TF_ASSIGN_OR_RETURN(bool dead_code_removed, HloDCE().Run(module));
  changed |= dead_code_removed;
  TF_RETURN_IF_ERROR(saved_schedule.Update(execution_threads));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));
  VLOG(1) << "Rematerialized " << instructions_rematerialized_
          << " instructions in module " << module->name() << "; "
          << net_instructions_added_ << " net instructions added";
  const int64_t current_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size + total_async_peak_memory;
  VLOG(1) << "Peak memory usage of module now "
          << HumanReadableNumBytes(current_peak_memory) << " ("
          << current_peak_memory << " bytes), was "
          << HumanReadableNumBytes(before_peak_memory) << " ("
          << before_peak_memory << " bytes)";
  const int64_t reduced_peak_memory = before_peak_memory - current_peak_memory;
  VLOG(1) << "Reduced peak memory by "
          << HumanReadableNumBytes(reduced_peak_memory) << " ("
          << reduced_peak_memory << " bytes)";
  sizes_.before_bytes = before_peak_memory;
  sizes_.after_bytes = current_peak_memory;
  XLA_VLOG_LINES(5, "After HloRematerialization:\n" + module->ToString());
  if (current_peak_memory > options_.memory_limit_bytes) {
    LOG(WARNING) << absl::StrFormat(
        "Can't reduce memory use below %s (%d bytes) by rematerialization; "
        "only reduced to %s (%d bytes), down from %s (%d bytes) originally",
        HumanReadableNumBytes(options_.memory_limit_bytes),
        options_.memory_limit_bytes, HumanReadableNumBytes(current_peak_memory),
        current_peak_memory, HumanReadableNumBytes(before_peak_memory),
        before_peak_memory);
  }
  return changed;
}
}  