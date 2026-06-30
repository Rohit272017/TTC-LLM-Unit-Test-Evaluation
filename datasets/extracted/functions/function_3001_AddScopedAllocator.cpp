#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/framework/allocator.h"
namespace tensorflow {
Status ScopedAllocatorContainer::AddScopedAllocator(
    const Tensor& backing_tensor, int32_t scope_id, const string& scope_name,
    const absl::Span<const ScopedAllocator::Field>& fields,
    int32_t expected_call_count) {
  VLOG(1) << "AddScopedAllocator " << mgr_->device_name()
          << " step_id_=" << step_id_ << " scope_id=" << scope_id;
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return errors::Internal("Cannot create ScopedAllocator because scope_id ",
                            scope_id, " for name ", scope_name,
                            " already exists");
  }
  for (auto& f : fields) {
    if (allocators_.find(f.scope_id) != allocators_.end()) {
      return errors::Internal(
          "Cannot create ScopedAllocator because field scope_id ", f.scope_id,
          " for name ", scope_name, " already exists");
    }
  }
  VLOG(2) << " container " << this << " step_id " << step_id_;
  ScopedAllocator* sa = new ScopedAllocator(
      backing_tensor, scope_id, scope_name, fields, expected_call_count, this);
  allocators_[scope_id] =
      ScopedAllocatorContainer::SAField(ScopedAllocator::kBackingIndex, sa);
  VLOG(2) << "#fields " << fields.size();
  for (int i = 0; i < fields.size(); ++i) {
    const ScopedAllocator::Field& f = fields[i];
    VLOG(2) << "Adding instance with for " << mgr_->device_name()
            << " scope_id=" << f.scope_id;
    allocators_[f.scope_id] = ScopedAllocatorContainer::SAField(
        i, new ScopedAllocatorInstance(sa, i));
  }
  return absl::OkStatus();
}
ScopedAllocator* ScopedAllocatorContainer::GetAllocator(int32_t scope_id) {
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    CHECK_EQ(ScopedAllocator::kBackingIndex, it->second.field_index);
    return it->second.scoped_allocator;
  } else {
    LOG(ERROR) << "Failed to find ScopedAllocator for " << scope_id
               << " in container for step " << step_id_ << " on "
               << mgr_->device_name();
    return nullptr;
  }
}
ScopedAllocatorInstance* ScopedAllocatorContainer::GetInstance(
    int32_t scope_id) {
  VLOG(2) << "GetInstance " << scope_id << " step " << step_id_ << " on "
          << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    return it->second.instance;
  }
  LOG(FATAL) << "Failed to find instance " << scope_id << " in container "
             << step_id_ << " on " << mgr_->device_name();
  return nullptr;
}
void ScopedAllocatorContainer::Drop(int32_t scope_id, ScopedAllocator* sa) {
  VLOG(2) << "Drop " << scope_id << " from container " << this << " step "
          << step_id_ << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  auto it = allocators_.find(scope_id);
  if (it != allocators_.end()) {
    if (it->second.field_index != ScopedAllocator::kBackingIndex) {
      it->second.instance->DropFromTable();
    }
    allocators_.erase(it);
  }
}
ScopedAllocatorContainer::~ScopedAllocatorContainer() {
  VLOG(2) << "~ScopedAllocatorContainer " << this << " step " << step_id_
          << " on " << mgr_->device_name();
  mutex_lock l(mu_);
  for (auto& it : allocators_) {
    if (it.second.field_index == ScopedAllocator::kBackingIndex) {
      delete it.second.scoped_allocator;
    } else {
      it.second.instance->DropFromTable();
    }
  }
}
ScopedAllocatorMgr::~ScopedAllocatorMgr() {
  mutex_lock l(mu_);
  for (auto it : per_step_map_) {
    while (!it.second->Unref()) {
    }
  }
}
void ScopedAllocatorMgr::Cleanup(int64_t step_id) {
  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it != per_step_map_.end()) {
    it->second->Unref();
    per_step_map_.erase(it);
  }
}
ScopedAllocatorContainer* ScopedAllocatorMgr::GetContainer(int64_t step_id) {
  VLOG(2) << "GetContainer " << step_id << " on " << device_name();
  ScopedAllocatorContainer* sac = nullptr;
  mutex_lock l(mu_);
  auto it = per_step_map_.find(step_id);
  if (it == per_step_map_.end()) {
    sac = new ScopedAllocatorContainer(this, step_id);
    per_step_map_[step_id] = sac;
  } else {
    sac = it->second;
  }
  return sac;
}
Status ScopedAllocatorMgr::AddScopedAllocator(
    const Tensor& backing_tensor, int64_t step_id, int32_t scope_id,
    const string& scope_name,
    const absl::Span<const ScopedAllocator::Field>& fields,
    int32_t expected_call_count) {
  ScopedAllocatorContainer* sac = GetContainer(step_id);
  return sac->AddScopedAllocator(backing_tensor, scope_id, scope_name, fields,
                                 expected_call_count);
}
size_t ScopedAllocatorMgr::PopulateFields(
    int32_t scope_id, const absl::Span<const TensorShape>& shapes,
    const DataType dtype, std::vector<ScopedAllocator::Field>* fields) {
  const int32_t num_fields = static_cast<int32>(shapes.size());
  fields->resize(num_fields);
  size_t offset = 0;
  for (int32_t i = 0; i < num_fields; ++i) {
    size_t bytes_requested = shapes[i].num_elements() * DataTypeSize(dtype);
    auto* field = &((*fields)[i]);
    field->scope_id = scope_id + 1 + i;
    field->bytes_requested = bytes_requested;
    field->offset = offset;
    offset += bytes_requested;
    size_t bytes_allocated = bytes_requested;
    size_t overshoot = offset % Allocator::kAllocatorAlignment;
    if (overshoot > 0) {
      size_t alignment_bytes = Allocator::kAllocatorAlignment - overshoot;
      bytes_allocated += alignment_bytes;
      offset += alignment_bytes;
    }
    field->bytes_allocated = bytes_allocated;
    VLOG(1) << "field=" << i << " scope_id=" << field->scope_id
            << " bytes_requested=" << field->bytes_requested
            << " offset=" << field->offset
            << " bytes_allocated=" << field->bytes_allocated;
  }
  return offset;
}
}  