#include "base/ast_internal/ast_impl.h"
#include <cstdint>
#include "absl/container/flat_hash_map.h"
namespace cel::ast_internal {
namespace {
const Type& DynSingleton() {
  static auto* singleton = new Type(TypeKind(DynamicType()));
  return *singleton;
}
}  
const Type& AstImpl::GetType(int64_t expr_id) const {
  auto iter = type_map_.find(expr_id);
  if (iter == type_map_.end()) {
    return DynSingleton();
  }
  return iter->second;
}
const Type& AstImpl::GetReturnType() const { return GetType(root_expr().id()); }
const Reference* AstImpl::GetReference(int64_t expr_id) const {
  auto iter = reference_map_.find(expr_id);
  if (iter == reference_map_.end()) {
    return nullptr;
  }
  return &iter->second;
}
}  