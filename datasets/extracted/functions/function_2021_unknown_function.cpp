#include "tensorflow/core/common_runtime/null_request_cost_accessor.h"
namespace tensorflow {
RequestCost* NullRequestCostAccessor::GetRequestCost() const { return nullptr; }
REGISTER_REQUEST_COST_ACCESSOR("null", NullRequestCostAccessor);
}  