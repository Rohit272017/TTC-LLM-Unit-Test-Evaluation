#include "quiche/quic/core/quic_connection_context.h"
#include "absl/base/attributes.h"
namespace quic {
namespace {
ABSL_CONST_INIT thread_local QuicConnectionContext* current_context = nullptr;
}  
QuicConnectionContext* QuicConnectionContext::Current() {
  return current_context;
}
QuicConnectionContextSwitcher::QuicConnectionContextSwitcher(
    QuicConnectionContext* new_context)
    : old_context_(QuicConnectionContext::Current()) {
  current_context = new_context;
  if (new_context && new_context->listener) {
    new_context->listener->Activate();
  }
}
QuicConnectionContextSwitcher::~QuicConnectionContextSwitcher() {
  QuicConnectionContext* current = QuicConnectionContext::Current();
  if (current && current->listener) {
    current->listener->Deactivate();
  }
  current_context = old_context_;
}
}  