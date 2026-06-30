#include "arolla/util/init_arolla.h"
#include <utility>
#include <vector>
#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "arolla/util/init_arolla_internal.h"
namespace arolla::init_arolla_internal {
namespace {
bool init_arolla_called = false;
const Registration* registry_head = nullptr;
void RunRegisteredInitializers() {
  static absl::NoDestructor<Coordinator> coordinator;
  auto* head = std::exchange(registry_head, nullptr);
  std::vector<const Initializer*> initializers;
  for (auto it = head; it != nullptr; it = it->next) {
    initializers.push_back(&it->initializer);
  }
  auto status = coordinator->Run(initializers);
  if (!status.ok()) {
    LOG(FATAL) << "Arolla initialization failed: " << status;
  }
}
}  
Registration::Registration(const Initializer& initializer)
    : initializer(initializer), next(registry_head) {
  registry_head = this;
}
void InitArollaSecondary() {
  if (init_arolla_called) {
    RunRegisteredInitializers();
  }
}
}  
namespace arolla {
void InitArolla() {
  [[maybe_unused]] static const bool done = [] {
    arolla::init_arolla_internal::init_arolla_called = true;
    arolla::init_arolla_internal::RunRegisteredInitializers();
    return true;
  }();
}
void CheckInitArolla() {
  constexpr absl::string_view message =
      ("The Arolla library is not initialized yet. Please ensure that "
       "arolla::InitArolla() was called before using any other Arolla"
       " functions."
      );
  if (!arolla::init_arolla_internal::init_arolla_called) {
    LOG(FATAL) << message;
  }
}
}  