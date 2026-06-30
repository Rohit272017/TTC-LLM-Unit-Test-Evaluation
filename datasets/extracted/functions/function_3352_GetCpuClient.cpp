#include <memory>
#include <utility>
#include "absl/status/statusor.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_options.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tsl/platform/statusor.h"
namespace xla {
absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetCpuClient(
    const PjrtClientFactoryOptions& option) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      xla::GetTfrtCpuClient(option.cpu_options.asynchronous));
  return std::move(client);
}
REGISTER_PJRT_CLIENT_FACTORY(cpu_client, tensorflow::DEVICE_CPU, GetCpuClient);
}  