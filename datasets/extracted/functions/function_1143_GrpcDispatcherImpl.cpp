#include "tensorflow/core/data/service/grpc_dispatcher_impl.h"
#include "grpcpp/server_context.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
namespace tensorflow {
namespace data {
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
GrpcDispatcherImpl::GrpcDispatcherImpl(
    const experimental::DispatcherConfig& config, ServerBuilder& server_builder)
    : impl_(config) {
  server_builder.RegisterService(this);
  VLOG(1) << "Registered data service dispatcher";
}
Status GrpcDispatcherImpl::Start() { return impl_.Start(); }
void GrpcDispatcherImpl::Stop() { impl_.Stop(); }
size_t GrpcDispatcherImpl::NumActiveIterations() {
  return impl_.NumActiveIterations();
}
DispatcherStateExport GrpcDispatcherImpl::ExportState() const {
  return impl_.ExportState();
}
#define HANDLER(method)                                                   \
  grpc::Status GrpcDispatcherImpl::method(ServerContext* context,         \
                                          const method##Request* request, \
                                          method##Response* response) {   \
    return ToGrpcStatus(impl_.method(request, response));                 \
  }
HANDLER(WorkerHeartbeat);
HANDLER(WorkerUpdate);
HANDLER(GetDatasetDef);
HANDLER(GetSplit);
HANDLER(GetVersion);
HANDLER(GetOrRegisterDataset);
HANDLER(ReleaseIterationClient);
HANDLER(MaybeRemoveTask);
HANDLER(GetOrCreateJob);
HANDLER(GetOrCreateIteration);
HANDLER(ClientHeartbeat);
HANDLER(GetWorkers);
HANDLER(GetDataServiceMetadata);
HANDLER(GetDataServiceConfig);
HANDLER(Snapshot);
HANDLER(GetSnapshotSplit);
HANDLER(GetSnapshotStreams);
HANDLER(DisableCompressionAtRuntime);
#undef HANDLER
}  
}  