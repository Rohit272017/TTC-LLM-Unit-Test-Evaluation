#include "tensorstore/kvstore/gcs/gcs_testbench.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"  
#include "grpcpp/client_context.h"  
#include "grpcpp/create_channel.h"  
#include "grpcpp/security/credentials.h"  
#include "grpcpp/support/status.h"  
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/os/subprocess.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"
ABSL_FLAG(std::string, testbench_binary, "",
          "Path to the gcs storage-testbench rest_server");
namespace gcs_testbench {
using ::google::storage::v2::Storage;
using ::tensorstore::internal::GrpcStatusToAbslStatus;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::IssueRequestOptions;
using ::tensorstore::transport_test_utils::TryPickUnusedPort;
StorageTestbench::StorageTestbench() = default;
std::string StorageTestbench::http_address() {
  return absl::StrFormat("localhost:%d", http_port);
}
std::string StorageTestbench::grpc_address() {
  return absl::StrFormat("localhost:%d", grpc_port);
}
void StorageTestbench::SpawnProcess() {
  if (running) return;
  const auto start_child = [&] {
    http_port = TryPickUnusedPort().value_or(0);
    ABSL_CHECK(http_port > 0);
    ABSL_LOG(INFO) << "Spawning testbench: http:
    {
      SubprocessOptions options{absl::GetFlag(FLAGS_testbench_binary),
                                {absl::StrFormat("--port=%d", http_port)}};
      TENSORSTORE_CHECK_OK_AND_ASSIGN(child, SpawnSubprocess(options));
    }
  };
  start_child();
  for (auto deadline = absl::Now() + absl::Seconds(30);;) {
    absl::SleepFor(absl::Milliseconds(200));
    if (!absl::IsUnavailable(child->Join(false).status())) {
      start_child();
    }
    auto result =
        GetDefaultHttpTransport()
            ->IssueRequest(
                HttpRequestBuilder(
                    "GET", absl::StrFormat("http:
                                           http_port))
                    .BuildRequest(),
                IssueRequestOptions()
                    .SetRequestTimeout(absl::Seconds(15))
                    .SetConnectTimeout(absl::Seconds(15)))
            .result();
    if (result.ok()) {
      if (result->status_code != 200) {
        ABSL_LOG(ERROR) << "Failed to start grpc server: " << *result;
      } else if (!absl::SimpleAtoi(result->payload.Flatten(), &grpc_port)) {
        ABSL_LOG(ERROR) << "Unexpected response from start_grpc: " << *result;
      } else {
        break;
      }
    } else {
      ABSL_LOG(ERROR) << "Failed to start grpc server: " << result.status();
    }
    if (absl::Now() < deadline && absl::IsUnavailable(result.status())) {
      continue;
    }
    ABSL_LOG(FATAL) << "Failed to start testbench: " << result.status();
  }
  running = true;
}
StorageTestbench::~StorageTestbench() {
  if (child) {
    child->Kill().IgnoreError();
    auto join_result = child->Join();
    if (!join_result.ok()) {
      ABSL_LOG(ERROR) << "Joining storage_testbench subprocess failed: "
                      << join_result.status();
    }
  }
}
absl::Status StorageTestbench::CreateBucket(std::string grpc_endpoint,
                                            std::string bucket) {
  google::storage::v2::CreateBucketRequest bucket_request =
      tensorstore::ParseTextProtoOrDie(R"pb(
        parent: 'projects/12345'
        bucket: { location: 'US' storage_class: 'STANDARD' }
        bucket_id: 'bucket'
        predefined_acl: 'publicReadWrite'
        predefined_default_object_acl: 'publicReadWrite'
      )pb");
  bucket_request.set_bucket_id(bucket);
  google::storage::v2::Bucket bucket_response;
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      grpc_endpoint, grpc::InsecureChannelCredentials());  
  if (!channel->WaitForConnected(
          absl::ToChronoTime(absl::Now() + absl::Milliseconds(100)))) {
    ABSL_LOG(WARNING) << "Failed to connect to grpc endpoint after 100ms: "
                      << grpc_endpoint;
  }
  auto stub = Storage::NewStub(std::move(channel));
  grpc::ClientContext client_context;
  grpc::Status status =
      stub->CreateBucket(&client_context, bucket_request, &bucket_response);
  return GrpcStatusToAbslStatus(status);
}
}  