#include "xla/python/outfeed_receiver.h"
#include <sys/types.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/sharding_builder.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"
namespace xla {
int constexpr kOutfeedHeaderWords = 2;
uint32_t constexpr kOutfeedHeaderStart = 271828;
uint32_t constexpr kOutfeedCidShutdown = 0;
class OutfeedData {
 public:
  OutfeedData(ifrt::PjRtDevice* device, uint32_t consumer_id, Shape shape)
      : device_(device),
        consumer_id_(consumer_id),
        shape_(shape),
        literal_(nullptr),
        literal_size_bytes_(0) {}
  ifrt::PjRtDevice* device() { return device_; }
  uint32_t consumer_id() const { return consumer_id_; }
  Shape shape() const { return shape_; }
  std::unique_ptr<Literal> literal() {
    CHECK(literal_);
    return std::move(literal_);
  }
  void SetLiteral(std::unique_ptr<Literal> literal);
  ssize_t literal_size_bytes() const { return literal_size_bytes_; }
  std::string DebugString() const;
 private:
  ifrt::PjRtDevice* device_;
  uint32_t consumer_id_;
  Shape shape_;
  std::unique_ptr<Literal> literal_;
  ssize_t literal_size_bytes_;
};
void OutfeedData::SetLiteral(std::unique_ptr<Literal> literal) {
  literal_ = std::move(literal);
  shape_ = literal_->shape();
  int total_size_bytes = 0;
  ShapeUtil::ForEachSubshape(
      shape_, [&](const Shape& literal_subshape, const ShapeIndex& index) {
        if (!literal_subshape.IsTuple()) {
          total_size_bytes += ShapeUtil::ByteSizeOf(literal_subshape, 8);
        }
      });
  literal_size_bytes_ = total_size_bytes;
}
std::string OutfeedData::DebugString() const {
  return absl::StrFormat("dev=%s; cons=%d; shape=%s", device_->DebugString(),
                         consumer_id_, shape_.ToString());
}
class OutfeedReceiverImpl {
 public:
  OutfeedReceiverImpl(
      OutfeedReceiver::Callback callback,
      absl::Span<ifrt::PjRtClient* const> clients,
      ssize_t max_callback_queue_size_bytes,
      const std::optional<ExecutableBuildOptions>& executable_build_options);
  OutfeedReceiverImpl(const OutfeedReceiverImpl&) = delete;
  OutfeedReceiverImpl& operator=(const OutfeedReceiverImpl&) = delete;
  ~OutfeedReceiverImpl();
  void Start();
  absl::StatusOr<XlaOp> AddOutfeedToBuilder(XlaBuilder* builder, XlaOp token,
                                            uint32_t consumer_id,
                                            std::vector<XlaOp> arrays,
                                            uint32_t device_idx);
  absl::Status RegisterOutfeed(uint32_t consumer_id, const Shape& shape);
 private:
  bool CallbackQueueHasSpace() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return callback_queue_size_bytes_ < max_callback_queue_size_bytes_;
  }
  bool ShutdownDone() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return (num_working_callback_threads_ == 0 && num_listening_threads_ == 0);
  }
  void CallbackThreadLoop(int device_idx);
  void DeviceListenerThreadLoop(int device_idx);
  absl::Status SendShutdownOutfeedHeader(int device_idx);
  absl::StatusOr<std::unique_ptr<Literal>> ReceiveRawFromOutfeed(
      ifrt::PjRtDevice* device, const Shape& shape);
  void EnqueueReceivedData(uint32_t device_idx,
                           std::unique_ptr<OutfeedData> received)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void Shutdown();
  OutfeedReceiver::Callback callback_;
  std::vector<ifrt::PjRtDevice*> devices_;
  uint64_t max_callback_queue_size_bytes_;
  std::optional<ExecutableBuildOptions> executable_build_options_;
  absl::Mutex mu_;
  absl::flat_hash_map<uint32_t, Shape> shape_registry_ ABSL_GUARDED_BY(mu_);
  uint64_t callback_queue_size_bytes_ ABSL_GUARDED_BY(mu_);
  int num_listening_threads_ ABSL_GUARDED_BY(mu_);
  bool shutdown_started_ ABSL_GUARDED_BY(mu_);
  int num_working_callback_threads_ ABSL_GUARDED_BY(mu_);
  std::vector<std::queue<std::unique_ptr<OutfeedData>>> callback_queues_
      ABSL_GUARDED_BY(mu_);
  std::unique_ptr<tsl::thread::ThreadPool> threads_;
};
OutfeedReceiverImpl::OutfeedReceiverImpl(
    OutfeedReceiver::Callback callback,
    absl::Span<ifrt::PjRtClient* const> clients,
    ssize_t max_callback_queue_size_bytes,
    const std::optional<ExecutableBuildOptions>& executable_build_options)
    : executable_build_options_(executable_build_options) {
  callback_ = callback;
  max_callback_queue_size_bytes_ = max_callback_queue_size_bytes;
  for (const auto& client : clients) {
    for (auto device : client->addressable_devices()) {
      devices_.push_back(tensorflow::down_cast<ifrt::PjRtDevice*>(device));
    }
  }
  CHECK_GT(devices_.size(), 0);
  callback_queues_ =
      std::vector<std::queue<std::unique_ptr<OutfeedData>>>(devices_.size());
  callback_queue_size_bytes_ = 0;
  num_listening_threads_ = 0;
  num_working_callback_threads_ = 0;
  shutdown_started_ = false;
}
void OutfeedReceiverImpl::Start() {
  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
  }
  int num_threads = 2 * devices_.size();
  threads_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "outfeed_receiver", num_threads);
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    threads_->Schedule(
        [this, device_idx]() { DeviceListenerThreadLoop(device_idx); });
    threads_->Schedule(
        [this, device_idx]() { CallbackThreadLoop(device_idx); });
  }
}
void OutfeedReceiverImpl::Shutdown() {
  VLOG(2) << "Shutdown start";
  {
    absl::MutexLock lock(&mu_);
    CHECK(!shutdown_started_);
    shutdown_started_ = true;
  }
  for (int device_idx = 0; device_idx < devices_.size(); ++device_idx) {
    TF_CHECK_OK(SendShutdownOutfeedHeader(device_idx));
  }
  VLOG(2) << "Shutdown waiting for listening and callback threads to stop";
  absl::MutexLock lock(&mu_);
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::ShutdownDone));
  VLOG(2) << "Shutdown done";
}
OutfeedReceiverImpl::~OutfeedReceiverImpl() {
  VLOG(2) << "~OutfeedReceiverImpl";
  Shutdown();
}
void OutfeedReceiverImpl::DeviceListenerThreadLoop(int device_idx) {
  {
    absl::MutexLock lock(&mu_);
    ++num_listening_threads_;
  }
  ifrt::PjRtDevice* device = devices_[device_idx];
  while (true) {
    Shape header_shape = ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords});
    std::unique_ptr<Literal> header =
        ReceiveRawFromOutfeed(device, header_shape).value();
    absl::Span<uint32_t> header_data = header->data<uint32_t>();
    CHECK_EQ(header_data.size(), kOutfeedHeaderWords);
    CHECK_EQ(header_data[0], kOutfeedHeaderStart);
    uint32_t consumer_id = header_data[1];
    Shape shape;
    {
      absl::MutexLock lock(&mu_);
      auto registered_shape = shape_registry_.find(consumer_id);
      if (registered_shape == shape_registry_.end()) {
        LOG(FATAL)
            << "[" << device->DebugString()
            << "] Cannot find registered shape for consumer ID " << consumer_id
            << ". Perhaps the code was compiled with a different instance "
            << "of OutfeedReceiver.";
      }
      shape = registered_shape->second;
    }
    auto received = std::make_unique<OutfeedData>(device, consumer_id, shape);
    VLOG(2) << "Listener received header " << received->DebugString();
    if (consumer_id == kOutfeedCidShutdown) {
      VLOG(2) << "[" << device->DebugString()
              << "] Listener received shutdown header";
      absl::MutexLock lock(&mu_);
      --num_listening_threads_;
      VLOG(2) << "[" << device->DebugString() << "] Enqueue shutdown callback";
      EnqueueReceivedData(device_idx, std::move(received));
      return;
    }
    std::unique_ptr<Literal> data =
        ReceiveRawFromOutfeed(device, shape).value();
    received->SetLiteral(std::move(data));
    absl::MutexLock lock(&mu_);
    EnqueueReceivedData(device_idx, std::move(received));
  }
}
void OutfeedReceiverImpl::EnqueueReceivedData(
    uint32_t device_idx, std::unique_ptr<OutfeedData> received)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  mu_.Await(absl::Condition(this, &OutfeedReceiverImpl::CallbackQueueHasSpace));
  ssize_t literal_size_bytes = received->literal_size_bytes();
  callback_queue_size_bytes_ += literal_size_bytes;
  VLOG(2) << "Listener enqueues data " << received->DebugString() << " of size "
          << literal_size_bytes << " bytes; "
          << (1 + callback_queues_[device_idx].size())
          << " callbacks in queue of total size " << callback_queue_size_bytes_
          << " bytes.\n";
  callback_queues_[device_idx].push(std::move(received));
}
absl::StatusOr<std::unique_ptr<Literal>>
OutfeedReceiverImpl::ReceiveRawFromOutfeed(ifrt::PjRtDevice* device,
                                           const Shape& shape) {
  auto literal = std::make_unique<Literal>(shape);
  TF_RETURN_IF_ERROR(
      device->client()->TransferFromOutfeed(device, literal.get()));
  return literal;
}
void OutfeedReceiverImpl::CallbackThreadLoop(int device_idx) {
  const ifrt::PjRtDevice* device = devices_[device_idx];
  {
    absl::MutexLock lock(&mu_);
    num_working_callback_threads_++;
  }
  while (true) {
    std::unique_ptr<OutfeedData> received;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(
          +[](std::queue<std::unique_ptr<OutfeedData>>* queue) {
            return !queue->empty();
          },
          &callback_queues_[device_idx]));
      received = std::move(callback_queues_[device_idx].front());
      callback_queues_[device_idx].pop();
      callback_queue_size_bytes_ -= received->literal_size_bytes();
      VLOG(2) << "[" << device->DebugString() << "] Dequeued callback for "
              << received->DebugString() << "; "
              << callback_queues_[device_idx].size()
              << " callbacks in queue of total size "
              << callback_queue_size_bytes_ << " bytes.\n";
    }
    if (received->consumer_id() == kOutfeedCidShutdown) {
      VLOG(2) << "[" << device->DebugString()
              << "] Callback loop received shutdown signal";
      {
        absl::MutexLock lock(&mu_);
        CHECK(callback_queues_[device_idx].empty());
        --num_working_callback_threads_;
      }
      VLOG(2) << "[" << device->DebugString() << "] Callback loop done";
      return;
    }
    {
      tsl::profiler::TraceMe traceme("OutfeedReceiver::Callback");
      callback_(received->device(), received->consumer_id(),
                received->literal());
    }
  }
}
absl::Status OutfeedReceiverImpl::SendShutdownOutfeedHeader(int device_idx) {
  const ifrt::PjRtDevice* device = devices_[device_idx];
  constexpr int consumer_id = kOutfeedCidShutdown;
  VLOG(2) << "[" << device->DebugString()
          << "] SendSpecialHeader cons=" << consumer_id;
  XlaBuilder builder(
      absl::StrFormat("special_outfeed_header_%d_%d", consumer_id, device_idx));
  XlaOp cst_operand = xla::ConstantR0<int32_t>(&builder, 0);
  XlaOp outfeed =
      AddOutfeedToBuilder(&builder, CreateToken(&builder), consumer_id, {}, 0)
          .value();
  XlaOp add_dep = xla::internal::XlaBuilderFriend::BuildAddDependency(
      &builder, cst_operand, outfeed, ShapeUtil::MakeScalarShape(S32));
  XlaComputation computation = builder.Build(add_dep).value();
  CompileOptions compile_options;
  if (executable_build_options_) {
    compile_options.executable_build_options = *executable_build_options_;
  }
  compile_options.executable_build_options.set_num_replicas(1);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device->Id().value();
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      devices_[device_idx]->client()->pjrt_client()->Compile(
                          computation, std::move(compile_options)));
  ExecuteOptions execute_options;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers,
      executable->Execute({{}}, execute_options));
  return absl::OkStatus();
}
absl::Status OutfeedReceiverImpl::RegisterOutfeed(uint32_t consumer_id,
                                                  const Shape& shape) {
  VLOG(2) << "RegisterShape cons=" << consumer_id
          << "; shape=" << shape.ToString();
  {
    absl::MutexLock lock(&mu_);
    auto found = shape_registry_.find(consumer_id);
    if (found != shape_registry_.end()) {
      if (!ShapeUtil::Equal(shape, found->second)) {
        return InvalidArgument(
            "Shape %s does not match previous shape %s used "
            "for consumer id %d",
            shape.DebugString(), found->second.DebugString(), consumer_id);
      }
    } else {
      shape_registry_.insert({consumer_id, shape});
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<XlaOp> OutfeedReceiverImpl::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays, uint32_t device_idx) {
  XlaOp data = Tuple(builder, std::move(arrays));
  Shape shape_with_layout = builder->GetShape(data).value();
  ShapeUtil::ForEachMutableSubshape(
      &shape_with_layout, [](Shape* subshape, const ShapeIndex&) {
        if (!subshape->has_layout()) {
          LayoutUtil::SetToDefaultLayout(subshape);
        }
      });
  TF_RETURN_IF_ERROR(RegisterOutfeed(consumer_id, shape_with_layout));
  std::vector<uint32_t> header{kOutfeedHeaderStart, consumer_id};
  XlaOp header_op = ConstantR1<uint32_t>(builder, header);
  builder->SetSharding(sharding_builder::AssignDevice(device_idx));
  token = OutfeedWithToken(
      header_op, token, ShapeUtil::MakeShape(U32, {kOutfeedHeaderWords}), "");
  if (consumer_id != kOutfeedCidShutdown) {
    token = OutfeedWithToken(data, token, shape_with_layout, "");
  }
  builder->ClearSharding();
  return token;
}
OutfeedReceiver::OutfeedReceiver(
    Callback callback, absl::Span<ifrt::PjRtClient* const> clients,
    ssize_t max_callback_queue_size_bytes,
    const std::optional<ExecutableBuildOptions>& executable_build_options) {
  p_impl_ = std::make_unique<OutfeedReceiverImpl>(callback, clients,
                                                  max_callback_queue_size_bytes,
                                                  executable_build_options);
}
OutfeedReceiver::~OutfeedReceiver() = default;
void OutfeedReceiver::Start() { p_impl_->Start(); }
absl::StatusOr<XlaOp> OutfeedReceiver::AddOutfeedToBuilder(
    XlaBuilder* builder, XlaOp token, uint32_t consumer_id,
    std::vector<XlaOp> arrays, uint32_t device_idx) {
  if (consumer_id == kOutfeedCidShutdown) {
    return InvalidArgument("Consumer ID cannot be a reserved value: %d",
                           consumer_id);
  }
  return p_impl_->AddOutfeedToBuilder(builder, token, consumer_id, arrays,
                                      device_idx);
}
absl::Status OutfeedReceiver::RegisterOutfeed(uint32_t consumer_id,
                                              const Shape& shape) {
  if (consumer_id == kOutfeedCidShutdown) {
    return InvalidArgument("Consumer ID cannot be a reserved value: %d",
                           consumer_id);
  }
  return p_impl_->RegisterOutfeed(consumer_id, shape);
}
}  