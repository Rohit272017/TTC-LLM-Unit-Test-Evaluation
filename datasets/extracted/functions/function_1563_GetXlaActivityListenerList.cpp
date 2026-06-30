#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"
namespace tensorflow {
namespace {
struct XlaActivityListenerList {
  absl::Mutex mutex;
  std::vector<std::unique_ptr<XlaActivityListener>> listeners
      TF_GUARDED_BY(mutex);
};
void FlushAllListeners();
XlaActivityListenerList* GetXlaActivityListenerList() {
  static XlaActivityListenerList* listener_list = new XlaActivityListenerList;
  static int unused = std::atexit(FlushAllListeners);
  (void)unused;
  return listener_list;
}
template <typename FnTy>
Status ForEachListener(FnTy fn) {
  XlaActivityListenerList* listener_list = GetXlaActivityListenerList();
  absl::ReaderMutexLock reader_lock(&listener_list->mutex);
  for (const std::unique_ptr<XlaActivityListener>& listener :
       listener_list->listeners) {
    TF_RETURN_IF_ERROR(fn(listener.get()));
  }
  return absl::OkStatus();
}
void FlushAllListeners() {
  Status s = ForEachListener([](XlaActivityListener* listener) {
    listener->Flush();
    return absl::OkStatus();
  });
  CHECK(s.ok());
}
}  
Status BroadcastXlaActivity(
    XlaAutoClusteringActivity auto_clustering_activity) {
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(auto_clustering_activity);
  });
}
Status BroadcastXlaActivity(
    XlaJitCompilationActivity jit_compilation_activity) {
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(jit_compilation_activity);
  });
}
Status BroadcastOptimizationRemark(XlaOptimizationRemark optimization_remark) {
  VLOG(2) << "OptimizationRemark: " << optimization_remark.DebugString();
  return ForEachListener([&](XlaActivityListener* listener) {
    return listener->Listen(optimization_remark);
  });
}
Status BroadcastOptimizationRemark(
    XlaOptimizationRemark::Warning optimization_warning,
    string debug_information) {
  XlaOptimizationRemark remark;
  remark.set_warning(optimization_warning);
  remark.set_debug_information(std::move(debug_information));
  return BroadcastOptimizationRemark(std::move(remark));
}
void RegisterXlaActivityListener(
    std::unique_ptr<XlaActivityListener> listener) {
  XlaActivityListenerList* listener_list = GetXlaActivityListenerList();
  absl::WriterMutexLock writer_lock(&listener_list->mutex);
  listener_list->listeners.push_back(std::move(listener));
}
void XlaActivityListener::Flush() {}
XlaActivityListener::~XlaActivityListener() {}
}  