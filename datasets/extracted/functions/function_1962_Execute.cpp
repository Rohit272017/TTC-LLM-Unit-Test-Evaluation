#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_THREADPOOL_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_THREADPOOL_H_
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#ifdef TFLITE_WITH_RUY
#include "ruy/context.h"  
#include "ruy/thread_pool.h"  
#else
#include "public/gemmlowp.h"
#endif
namespace tflite {
namespace cpu_backend_threadpool {
#ifdef TFLITE_WITH_RUY
using Task = ruy::Task;
template <typename TaskType>
void Execute(int tasks_count, TaskType* tasks,
             CpuBackendContext* cpu_backend_context) {
  TFLITE_DCHECK_LE(tasks_count, cpu_backend_context->max_num_threads());
  cpu_backend_context->ruy_context()->mutable_thread_pool()->Execute(
      tasks_count, tasks);
}
#else  
using Task = gemmlowp::Task;
template <typename TaskType>
void Execute(int tasks_count, TaskType* tasks,
             CpuBackendContext* cpu_backend_context) {
  TFLITE_DCHECK_LE(tasks_count, cpu_backend_context->max_num_threads());
  cpu_backend_context->gemmlowp_context()->workers_pool()->Execute(tasks_count,
                                                                   tasks);
}
#endif
}  
}  
#endif  