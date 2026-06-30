#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_H_
#include <stdint.h>
#include "tensorflow/c/c_api_macros.h"  
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TF_DeviceContext TF_DeviceContext;
typedef struct TFDevice_AllocatorAttributes {
  uint32_t value;
  int32_t scope_id;
} TFDevice_AllocatorAttributes;
typedef struct TFE_CancellationManager TFE_CancellationManager;
typedef struct TF_RendezvousArgsStruct {
  TF_DeviceContext* device_context;
  TFDevice_AllocatorAttributes alloc_attrs;
  TFE_CancellationManager* cancellation_manager;
} TF_RendezvousArgsStruct;
typedef struct TF_RendezvousParsedKey {
  char* full_key;
  uint32_t full_key_size;
} TF_RendezvousParsedKey;
typedef struct TF_RendezvousSend_Params {
  const TF_RendezvousParsedKey* key;
  const TF_RendezvousArgsStruct* args;
  TF_Tensor* tensor;
  bool is_dead;
  TF_Status* status;  
} TF_RendezvousSend_Params;
typedef void (*TF_RendezvousSend_Function)(void*, TF_RendezvousSend_Params*);
typedef struct TF_RendezvousDoneCallback_Params {
  void* context;
  const TF_Status* status;
  const TF_Tensor* tensor;
  bool is_dead;
} TF_RendezvousDoneCallback_Params;
typedef void (*TF_RendezvousDoneCallback_Function)(
    void*, TF_RendezvousDoneCallback_Params*);
typedef struct TF_RendezvousDoneCallbackImpl {
  void* context;
  TF_RendezvousDoneCallback_Function callback;
} TF_RendezvousDoneCallbackImpl;
typedef struct TF_RendezvousAsyncRecv_Params {
  void* context;
  const TF_RendezvousParsedKey* key;
  const TF_RendezvousArgsStruct* args;
  TF_RendezvousDoneCallbackImpl on_done;
} TF_RendezvousAsyncRecv_Params;
typedef void (*TF_RendezvousAsyncRecv_Function)(void*,
                                                TF_RendezvousAsyncRecv_Params*);
typedef void (*TF_RendezvousStartAbort_Function)(void* context,
                                                 const TF_Status*);
typedef struct TF_RendezvousThunk {
  void* rendezvous;
  TF_RendezvousSend_Function send_func;
  TF_RendezvousAsyncRecv_Function async_recv_func;
  TF_RendezvousStartAbort_Function start_abort_func;
} TF_RendezvousThunk;
#ifdef __cplusplus
}  
#endif
#endif  