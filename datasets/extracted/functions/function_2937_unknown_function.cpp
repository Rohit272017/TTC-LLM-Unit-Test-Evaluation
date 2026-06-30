#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_C_API_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_PLUGIN_C_API_H_
#include <cstddef>
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "xla/c/c_api_decl.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#define TFNPD_MAJOR 0
#define TFNPD_MINOR 0
#define TFNPD_PATCH 1
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TFNPD_DeviceEvent TFNPD_DeviceEvent;
typedef TFNPD_DeviceEvent* TFNPD_NewDeviceEvent();
typedef void TFNPD_DeviceEventAwait(TFNPD_DeviceEvent* event,
                                    TF_Status* status);
typedef bool TFNPD_DeviceEventIsReady(TFNPD_DeviceEvent* event);
typedef void TFNPD_DeviceEventAndThen(TFNPD_DeviceEvent* event,
                                      void (*callback)(void*),
                                      void* callback_arg);
typedef void TFNPD_DeviceEventDelete(TFNPD_DeviceEvent* event);
typedef struct TFNPD_DeviceAllocator TFNPD_DeviceAllocator;
typedef TFNPD_DeviceAllocator* TFNPD_DeviceAllocatorCreate(int device_ordinal);
typedef void* TFNPD_DeviceAllocateRaw(TFNPD_DeviceAllocator* allocator,
                                      size_t alignment, size_t num_bytes);
typedef void TFNPD_DeviceDeallocateRaw(TFNPD_DeviceAllocator* allocator,
                                       void* ptr);
typedef TF_StringView TFNPD_DeviceAllocatorName(
    TFNPD_DeviceAllocator* allocator);
typedef bool TFNPD_DeviceAllocatorAllocatesOpaqueHandle(
    TFNPD_DeviceAllocator* allocator);
typedef void TFNPD_DeviceAllocatorDelete(TFNPD_DeviceAllocator* allocator);
typedef struct TFNPD_DeviceContext TFNPD_DeviceContext;
typedef TFNPD_DeviceContext* TFNPD_DeviceContextCreate(int device_ordinal);
typedef TFNPD_DeviceEvent* TFNPD_DeviceTensorToHostTensor(
    TFNPD_DeviceContext* device_context, const TF_Tensor* device_tensor,
    TF_Tensor* cpu_tensor, TF_Status* status);
typedef TFNPD_DeviceEvent* TFNPD_HostTensorToDeviceTensor(
    TFNPD_DeviceContext* device_context, const TF_Tensor* cpu_tensor,
    TF_Tensor* device_tensor, TF_Status* status);
typedef TFNPD_DeviceEvent* TFNPD_SameDeviceTensorCopy(
    TFNPD_DeviceContext* context);
typedef PJRT_Buffer* TFNPD_SameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                                    PJRT_Client* c_client,
                                                    TF_Status* status);
typedef void TFNPD_DeviceContextDelete(TFNPD_DeviceContext* context);
typedef void TFNPD_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* serialized_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* serialized_device_shape,
    TF_Status* tf_status);
typedef int32_t TFNPD_GetDeviceCount(TF_Status* status);
typedef void TFNPD_InitPluginInternalDeviceStates(TF_Status* status);
#define TFNPD_API_STRUCT_FN(fn_type) fn_type* fn_type
typedef struct {
  size_t struct_size;
  void* priv;
  TFNPD_API_STRUCT_FN(TFNPD_NewDeviceEvent);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceEventAwait);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceEventIsReady);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceEventAndThen);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceEventDelete);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceAllocatorCreate);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceAllocateRaw);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceDeallocateRaw);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceAllocatorName);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceAllocatorAllocatesOpaqueHandle);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceAllocatorDelete);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceContextCreate);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceContextDelete);
  TFNPD_API_STRUCT_FN(TFNPD_DeviceTensorToHostTensor);
  TFNPD_API_STRUCT_FN(TFNPD_HostTensorToDeviceTensor);
  TFNPD_API_STRUCT_FN(TFNPD_SameDeviceTensorCopy);
  TFNPD_API_STRUCT_FN(TFNPD_SameDevicePjRtBufferCopy);
  TFNPD_API_STRUCT_FN(TFNPD_XlaShapeToDeviceShapeRepresentation);
  TFNPD_API_STRUCT_FN(TFNPD_GetDeviceCount);
  TFNPD_API_STRUCT_FN(TFNPD_InitPluginInternalDeviceStates);
} TFNPD_Api;
const size_t TFNPD_Api_STRUCT_SIZE =
    TF_OFFSET_OF_END(TFNPD_Api, TFNPD_InitPluginInternalDeviceStates);
#undef TFNPD_API_STRUCT_FN
typedef struct TFNPD_PluginParams {
  size_t struct_size;
  void* ext;  
  const char* device_type;              
  const char* compilation_device_name;  
  int32_t priority;                     
  bool is_pluggable_device;         
  bool use_pjrt_on_demand_compile;  
} TFNPD_PluginParams;
const size_t TFNPD_PLUGIN_PARAMS_STRUCT_SIZE =
    TF_OFFSET_OF_END(TFNPD_PluginParams, is_pluggable_device);
const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params,
                                  TF_Status* tf_status);
#if defined(__cplusplus)
}  
#endif  
#endif  