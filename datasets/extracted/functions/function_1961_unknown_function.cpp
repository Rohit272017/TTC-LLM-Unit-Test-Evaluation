#ifndef XLA_PJRT_C_PJRT_C_API_H_
#define XLA_PJRT_C_PJRT_C_API_H_
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define PJRT_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)
#define PJRT_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                        \
  enum { sname##_STRUCT_SIZE = PJRT_STRUCT_SIZE(sname, last_field) }
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  PJRT_Extension_Type_Gpu_Custom_Call = 0,
  PJRT_Extension_Type_Profiler,
  PJRT_Extension_Type_Custom_Partitioner,
  PJRT_Extension_Type_Stream,
  PJRT_Extension_Type_Layouts,
  PJRT_Extension_Type_FFI,
} PJRT_Extension_Type;
typedef struct PJRT_Extension_Base {
  size_t struct_size;
  PJRT_Extension_Type type;
  struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Extension_Base, next);
#define PJRT_API_MAJOR 0
#define PJRT_API_MINOR 55
struct PJRT_Api_Version {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  int major_version;  
  int minor_version;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Api_Version, minor_version);
typedef struct PJRT_Error PJRT_Error;
struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Destroy_Args, error);
typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);
struct PJRT_Error_Message_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  const char* message;  
  size_t message_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Message_Args, message_size);
typedef void PJRT_Error_Message(PJRT_Error_Message_Args* args);
typedef enum {
  PJRT_Error_Code_CANCELLED = 1,
  PJRT_Error_Code_UNKNOWN = 2,
  PJRT_Error_Code_INVALID_ARGUMENT = 3,
  PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
  PJRT_Error_Code_NOT_FOUND = 5,
  PJRT_Error_Code_ALREADY_EXISTS = 6,
  PJRT_Error_Code_PERMISSION_DENIED = 7,
  PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
  PJRT_Error_Code_FAILED_PRECONDITION = 9,
  PJRT_Error_Code_ABORTED = 10,
  PJRT_Error_Code_OUT_OF_RANGE = 11,
  PJRT_Error_Code_UNIMPLEMENTED = 12,
  PJRT_Error_Code_INTERNAL = 13,
  PJRT_Error_Code_UNAVAILABLE = 14,
  PJRT_Error_Code_DATA_LOSS = 15,
  PJRT_Error_Code_UNAUTHENTICATED = 16
} PJRT_Error_Code;
struct PJRT_Error_GetCode_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  PJRT_Error_Code code;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_GetCode_Args, code);
typedef PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args);
typedef PJRT_Error* (*PJRT_CallbackError)(PJRT_Error_Code code,
                                          const char* message,
                                          size_t message_size);
typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64,
  PJRT_NamedValue_kInt64List,
  PJRT_NamedValue_kFloat,
  PJRT_NamedValue_kBool,
} PJRT_NamedValue_Type;
struct PJRT_NamedValue {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  size_t value_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_NamedValue, value_size);
struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Plugin_Initialize_Args, extension_start);
typedef PJRT_Error* PJRT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args);
struct PJRT_Plugin_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_NamedValue* attributes;  
  size_t num_attributes;              
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Plugin_Attributes_Args, attributes);
typedef PJRT_Error* PJRT_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args);
typedef struct PJRT_Event PJRT_Event;
struct PJRT_Event_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Destroy_Args, event);
typedef PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args);
struct PJRT_Event_IsReady_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
  bool is_ready;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_IsReady_Args, is_ready);
typedef PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args);
struct PJRT_Event_Error_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Error_Args, event);
typedef PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args);
struct PJRT_Event_Await_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Await_Args, event);
typedef PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args);
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);
struct PJRT_Event_OnReady_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
  PJRT_Event_OnReadyCallback callback;
  void* user_arg;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_OnReady_Args, user_arg);
typedef PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args);
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_TopologyDescription PJRT_TopologyDescription;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef void (*PJRT_KeyValueGetCallback_ValueDeleter)(char* value);
struct PJRT_KeyValueGetCallback_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* key;
  size_t key_size;
  int timeout_in_ms;
  PJRT_CallbackError* callback_error;
  void* user_arg;
  char* value;        
  size_t value_size;  
  PJRT_KeyValueGetCallback_ValueDeleter value_deleter_callback;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_KeyValueGetCallback_Args,
                          value_deleter_callback);
typedef PJRT_Error* (*PJRT_KeyValueGetCallback)(
    PJRT_KeyValueGetCallback_Args* args);
struct PJRT_KeyValuePutCallback_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* key;
  size_t key_size;
  const char* value;
  size_t value_size;
  PJRT_CallbackError* callback_error;
  void* user_arg;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_KeyValuePutCallback_Args, user_arg);
typedef PJRT_Error* (*PJRT_KeyValuePutCallback)(
    PJRT_KeyValuePutCallback_Args* args);
struct PJRT_Client_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_NamedValue* create_options;
  size_t num_options;
  PJRT_KeyValueGetCallback kv_get_callback;
  void* kv_get_user_arg;
  PJRT_KeyValuePutCallback kv_put_callback;
  void* kv_put_user_arg;
  PJRT_Client* client;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Create_Args, client);
typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);
struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Destroy_Args, client);
typedef PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);
struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_name;  
  size_t platform_name_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_PlatformName_Args, platform_name_size);
typedef PJRT_Error* PJRT_Client_PlatformName(
    PJRT_Client_PlatformName_Args* args);
struct PJRT_Client_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int process_index;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_ProcessIndex_Args, process_index);
typedef PJRT_Error* PJRT_Client_ProcessIndex(
    PJRT_Client_ProcessIndex_Args* args);
struct PJRT_Client_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_version;  
  size_t platform_version_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_PlatformVersion_Args,
                          platform_version_size);
typedef PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args);
struct PJRT_Client_TopologyDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_TopologyDescription* topology;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_TopologyDescription_Args, topology);
typedef PJRT_Error* PJRT_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args* args);
struct PJRT_Client_Devices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* devices;  
  size_t num_devices;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Devices_Args, num_devices);
typedef PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);
struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* addressable_devices;  
  size_t num_addressable_devices;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_AddressableDevices_Args,
                          num_addressable_devices);
typedef PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args);
struct PJRT_Client_LookupDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int id;
  PJRT_Device* device;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupDevice_Args, device);
typedef PJRT_Error* PJRT_Client_LookupDevice(
    PJRT_Client_LookupDevice_Args* args);
struct PJRT_Client_LookupAddressableDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int local_hardware_id;
  PJRT_Device* addressable_device;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupAddressableDevice_Args,
                          addressable_device);
typedef PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args);
struct PJRT_Client_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Memory* const* addressable_memories;  
  size_t num_addressable_memories;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_AddressableMemories_Args,
                          num_addressable_memories);
typedef PJRT_Error* PJRT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args);
struct PJRT_Program {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  char* code;  
  size_t code_size;
  const char* format;
  size_t format_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Program, format_size);
struct PJRT_Client_Compile_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const PJRT_Program* program;
  const char* compile_options;
  size_t compile_options_size;
  PJRT_LoadedExecutable* executable;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Compile_Args, executable);
typedef PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args);
struct PJRT_Client_DefaultDeviceAssignment_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int num_replicas;
  int num_partitions;
  size_t default_assignment_size;
  int* default_assignment;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_DefaultDeviceAssignment_Args,
                          default_assignment);
typedef PJRT_Error* PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args);
typedef enum {
  PJRT_Buffer_Type_INVALID,
  PJRT_Buffer_Type_PRED,
  PJRT_Buffer_Type_S8,
  PJRT_Buffer_Type_S16,
  PJRT_Buffer_Type_S32,
  PJRT_Buffer_Type_S64,
  PJRT_Buffer_Type_U8,
  PJRT_Buffer_Type_U16,
  PJRT_Buffer_Type_U32,
  PJRT_Buffer_Type_U64,
  PJRT_Buffer_Type_F16,
  PJRT_Buffer_Type_F32,
  PJRT_Buffer_Type_F64,
  PJRT_Buffer_Type_BF16,
  PJRT_Buffer_Type_C64,
  PJRT_Buffer_Type_C128,
  PJRT_Buffer_Type_F8E5M2,
  PJRT_Buffer_Type_F8E4M3FN,
  PJRT_Buffer_Type_F8E4M3B11FNUZ,
  PJRT_Buffer_Type_F8E5M2FNUZ,
  PJRT_Buffer_Type_F8E4M3FNUZ,
  PJRT_Buffer_Type_S4,
  PJRT_Buffer_Type_U4,
  PJRT_Buffer_Type_TOKEN,
  PJRT_Buffer_Type_S2,
  PJRT_Buffer_Type_U2,
  PJRT_Buffer_Type_F8E4M3,
  PJRT_Buffer_Type_F8E3M4,
} PJRT_Buffer_Type;
typedef enum {
  PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
  PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
  PJRT_HostBufferSemantics_kImmutableZeroCopy,
  PJRT_HostBufferSemantics_kMutableZeroCopy,
} PJRT_HostBufferSemantics;
typedef enum {
  PJRT_Buffer_MemoryLayout_Type_Tiled = 0,
  PJRT_Buffer_MemoryLayout_Type_Strides,
} PJRT_Buffer_MemoryLayout_Type;
struct PJRT_Buffer_MemoryLayout_Tiled {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const int64_t* minor_to_major;
  size_t minor_to_major_size;
  const int64_t* tile_dims;
  const size_t* tile_dim_sizes;
  size_t num_tiles;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout_Tiled, num_tiles);
struct PJRT_Buffer_MemoryLayout_Strides {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const int64_t* byte_strides;
  size_t num_byte_strides;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout_Strides, num_byte_strides);
struct PJRT_Buffer_MemoryLayout {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  union {
    PJRT_Buffer_MemoryLayout_Tiled tiled;
    PJRT_Buffer_MemoryLayout_Strides strides;
  };
  PJRT_Buffer_MemoryLayout_Type type;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout, type);
struct PJRT_Client_BufferFromHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const void* data;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  const int64_t* byte_strides;
  size_t num_byte_strides;
  PJRT_HostBufferSemantics host_buffer_semantics;
  PJRT_Device* device;
  PJRT_Memory* memory;
  PJRT_Buffer_MemoryLayout* device_layout;
  PJRT_Event* done_with_host_buffer;  
  PJRT_Buffer* buffer;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_BufferFromHostBuffer_Args, buffer);
typedef PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args);
struct PJRT_Client_CreateViewOfDeviceBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  void* device_buffer_ptr;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Buffer_Type element_type;
  PJRT_Buffer_MemoryLayout* layout;
  PJRT_Device* device;
  void (*on_delete_callback)(void* device_buffer_ptr, void* user_arg);
  void* on_delete_callback_arg;
  intptr_t stream;
  PJRT_Buffer* buffer;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_CreateViewOfDeviceBuffer_Args, buffer);
typedef PJRT_Error* PJRT_Client_CreateViewOfDeviceBuffer(
    PJRT_Client_CreateViewOfDeviceBuffer_Args* args);
struct PJRT_DeviceDescription_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int id;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Id_Args, id);
typedef PJRT_Error* PJRT_DeviceDescription_Id(
    PJRT_DeviceDescription_Id_Args* args);
struct PJRT_DeviceDescription_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int process_index;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_ProcessIndex_Args,
                          process_index);
typedef PJRT_Error* PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args);
struct PJRT_DeviceDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  size_t num_attributes;              
  const PJRT_NamedValue* attributes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Attributes_Args, attributes);
typedef PJRT_Error* PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args);
struct PJRT_DeviceDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* device_kind;  
  size_t device_kind_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Kind_Args, device_kind_size);
typedef PJRT_Error* PJRT_DeviceDescription_Kind(
    PJRT_DeviceDescription_Kind_Args* args);
struct PJRT_DeviceDescription_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* debug_string;  
  size_t debug_string_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_DebugString_Args,
                          debug_string_size);
typedef PJRT_Error* PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args);
struct PJRT_DeviceDescription_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* to_string;  
  size_t to_string_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_ToString_Args, to_string_size);
typedef PJRT_Error* PJRT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args);
struct PJRT_Device_GetDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_DeviceDescription* device_description;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_GetDescription_Args, device_description);
typedef PJRT_Error* PJRT_Device_GetDescription(
    PJRT_Device_GetDescription_Args* args);
struct PJRT_Device_IsAddressable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  bool is_addressable;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_IsAddressable_Args, is_addressable);
typedef PJRT_Error* PJRT_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args* args);
struct PJRT_Device_LocalHardwareId_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int local_hardware_id;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_LocalHardwareId_Args, local_hardware_id);
typedef PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args);
struct PJRT_Device_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_Memory* const* memories;  
  size_t num_memories;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_AddressableMemories_Args, num_memories);
typedef PJRT_Error* PJRT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args);
struct PJRT_Device_DefaultMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_Memory* memory;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_DefaultMemory_Args, memory);
typedef PJRT_Error* PJRT_Device_DefaultMemory(
    PJRT_Device_DefaultMemory_Args* args);
struct PJRT_Device_MemoryStats_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int64_t bytes_in_use;  
  int64_t peak_bytes_in_use;      
  bool peak_bytes_in_use_is_set;  
  int64_t num_allocs;      
  bool num_allocs_is_set;  
  int64_t largest_alloc_size;      
  bool largest_alloc_size_is_set;  
  int64_t bytes_limit;      
  bool bytes_limit_is_set;  
  int64_t bytes_reserved;      
  bool bytes_reserved_is_set;  
  int64_t peak_bytes_reserved;      
  bool peak_bytes_reserved_is_set;  
  int64_t bytes_reservable_limit;      
  bool bytes_reservable_limit_is_set;  
  int64_t largest_free_block_bytes;      
  bool largest_free_block_bytes_is_set;  
  int64_t pool_bytes;           
  bool pool_bytes_is_set;       
  int64_t peak_pool_bytes;      
  bool peak_pool_bytes_is_set;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_MemoryStats_Args, peak_pool_bytes_is_set);
typedef PJRT_Error* PJRT_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args);
struct PJRT_Memory_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int id;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Id_Args, id);
typedef PJRT_Error* PJRT_Memory_Id(PJRT_Memory_Id_Args* args);
struct PJRT_Memory_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* kind;  
  size_t kind_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Kind_Args, kind_size);
typedef PJRT_Error* PJRT_Memory_Kind(PJRT_Memory_Kind_Args* args);
struct PJRT_Memory_Kind_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int kind_id;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Kind_Id_Args, kind_id);
typedef PJRT_Error* PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args);
struct PJRT_Memory_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* debug_string;  
  size_t debug_string_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_DebugString_Args, debug_string_size);
typedef PJRT_Error* PJRT_Memory_DebugString(PJRT_Memory_DebugString_Args* args);
struct PJRT_Memory_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* to_string;  
  size_t to_string_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_ToString_Args, to_string_size);
typedef PJRT_Error* PJRT_Memory_ToString(PJRT_Memory_ToString_Args* args);
struct PJRT_Memory_AddressableByDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  PJRT_Device* const* devices;  
  size_t num_devices;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_AddressableByDevices_Args, num_devices);
typedef PJRT_Error* PJRT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args);
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;
struct PJRT_ExecuteContext_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteContext* context;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteContext_Create_Args, context);
typedef PJRT_Error* PJRT_ExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args);
struct PJRT_ExecuteContext_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteContext* context;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteContext_Destroy_Args, context);
typedef PJRT_Error* PJRT_ExecuteContext_Destroy(
    PJRT_ExecuteContext_Destroy_Args* args);
struct PJRT_Executable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Destroy_Args, executable);
typedef PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);
struct PJRT_LoadedExecutable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Destroy_Args, executable);
typedef PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args);
struct PJRT_LoadedExecutable_GetExecutable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* loaded_executable;
  PJRT_Executable* executable;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_GetExecutable_Args, executable);
typedef PJRT_Error* PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args);
struct PJRT_Executable_Name_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  const char* executable_name;  
  size_t executable_name_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Name_Args, executable_name_size);
typedef PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);
struct PJRT_Executable_NumReplicas_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_replicas;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumReplicas_Args, num_replicas);
typedef PJRT_Error* PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args);
struct PJRT_Executable_NumPartitions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_partitions;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumPartitions_Args, num_partitions);
typedef PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args);
struct PJRT_LoadedExecutable_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_Device* const* addressable_devices;  
  size_t num_addressable_devices;           
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_AddressableDevices_Args,
                          num_addressable_devices);
typedef PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args);
struct PJRT_Executable_OptimizedProgram_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  PJRT_Program* program;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OptimizedProgram_Args, program);
typedef PJRT_Error* PJRT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args);
struct PJRT_LoadedExecutable_Delete_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Delete_Args, executable);
typedef PJRT_Error* PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args);
struct PJRT_LoadedExecutable_IsDeleted_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  bool is_deleted;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_IsDeleted_Args, is_deleted);
typedef PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args);
typedef struct PJRT_Chunk {
  void* data;
  size_t size;
  void (*deleter)(void* data, void* deleter_arg);
  void* deleter_arg;
} PJRT_Chunk;
typedef struct PJRT_CopyToDeviceStream PJRT_CopyToDeviceStream;
struct PJRT_TransferMetadata;
typedef PJRT_Error* (*PJRT_SendCallback)(PJRT_Chunk* chunk,
                                         PJRT_CallbackError* callback_error,
                                         size_t total_size_in_bytes, bool done,
                                         void* user_arg);
typedef void (*PJRT_RecvCallback)(PJRT_CopyToDeviceStream* stream,
                                  void* user_arg);
struct PJRT_SendCallbackInfo {
  int64_t channel_id;
  void* user_arg;
  PJRT_SendCallback send_callback;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_SendCallbackInfo, send_callback);
struct PJRT_RecvCallbackInfo {
  int64_t channel_id;
  void* user_arg;
  PJRT_RecvCallback recv_callback;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RecvCallbackInfo, recv_callback);
struct PJRT_ExecuteOptions {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_SendCallbackInfo** send_callbacks;
  PJRT_RecvCallbackInfo** recv_callbacks;
  size_t num_send_ops;
  size_t num_recv_ops;
  int launch_id;
  const int64_t* non_donatable_input_indices;
  size_t num_non_donatable_input_indices;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteOptions, num_non_donatable_input_indices);
struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_ExecuteOptions* options;
  PJRT_Buffer* const* const* argument_lists;
  size_t num_devices;
  size_t num_args;
  PJRT_Buffer** const* output_lists;  
  PJRT_Event** device_complete_events;  
  PJRT_Device* execute_device;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Execute_Args, execute_device);
typedef PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args);
struct PJRT_Executable_NumOutputs_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumOutputs_Args, num_outputs);
typedef PJRT_Error* PJRT_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args* args);
struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  int64_t size_in_bytes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args,
                          size_in_bytes);  
typedef PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);
struct PJRT_Executable_Fingerprint_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  const char* executable_fingerprint;  
  size_t executable_fingerprint_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Fingerprint_Args,
                          executable_fingerprint_size);
typedef PJRT_Error* PJRT_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* args);
struct PJRT_Executable_GetCostAnalysis_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_properties;  
  const PJRT_NamedValue* properties;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_GetCostAnalysis_Args, properties);
typedef PJRT_Error* PJRT_Executable_GetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args* args);
struct PJRT_Executable_GetCompiledMemoryStats_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  int64_t generated_code_size_in_bytes;  
  int64_t argument_size_in_bytes;        
  int64_t output_size_in_bytes;          
  int64_t alias_size_in_bytes;  
  int64_t temp_size_in_bytes;   
  int64_t host_generated_code_size_in_bytes;  
  int64_t host_argument_size_in_bytes;        
  int64_t host_output_size_in_bytes;          
  int64_t host_alias_size_in_bytes;           
  int64_t host_temp_size_in_bytes;            
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_GetCompiledMemoryStats_Args,
                          host_temp_size_in_bytes);
typedef PJRT_Error* PJRT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args);
struct PJRT_Executable_OutputElementTypes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  PJRT_Buffer_Type* output_types;  
  size_t num_output_types;         
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputElementTypes_Args,
                          num_output_types);
typedef PJRT_Error* PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args);
struct PJRT_Executable_OutputDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;
  const int64_t* dims;  
  const size_t* dim_sizes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputDimensions_Args, dim_sizes);
typedef PJRT_Error* PJRT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args);
struct PJRT_Executable_OutputMemoryKinds_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;
  const char* const* memory_kinds;  
  const size_t* memory_kind_sizes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputMemoryKinds_Args,
                          memory_kind_sizes);
typedef PJRT_Error* PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args);
typedef struct PJRT_SerializedExecutable PJRT_SerializedExecutable;
struct PJRT_Executable_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Executable* executable;
  const char* serialized_bytes;  
  size_t serialized_bytes_size;  
  PJRT_SerializedExecutable* serialized_executable;  
  void (*serialized_executable_deleter)(
      PJRT_SerializedExecutable* exec);  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Serialize_Args,
                          serialized_executable_deleter);
typedef PJRT_Error* PJRT_Executable_Serialize(
    PJRT_Executable_Serialize_Args* args);
struct PJRT_Executable_DeserializeAndLoad_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* serialized_executable;
  size_t serialized_executable_size;
  PJRT_LoadedExecutable* loaded_executable;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_DeserializeAndLoad_Args,
                          loaded_executable);
typedef PJRT_Error* PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args);
struct PJRT_LoadedExecutable_Fingerprint_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  const char* executable_fingerprint;  
  size_t executable_fingerprint_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Fingerprint_Args,
                          executable_fingerprint_size);
typedef PJRT_Error* PJRT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args);
struct PJRT_Buffer_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Destroy_Args, buffer);
typedef PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);
struct PJRT_Buffer_ElementType_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Buffer_Type type;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ElementType_Args, type);
typedef PJRT_Error* PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args);
struct PJRT_Buffer_Dimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  const int64_t* dims;  
  size_t num_dims;      
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Dimensions_Args, num_dims);
typedef PJRT_Error* PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args);
struct PJRT_Buffer_UnpaddedDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  const int64_t* unpadded_dims;  
  size_t num_dims;               
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_UnpaddedDimensions_Args, num_dims);
typedef PJRT_Error* PJRT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args);
struct PJRT_Buffer_DynamicDimensionIndices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  const size_t* dynamic_dim_indices;  
  size_t num_dynamic_dims;            
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_DynamicDimensionIndices_Args,
                          num_dynamic_dims);
typedef PJRT_Error* PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args);
struct PJRT_Buffer_GetMemoryLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Buffer_MemoryLayout layout;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_GetMemoryLayout_Args, layout);
typedef PJRT_Error* PJRT_Buffer_GetMemoryLayout(
    PJRT_Buffer_GetMemoryLayout_Args* args);
struct PJRT_Buffer_ToHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* src;
  PJRT_Buffer_MemoryLayout* host_layout;
  void* dst;  
  size_t dst_size;  
  PJRT_Event* event;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ToHostBuffer_Args, event);
typedef PJRT_Error* PJRT_Buffer_ToHostBuffer(
    PJRT_Buffer_ToHostBuffer_Args* args);
struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  size_t on_device_size_in_bytes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OnDeviceSizeInBytes_Args,
                          on_device_size_in_bytes);
typedef PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
struct PJRT_Buffer_Delete_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Delete_Args, buffer);
typedef PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args);
struct PJRT_Buffer_IsDeleted_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_deleted;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsDeleted_Args, is_deleted);
typedef PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);
struct PJRT_Buffer_CopyToDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* dst_device;
  PJRT_Buffer* dst_buffer;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyToDevice_Args, dst_buffer);
typedef PJRT_Error* PJRT_Buffer_CopyToDevice(
    PJRT_Buffer_CopyToDevice_Args* args);
struct PJRT_Buffer_CopyToMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Memory* dst_memory;
  PJRT_Buffer* dst_buffer;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyToMemory_Args, dst_buffer);
typedef PJRT_Error* PJRT_Buffer_CopyToMemory(
    PJRT_Buffer_CopyToMemory_Args* args);
struct PJRT_Buffer_IsOnCpu_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_on_cpu;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsOnCpu_Args, is_on_cpu);
typedef PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);
struct PJRT_Buffer_Device_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* device;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Device_Args, device);
typedef PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args);
struct PJRT_Buffer_Memory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Memory* memory;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Memory_Args, memory);
typedef PJRT_Error* PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args* args);
struct PJRT_Buffer_ReadyEvent_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Event* event;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ReadyEvent_Args, event);
typedef PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);
struct PJRT_Buffer_UnsafePointer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  uintptr_t buffer_pointer;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_UnsafePointer_Args, buffer_pointer);
typedef PJRT_Error* PJRT_Buffer_UnsafePointer(
    PJRT_Buffer_UnsafePointer_Args* args);
struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IncreaseExternalReferenceCount_Args,
                          buffer);
typedef PJRT_Error* PJRT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);
struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                          buffer);
typedef PJRT_Error* PJRT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);
struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  void* device_memory_ptr;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
                          device_memory_ptr);
typedef PJRT_Error* PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);
struct PJRT_CopyToDeviceStream_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_Destroy_Args, stream);
typedef PJRT_Error* PJRT_CopyToDeviceStream_Destroy(
    PJRT_CopyToDeviceStream_Destroy_Args* args);
struct PJRT_CopyToDeviceStream_AddChunk_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  PJRT_Chunk* chunk;
  PJRT_Event* transfer_complete;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_AddChunk_Args,
                          transfer_complete);
typedef PJRT_Error* PJRT_CopyToDeviceStream_AddChunk(
    PJRT_CopyToDeviceStream_AddChunk_Args* args);
struct PJRT_CopyToDeviceStream_TotalBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  int64_t total_bytes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_TotalBytes_Args, total_bytes);
typedef PJRT_Error* PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args* args);
struct PJRT_CopyToDeviceStream_GranuleSize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  int64_t granule_size_in_bytes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_GranuleSize_Args,
                          granule_size_in_bytes);
typedef PJRT_Error* PJRT_CopyToDeviceStream_GranuleSize(
    PJRT_CopyToDeviceStream_GranuleSize_Args* args);
struct PJRT_CopyToDeviceStream_CurrentBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  int64_t current_bytes;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_CurrentBytes_Args,
                          current_bytes);
typedef PJRT_Error* PJRT_CopyToDeviceStream_CurrentBytes(
    PJRT_CopyToDeviceStream_CurrentBytes_Args* args);
struct PJRT_TopologyDescription_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* topology_name;
  size_t topology_name_size;
  const PJRT_NamedValue* create_options;
  size_t num_options;
  PJRT_TopologyDescription* topology;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Create_Args, topology);
typedef PJRT_Error* PJRT_TopologyDescription_Create(
    PJRT_TopologyDescription_Create_Args* args);
struct PJRT_TopologyDescription_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Destroy_Args, topology);
typedef PJRT_Error* PJRT_TopologyDescription_Destroy(
    PJRT_TopologyDescription_Destroy_Args* args);
struct PJRT_TopologyDescription_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const char* platform_version;  
  size_t platform_version_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_PlatformVersion_Args,
                          platform_version_size);
typedef PJRT_Error* PJRT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args);
struct PJRT_TopologyDescription_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const char* platform_name;  
  size_t platform_name_size;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_PlatformName_Args,
                          platform_name_size);
typedef PJRT_Error* PJRT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args);
struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  PJRT_DeviceDescription* const* descriptions;  
  size_t num_descriptions;                      
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_GetDeviceDescriptions_Args,
                          num_descriptions);
typedef PJRT_Error* PJRT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args);
typedef struct PJRT_SerializedTopology PJRT_SerializedTopology;
struct PJRT_TopologyDescription_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const char* serialized_bytes;  
  size_t serialized_bytes_size;  
  PJRT_SerializedTopology* serialized_topology;  
  void (*serialized_topology_deleter)(
      PJRT_SerializedTopology* serialized_topology);  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Serialize_Args,
                          serialized_topology_deleter);
typedef PJRT_Error* PJRT_TopologyDescription_Serialize(
    PJRT_TopologyDescription_Serialize_Args* args);
struct PJRT_TopologyDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const PJRT_NamedValue* attributes;  
  size_t num_attributes;              
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Attributes_Args,
                          num_attributes);
typedef PJRT_Error* PJRT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args* args);
struct PJRT_Compile_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  const PJRT_Program* program;
  const char* compile_options;
  size_t compile_options_size;
  PJRT_Client* client;
  PJRT_Executable* executable;  
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Compile_Args, executable);
typedef PJRT_Error* PJRT_Compile(PJRT_Compile_Args* args);
#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type
typedef struct PJRT_Api {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Api_Version pjrt_api_version;
  _PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_Message);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_GetCode);
  _PJRT_API_STRUCT_FIELD(PJRT_Plugin_Initialize);
  _PJRT_API_STRUCT_FIELD(PJRT_Plugin_Attributes);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_IsReady);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_Error);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_Await);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_OnReady);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformVersion);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Devices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_LookupDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_LookupAddressableDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_AddressableMemories);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Compile);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_DefaultDeviceAssignment);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_BufferFromHostBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Attributes);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Kind);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_DebugString);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_ToString);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_GetDescription);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_IsAddressable);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_LocalHardwareId);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_AddressableMemories);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_DefaultMemory);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_MemoryStats);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Kind);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_DebugString);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_ToString);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_AddressableByDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Name);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumReplicas);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumPartitions);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumOutputs);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_SizeOfGeneratedCodeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_GetCostAnalysis);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputMemoryKinds);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OptimizedProgram);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Serialize);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_GetExecutable);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Execute);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_DeserializeAndLoad);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Fingerprint);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ElementType);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Dimensions);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_UnpaddedDimensions);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_DynamicDimensionIndices);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_GetMemoryLayout);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Device);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Memory);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ToHostBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsOnCpu);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ReadyEvent);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_UnsafePointer);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IncreaseExternalReferenceCount);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_DecreaseExternalReferenceCount);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OpaqueDeviceMemoryDataPointer);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_AddChunk);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_TotalBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_GranuleSize);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_CurrentBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_PlatformVersion);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_GetDeviceDescriptions);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Serialize);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Attributes);
  _PJRT_API_STRUCT_FIELD(PJRT_Compile);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputElementTypes);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputDimensions);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToMemory);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_CreateViewOfDeviceBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Fingerprint);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_TopologyDescription);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_GetCompiledMemoryStats);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Kind_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_ExecuteContext_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_ExecuteContext_Destroy);
} PJRT_Api;
enum {
  PJRT_Api_STRUCT_SIZE =
      PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_TopologyDescription)
};
#undef _PJRT_API_STRUCT_FIELD
#ifdef __cplusplus
}
#endif
#endif  