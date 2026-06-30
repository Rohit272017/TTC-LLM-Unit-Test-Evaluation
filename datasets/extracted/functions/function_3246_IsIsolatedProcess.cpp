#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  
#define EXPAND_VA_ARGS(...) , ##__VA_ARGS__
#define NNAPI_LOG(format, ...) \
  fprintf(stderr, format "\n" EXPAND_VA_ARGS(__VA_ARGS__));
namespace {
#ifdef __ANDROID__
const int kFirstIsolatedUid = 99000;
const int kLastIsolatedUid = 99999;
const int kFirstAppZygoteIsolatedUid = 90000;
const int kLastAppZygoteIsolatedUid = 98999;
bool IsIsolatedProcess() {
  int uid = getuid();
  return (uid >= kFirstIsolatedUid && uid <= kLastIsolatedUid) ||
         (uid >= kFirstAppZygoteIsolatedUid &&
          uid <= kLastAppZygoteIsolatedUid);
}
int32_t GetAndroidSdkVersion() {
  const char* sdkProp = "ro.build.version.sdk";
  char sdkVersion[PROP_VALUE_MAX];
  int length = __system_property_get(sdkProp, sdkVersion);
  if (length != 0) {
    int32_t result = 0;
    for (int i = 0; i < length; ++i) {
      int digit = sdkVersion[i] - '0';
      if (digit < 0 || digit > 9) {
        return 0xffff;
      }
      result = result * 10 + digit;
    }
    return result;
  }
  return 0;
}
#endif  
void* LoadFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}
#ifndef __ANDROID__
int ASharedMemory_create(const char* name, size_t size) {
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    return fd;
  }
  int result = ftruncate(fd, size);
  if (result < 0) {
    close(fd);
    return -1;
  }
  return fd;
}
uint32_t CalculateAndroidSdkVersion(NnApi const& nnapi) {
  bool has_10 = nnapi.ANeuralNetworksMemory_createFromFd != nullptr;
  bool has_11 =
      nnapi.ANeuralNetworksModel_relaxComputationFloat32toFloat16 != nullptr;
  bool has_12 = nnapi.ANeuralNetworks_getDeviceCount != nullptr;
  bool has_13 = nnapi.ANeuralNetworksCompilation_setTimeout != nullptr;
  bool has_14 = nnapi.ANeuralNetworks_getRuntimeFeatureLevel != nullptr;
  uint32_t sdk_version = 0;
  if (has_10) {
    sdk_version = 27;
  }
  if (sdk_version == 27 && has_11) {
    sdk_version = 28;
  }
  if (sdk_version == 28 && has_12) {
    sdk_version = 29;
  }
  if (sdk_version == 29 && has_13) {
    sdk_version = 30;
  }
  if (sdk_version == 30 && has_14) {
    sdk_version = 31;
  }
  return sdk_version;
}
#else
ASharedMemory_create_fn getASharedMemory_create() {
  void* libandroid = nullptr;
  libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
  if (libandroid != nullptr) {
    return reinterpret_cast<ASharedMemory_create_fn>(
        LoadFunction(libandroid, "ASharedMemory_create", false));
  }
  std::string libandroid_error = dlerror();
  void* cutils_handle = dlopen("libcutils.so", RTLD_LAZY | RTLD_LOCAL);
  if (cutils_handle != nullptr) {
    return reinterpret_cast<ASharedMemory_create_fn>(
        LoadFunction(cutils_handle, "ashmem_create_region", false));
  }
  NNAPI_LOG(
      "nnapi error: unable to open both library %s (%s) and library %s "
      "(%s)",
      "libandroid.so", libandroid_error.c_str(), "libcutils.so", dlerror());
  return nullptr;
}
#endif  
#define LOAD_FUNCTION(handle, name)         \
  nnapi.name = reinterpret_cast<name##_fn>( \
      LoadFunction(handle, #name,  false));
#define LOAD_FUNCTION_OPTIONAL(handle, name) \
  nnapi.name = reinterpret_cast<name##_fn>(  \
      LoadFunction(handle, #name,  true));
#define LOAD_FUNCTION_RENAME(handle, name, symbol) \
  nnapi.name = reinterpret_cast<name##_fn>(        \
      LoadFunction(handle, symbol,  false));
const NnApi LoadNnApi() {
  NnApi nnapi = {};
  nnapi.android_sdk_version = 0;
#ifdef __ANDROID__
  nnapi.android_sdk_version = GetAndroidSdkVersion();
  if (nnapi.android_sdk_version < 27) {
    NNAPI_LOG("nnapi error: requires android sdk version to be at least %d",
              27);
    nnapi.nnapi_exists = false;
    return nnapi;
  }
  if (nnapi.android_sdk_version <= 33 && IsIsolatedProcess()) {
    NNAPI_LOG("NNAPI is disabled in an isolated process");
    nnapi.nnapi_exists = false;
    return nnapi;
  }
#endif  
  void* libneuralnetworks = nullptr;
  static const char nnapi_library_name[] = "libneuralnetworks.so";
  libneuralnetworks = dlopen(nnapi_library_name, RTLD_LAZY | RTLD_LOCAL);
#ifdef __ANDROID__
  if (libneuralnetworks == nullptr) {
    const char* error = dlerror();
    if (error) {
      NNAPI_LOG("%s\n", error);
    }
    NNAPI_LOG("nnapi error: unable to open library %s", nnapi_library_name);
  }
#endif  
  nnapi.nnapi_exists = libneuralnetworks != nullptr;
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_createFromFd);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperand);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_setOperandValue);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_setOperandValueFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperation);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_identifyInputsAndOutputs);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_setPreference);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInput);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setOutput);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksExecution_setOutputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_startCompute);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_wait);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_free);
#ifdef __ANDROID__
  nnapi.ASharedMemory_create = getASharedMemory_create();
#else
  if (libneuralnetworks != nullptr) {
    nnapi.ASharedMemory_create = ASharedMemory_create;
  }
#endif  
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDeviceCount);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDevice);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getName);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getVersion);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksDevice_getFeatureLevel);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getSupportedOperationsForDevices);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_createForDevices);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setCaching);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_compute);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getOutputOperandRank);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getOutputOperandDimensions);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_create);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_free);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_burstCompute);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemory_createFromAHardwareBuffer);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setMeasureTiming);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getDuration);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksDevice_getExtensionSupport);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getExtensionOperandType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getExtensionOperationType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_setOperandExtensionData);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setPriority);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setLoopTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_create);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_free);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_addInputRole);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_addOutputRole);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_setDimensions);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_finish);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemory_createFromDesc);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemory_copy);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksEvent_createFromSyncFenceFd);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksEvent_getSyncFenceFd);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_startComputeWithDependencies);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworks_getRuntimeFeatureLevel);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_enableInputAndOutputPadding);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setReusable);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         SL_ANeuralNetworksDiagnostic_registerCallbacks);
#ifndef __ANDROID__
  if (nnapi.nnapi_exists && nnapi.android_sdk_version == 0) {
    nnapi.android_sdk_version = CalculateAndroidSdkVersion(nnapi);
  }
#endif  
  if (nnapi.ANeuralNetworks_getRuntimeFeatureLevel) {
    nnapi.nnapi_runtime_feature_level =
        nnapi.ANeuralNetworks_getRuntimeFeatureLevel();
  } else {
    nnapi.nnapi_runtime_feature_level = nnapi.android_sdk_version;
  }
  return nnapi;
}
}  
std::unique_ptr<const NnApi> CreateNnApiFromSupportLibrary(
    const NnApiSLDriverImplFL5* nnapi_support_library_driver) {
  auto nnapi = std::make_unique<NnApi>();
  nnapi->nnapi_exists = true;
  nnapi->android_sdk_version = ANEURALNETWORKS_FEATURE_LEVEL_5;
  nnapi->nnapi_runtime_feature_level =
      nnapi_support_library_driver->base.implFeatureLevel;
#define ASSIGN_SL_FUNCTION_TO_NNAPI(name) \
  nnapi->name = nnapi_support_library_driver->name;
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemory_createFromFd);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemory_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_create);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_finish);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_addOperand);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_setOperandValue);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_setOperandValueFromMemory);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_addOperation);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_identifyInputsAndOutputs);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  nnapi->ANeuralNetworksCompilation_create = nullptr;
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_setPreference);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_finish);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_create);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setInput);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setInputFromMemory);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setOutput);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setOutputFromMemory);
  nnapi->ANeuralNetworksExecution_startCompute = nullptr;
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksEvent_wait);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksEvent_free);
#ifdef __ANDROID__
  nnapi->ASharedMemory_create = getASharedMemory_create();
#else
  nnapi->ASharedMemory_create = ASharedMemory_create;
#endif  
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworks_getDeviceCount);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworks_getDevice);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksDevice_getName);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksDevice_getVersion);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksDevice_getFeatureLevel);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksDevice_getType);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksModel_getSupportedOperationsForDevices);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_createForDevices);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_setCaching);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_setTimeout);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksCompilation_setPriority);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_compute);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setTimeout);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setLoopTimeout);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_getOutputOperandRank);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksExecution_getOutputOperandDimensions);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksBurst_create);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksBurst_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_burstCompute);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemory_createFromAHardwareBuffer);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setMeasureTiming);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_getDuration);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksDevice_getExtensionSupport);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_getExtensionOperandType);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_getExtensionOperationType);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksModel_setOperandExtensionData);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_create);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_free);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_addInputRole);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_addOutputRole);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_setDimensions);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemoryDesc_finish);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemory_createFromDesc);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksMemory_copy);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksEvent_createFromSyncFenceFd);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksEvent_getSyncFenceFd);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksExecution_startComputeWithDependencies);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      ANeuralNetworksExecution_enableInputAndOutputPadding);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworksExecution_setReusable);
  ASSIGN_SL_FUNCTION_TO_NNAPI(ANeuralNetworks_getRuntimeFeatureLevel);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed);
  ASSIGN_SL_FUNCTION_TO_NNAPI(
      SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed);
  ASSIGN_SL_FUNCTION_TO_NNAPI(SL_ANeuralNetworksDiagnostic_registerCallbacks);
  return nnapi;
}
const NnApi* NnApiImplementation() {
  static const NnApi nnapi = LoadNnApi();
  return &nnapi;
}