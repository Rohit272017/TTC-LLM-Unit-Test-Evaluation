#include "tensorflow/lite/delegates/gpu/async_buffers.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/android_hardware_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
namespace {
PFNGLBUFFERSTORAGEEXTERNALEXTPROC glBufferStorageExternalEXT;
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC eglGetNativeClientBufferANDROID;
bool IsGlSupported() {
  static const bool extensions_allowed = [] {
    eglGetNativeClientBufferANDROID =
        reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    glBufferStorageExternalEXT =
        reinterpret_cast<PFNGLBUFFERSTORAGEEXTERNALEXTPROC>(
            eglGetProcAddress("glBufferStorageExternalEXT"));
    return eglGetNativeClientBufferANDROID && glBufferStorageExternalEXT;
  }();
  return extensions_allowed;
}
}  
namespace tflite {
namespace gpu {
absl::Status AsyncBuffer::MapAHardwareBufferToGlBuffer() {
  if (!IsGlSupported()) {
    return absl::UnknownError(
        "No GL extension functions found to bind AHardwareBuffer and "
        "OpenGL buffer");
  }
  EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(ahwb_);
  if (!native_buffer) {
    return absl::UnknownError("Can't get native buffer");
  }
  glBufferStorageExternalEXT(GL_SHADER_STORAGE_BUFFER, 0, bytes_, native_buffer,
                             GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                                 GL_MAP_COHERENT_BIT_EXT |
                                 GL_MAP_PERSISTENT_BIT_EXT);
  return gl::GetOpenGlErrors();
}
absl::Status AsyncBuffer::AllocateOpenGlBuffer() {
  if (opengl_buffer_ == GL_INVALID_INDEX) {
    glGenBuffers(1, &opengl_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    absl::Status status = MapAHardwareBufferToGlBuffer();
    if (!status.ok()) {
      if (ahwb_ != nullptr) {
        if (OptionalAndroidHardwareBuffer::Instance().Supported()) {
          OptionalAndroidHardwareBuffer::Instance().Release(ahwb_);
        }
        ahwb_ = nullptr;
      }
      glBufferData(GL_SHADER_STORAGE_BUFFER, bytes_, nullptr, GL_STREAM_COPY);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  }
  return absl::OkStatus();
}
absl::Status AsyncBuffer::GetOpenGlBuffer(GLuint& buffer_ref) {
  if (!valid_) {
    absl::Status status = AllocateOpenGlBuffer();
    if (!status.ok()) {
      return status;
    }
  }
  valid_ = true;
  buffer_ref = opengl_buffer_;
  return absl::OkStatus();
}
}  
}  