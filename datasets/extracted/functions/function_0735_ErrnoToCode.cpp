#include "tsl/platform/errors.h"
#include <errno.h>
#include <string.h>
#include "tsl/platform/status.h"
#include "tsl/platform/strcat.h"
namespace tsl {
namespace errors {
namespace {
absl::StatusCode ErrnoToCode(int err_number) {
  absl::StatusCode code;
  switch (err_number) {
    case 0:
      code = absl::StatusCode::kOk;
      break;
    case EINVAL:        
    case ENAMETOOLONG:  
    case E2BIG:         
    case EDESTADDRREQ:  
    case EDOM:          
    case EFAULT:        
    case EILSEQ:        
    case ENOPROTOOPT:   
    case ENOSTR:        
    case ENOTSOCK:      
    case ENOTTY:        
    case EPROTOTYPE:    
    case ESPIPE:        
      code = absl::StatusCode::kInvalidArgument;
      break;
    case ETIMEDOUT:  
    case ETIME:      
      code = absl::StatusCode::kDeadlineExceeded;
      break;
    case ENODEV:  
    case ENOENT:  
    case ENXIO:   
    case ESRCH:   
      code = absl::StatusCode::kNotFound;
      break;
    case EEXIST:         
    case EADDRNOTAVAIL:  
    case EALREADY:       
      code = absl::StatusCode::kAlreadyExists;
      break;
    case EPERM:   
    case EACCES:  
    case EROFS:   
      code = absl::StatusCode::kPermissionDenied;
      break;
    case ENOTEMPTY:   
    case EISDIR:      
    case ENOTDIR:     
    case EADDRINUSE:  
    case EBADF:       
    case EBUSY:       
    case ECHILD:      
    case EISCONN:     
#if !defined(_WIN32) && !defined(__HAIKU__)
    case ENOTBLK:  
#endif
    case ENOTCONN:  
    case EPIPE:     
#if !defined(_WIN32)
    case ESHUTDOWN:  
#endif
    case ETXTBSY:  
      code = absl::StatusCode::kFailedPrecondition;
      break;
    case ENOSPC:  
#if !defined(_WIN32)
    case EDQUOT:  
#endif
    case EMFILE:   
    case EMLINK:   
    case ENFILE:   
    case ENOBUFS:  
    case ENODATA:  
    case ENOMEM:   
    case ENOSR:    
#if !defined(_WIN32) && !defined(__HAIKU__)
    case EUSERS:  
#endif
      code = absl::StatusCode::kResourceExhausted;
      break;
    case EFBIG:      
    case EOVERFLOW:  
    case ERANGE:     
      code = absl::StatusCode::kOutOfRange;
      break;
    case ENOSYS:        
    case ENOTSUP:       
    case EAFNOSUPPORT:  
#if !defined(_WIN32)
    case EPFNOSUPPORT:  
#endif
    case EPROTONOSUPPORT:  
#if !defined(_WIN32) && !defined(__HAIKU__)
    case ESOCKTNOSUPPORT:  
#endif
    case EXDEV:  
      code = absl::StatusCode::kUnimplemented;
      break;
    case EAGAIN:        
    case ECONNREFUSED:  
    case ECONNABORTED:  
    case ECONNRESET:    
    case EINTR:         
#if !defined(_WIN32)
    case EHOSTDOWN:  
#endif
    case EHOSTUNREACH:  
    case ENETDOWN:      
    case ENETRESET:     
    case ENETUNREACH:   
    case ENOLCK:        
    case ENOLINK:       
#if !(defined(__APPLE__) || defined(__FreeBSD__) || defined(_WIN32) || \
      defined(__HAIKU__))
    case ENONET:  
#endif
      code = absl::StatusCode::kUnavailable;
      break;
    case EDEADLK:  
#if !defined(_WIN32)
    case ESTALE:  
#endif
      code = absl::StatusCode::kAborted;
      break;
    case ECANCELED:  
      code = absl::StatusCode::kCancelled;
      break;
    case EBADMSG:      
    case EIDRM:        
    case EINPROGRESS:  
    case EIO:          
    case ELOOP:        
    case ENOEXEC:      
    case ENOMSG:       
    case EPROTO:       
#if !defined(_WIN32) && !defined(__HAIKU__)
    case EREMOTE:  
#endif
      code = absl::StatusCode::kUnknown;
      break;
    default: {
      code = absl::StatusCode::kUnknown;
      break;
    }
  }
  return code;
}
}  
absl::Status IOError(const string& context, int err_number) {
  auto code = ErrnoToCode(err_number);
  return absl::Status(code,
                      strings::StrCat(context, "; ", strerror(err_number)));
}
bool IsAborted(const absl::Status& status) {
  return status.code() == tsl::error::Code::ABORTED;
}
bool IsAlreadyExists(const absl::Status& status) {
  return status.code() == tsl::error::Code::ALREADY_EXISTS;
}
bool IsCancelled(const absl::Status& status) {
  return status.code() == tsl::error::Code::CANCELLED;
}
bool IsDataLoss(const absl::Status& status) {
  return status.code() == tsl::error::Code::DATA_LOSS;
}
bool IsDeadlineExceeded(const absl::Status& status) {
  return status.code() == tsl::error::Code::DEADLINE_EXCEEDED;
}
bool IsFailedPrecondition(const absl::Status& status) {
  return status.code() == tsl::error::Code::FAILED_PRECONDITION;
}
bool IsInternal(const absl::Status& status) {
  return status.code() == tsl::error::Code::INTERNAL;
}
bool IsInvalidArgument(const absl::Status& status) {
  return status.code() == tsl::error::Code::INVALID_ARGUMENT;
}
bool IsNotFound(const absl::Status& status) {
  return status.code() == tsl::error::Code::NOT_FOUND;
}
bool IsOutOfRange(const absl::Status& status) {
  return status.code() == tsl::error::Code::OUT_OF_RANGE;
}
bool IsPermissionDenied(const absl::Status& status) {
  return status.code() == tsl::error::Code::PERMISSION_DENIED;
}
bool IsResourceExhausted(const absl::Status& status) {
  return status.code() == tsl::error::Code::RESOURCE_EXHAUSTED;
}
bool IsUnauthenticated(const absl::Status& status) {
  return status.code() == tsl::error::Code::UNAUTHENTICATED;
}
bool IsUnavailable(const absl::Status& status) {
  return status.code() == tsl::error::Code::UNAVAILABLE;
}
bool IsUnimplemented(const absl::Status& status) {
  return status.code() == tsl::error::Code::UNIMPLEMENTED;
}
bool IsUnknown(const absl::Status& status) {
  return status.code() == tsl::error::Code::UNKNOWN;
}
}  
}  