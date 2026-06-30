#include "arolla/util/demangle.h"
#include <cstdlib>
#include <string>
#include <typeinfo>
#include "arolla/util/bytes.h"
#if defined(__GXX_RTTI)
#define AROLLA_HAS_CXA_DEMANGLE
#endif
#ifdef AROLLA_HAS_CXA_DEMANGLE
#include <cxxabi.h>  
#endif
namespace arolla {
std::string TypeName(const std::type_info& ti) {
  if (ti == typeid(arolla::Bytes)) {
    return "arolla::Bytes";
  }
  int status = 0;
  char* demangled = nullptr;
#ifdef AROLLA_HAS_CXA_DEMANGLE
  demangled = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
#endif
  if (status == 0 && demangled != nullptr) {  
    std::string out = demangled;
    free(demangled);
    return out;
  } else {
    return ti.name();
  }
}
}  