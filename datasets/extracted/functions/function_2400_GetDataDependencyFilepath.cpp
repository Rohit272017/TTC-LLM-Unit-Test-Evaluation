#include "tsl/platform/resource_loader.h"
#include <cstdlib>
#include <string>
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/test.h"
namespace tsl {
std::string GetDataDependencyFilepath(const std::string& relative_path) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  if (!srcdir) {
    LOG(FATAL) << "Environment variable TEST_SRCDIR unset!";  
  }
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (!workspace) {
    LOG(FATAL) << "Environment variable TEST_WORKSPACE unset!";  
  }
  return kIsOpenSource
             ? io::JoinPath(srcdir, workspace, relative_path)
             : io::JoinPath(srcdir, workspace, "third_party", relative_path);
}
}  