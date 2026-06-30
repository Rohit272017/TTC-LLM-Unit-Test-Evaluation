#include "tensorflow/core/common_runtime/device_mgr.h"
#include <memory>
#include <vector>
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
DeviceMgr::~DeviceMgr() {}
}  