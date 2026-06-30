#include "tensorflow/lite/acceleration/configuration/stable_delegate_plugin.h"
namespace tflite {
namespace delegates {
TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(StableDelegatePlugin,
                                          StableDelegatePlugin::New);
}  
}  