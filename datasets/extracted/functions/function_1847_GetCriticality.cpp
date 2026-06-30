#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_CRITICALITY_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_CRITICALITY_H_
namespace tsl {
namespace criticality {
inline Criticality GetCriticality() {
  return Criticality::kCritical;
}
}  
}  
#endif  