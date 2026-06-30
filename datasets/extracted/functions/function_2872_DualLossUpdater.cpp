#ifndef TENSORFLOW_CORE_KERNELS_LOSS_H_
#define TENSORFLOW_CORE_KERNELS_LOSS_H_
#include "tensorflow/core/lib/core/status.h"
namespace tensorflow {
class DualLossUpdater {
 public:
  virtual ~DualLossUpdater() {}
  virtual double ComputeUpdatedDual(
      const int num_loss_partitions, const double label,
      const double example_weight, const double current_dual, const double wx,
      const double weighted_example_norm) const = 0;
  virtual double ComputeDualLoss(const double current_dual,
                                 const double example_label,
                                 const double example_weight) const = 0;
  virtual double ComputePrimalLoss(const double wx, const double example_label,
                                   const double example_weight) const = 0;
  virtual double PrimalLossDerivative(const double wx,
                                      const double example_label,
                                      const double example_weight) const = 0;
  virtual double SmoothnessConstant() const = 0;
  virtual Status ConvertLabel(float* const example_label) const = 0;
};
}  
#endif  