#include "tensorflow/c/experimental/gradients/array_grad.h"
#include "tensorflow/c/eager/abstract_context.h"
namespace tensorflow {
namespace gradients {
namespace {
class IdentityNGradientFunction : public GradientFunction {
 public:
  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
    for (int i = 0; i < grad_outputs.size(); i++) {
      auto grad_input = grad_outputs[i];
      if (grad_input) {
        grad_input->Ref();
      }
      grad_inputs[i] = grad_input;
    }
    return absl::OkStatus();
  }
  ~IdentityNGradientFunction() override {}
};
}  
GradientFunction* IdentityNRegisterer(const ForwardOperation& op) {
  return new IdentityNGradientFunction;
}
}  
}  