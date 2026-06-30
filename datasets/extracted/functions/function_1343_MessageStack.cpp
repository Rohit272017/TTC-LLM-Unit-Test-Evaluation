#include "tensorflow/lite/testing/message.h"
#include <stack>
#include <string>
#include "tensorflow/lite/testing/tokenize.h"
namespace tflite {
namespace testing {
class MessageStack : public TokenProcessor {
 public:
  explicit MessageStack(Message* first_node) {
    nodes_.push(first_node);
    valid_ = true;
  }
  void ConsumeToken(std::string* token) override {
    if (!valid_) return;
    Message* current_node = nodes_.top();
    if (*token == "{") {
      if (previous_token_.empty()) {
        valid_ = false;
        return;
      }
      nodes_.push(current_node ? current_node->AddChild(previous_token_)
                               : nullptr);
      previous_token_.clear();
    } else if (*token == "}") {
      if (nodes_.size() == 1 || !previous_token_.empty()) {
        valid_ = false;
        return;
      }
      if (current_node) {
        current_node->Finish();
      }
      nodes_.pop();
    } else if (*token == ":") {
      if (previous_token_.empty()) {
        valid_ = false;
        return;
      }
    } else {
      if (previous_token_.empty()) {
        previous_token_.swap(*token);
      } else {
        if (current_node) {
          current_node->SetField(previous_token_, *token);
        }
        previous_token_.clear();
      }
    }
  }
  bool valid() const { return valid_; }
 private:
  std::stack<Message*> nodes_;
  std::string previous_token_;
  bool valid_;
};
bool Message::Read(std::istream* input, Message* message) {
  MessageStack stack(message);
  Tokenize(input, &stack);
  return stack.valid();
}
}  
}  