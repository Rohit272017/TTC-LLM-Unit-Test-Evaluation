#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_COMPOSABLE_SPLITTER_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_COMPOSABLE_SPLITTER_H_
#include <vector>
#include "tensorflow/tools/proto_splitter/cc/composable_splitter_base.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/protobuf.h"
namespace tensorflow {
namespace tools::proto_splitter {
class ComposableSplitter : public ComposableSplitterBase {
 public:
  explicit ComposableSplitter(tsl::protobuf::Message* message)
      : ComposableSplitterBase(message), message_(message) {}
  explicit ComposableSplitter(tsl::protobuf::Message* message,
                              ComposableSplitterBase* parent_splitter,
                              std::vector<FieldType>* fields_in_parent)
      : ComposableSplitterBase(message, parent_splitter, fields_in_parent),
        message_(message) {}
 protected:
  tsl::protobuf::Message* message() { return message_; }
 private:
  tsl::protobuf::Message* message_;
};
}  
}  
#endif  