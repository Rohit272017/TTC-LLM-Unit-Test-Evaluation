#include "tensorflow/compiler/mlir/tensorflow/utils/string_util.h"
#include <ostream>
#include <string>
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Operation.h"  
namespace tensorflow {
std::string OpAsString(mlir::Operation& op) {
  std::string out;
  llvm::raw_string_ostream op_stream(out);
  op.print(op_stream, mlir::OpPrintingFlags()
                          .elideLargeElementsAttrs()
                          .assumeVerified()
                          .skipRegions()
                          .printGenericOpForm());
  return out;
}
std::string AttrAsString(mlir::Attribute& attr) {
  std::string out;
  llvm::raw_string_ostream attr_stream(out);
  attr.print(attr_stream);
  return out;
}
std::ostream& operator<<(std::ostream& o, const LoggableOperation& op) {
  return o << OpAsString(op.v);
}
std::ostream& operator<<(std::ostream& o, const LoggableAttribute& attr) {
  return o << AttrAsString(attr.v);
}
std::ostream& operator<<(std::ostream& o, const LoggableStringRef& ref) {
  return o << ref.v.str();
}
}  