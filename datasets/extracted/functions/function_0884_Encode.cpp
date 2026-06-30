#include "langsvr/lsp/encode.h"
namespace langsvr::lsp {
Result<const json::Value*> Encode(Null, json::Builder& b) {
    return b.Null();
}
Result<const json::Value*> Encode(Boolean in, json::Builder& b) {
    return b.Bool(in);
}
Result<const json::Value*> Encode(Integer in, json::Builder& b) {
    return b.I64(in);
}
Result<const json::Value*> Encode(Uinteger in, json::Builder& b) {
    return b.U64(in);
}
Result<const json::Value*> Encode(Decimal in, json::Builder& b) {
    return b.F64(in);
}
Result<const json::Value*> Encode(const String& in, json::Builder& b) {
    return b.String(in);
}
}  