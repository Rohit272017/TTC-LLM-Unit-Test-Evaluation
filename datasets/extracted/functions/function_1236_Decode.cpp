#include "langsvr/lsp/decode.h"
namespace langsvr::lsp {
Result<SuccessType> Decode(const json::Value& v, Null&) {
    return v.Null();
}
Result<SuccessType> Decode(const json::Value& v, Boolean& out) {
    auto res = v.Bool();
    if (res == Success) [[likely]] {
        out = res.Get();
        return Success;
    }
    return res.Failure();
}
Result<SuccessType> Decode(const json::Value& v, Integer& out) {
    auto res = v.I64();
    if (res == Success) [[likely]] {
        out = res.Get();
        return Success;
    }
    return res.Failure();
}
Result<SuccessType> Decode(const json::Value& v, Uinteger& out) {
    auto res = v.U64();
    if (res == Success) [[likely]] {
        out = res.Get();
        return Success;
    }
    return res.Failure();
}
Result<SuccessType> Decode(const json::Value& v, Decimal& out) {
    auto res = v.F64();
    if (res == Success) [[likely]] {
        out = res.Get();
        return Success;
    }
    return res.Failure();
}
Result<SuccessType> Decode(const json::Value& v, String& out) {
    auto res = v.String();
    if (res == Success) [[likely]] {
        out = res.Get();
        return Success;
    }
    return res.Failure();
}
}  