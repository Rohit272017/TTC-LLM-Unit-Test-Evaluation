#include "langsvr/content_stream.h"
#include <sstream>
#include <string>
#include "langsvr/reader.h"
#include "langsvr/writer.h"
#include "src/utils/replace_all.h"
namespace langsvr {
namespace {
static constexpr std::string_view kContentLength = "Content-Length: ";
Result<SuccessType> Match(Reader& reader, std::string_view str) {
    auto got = reader.String(str.size());
    if (got != Success) {
        return got.Failure();
    }
    if (got.Get() != str) {
        std::stringstream err;
        err << "expected '" << str << "' got '" << got.Get() << "'";
        return Failure{err.str()};
    }
    return Success;
}
}  
Result<std::string> ReadContent(Reader& reader) {
    if (auto match = Match(reader, kContentLength); match != Success) {
        return match.Failure();
    }
    uint64_t len = 0;
    while (true) {
        char c = 0;
        if (auto read = reader.Read(reinterpret_cast<std::byte*>(&c), sizeof(c));
            read != sizeof(c)) {
            return Failure{"end of stream while parsing content length"};
        }
        if (c >= '0' && c <= '9') {
            len = len * 10 + static_cast<uint64_t>(c - '0');
            continue;
        }
        if (c == '\r') {
            break;
        }
        return Failure{"invalid content length value"};
    }
    auto got = reader.String(3);
    if (got != Success) {
        return got.Failure();
    }
    if (got.Get() != "\n\r\n") {
        auto fmt = [](std::string s) {
            s = ReplaceAll(s, "\n", "␊");
            s = ReplaceAll(s, "\r", "␍");
            return s;
        };
        std::stringstream err;
        err << "expected '␍␊␍␊' got '␍" << fmt(got.Get()) << "'";
        return Failure{err.str()};
    }
    return reader.String(len);
}
Result<SuccessType> WriteContent(Writer& writer, std::string_view content) {
    std::stringstream ss;
    ss << kContentLength << content.length() << "\r\n\r\n" << content;
    return writer.String(ss.str());
}
}  