#include "tensorstore/util/utf8_string.h"
#include "tensorstore/internal/riegeli/delimited.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace serialization {
bool Serializer<Utf8String>::Encode(EncodeSink& sink, const Utf8String& value) {
  return serialization::WriteDelimited(sink.writer(), value.utf8);
}
bool Serializer<Utf8String>::Decode(DecodeSource& source, Utf8String& value) {
  return serialization::ReadDelimitedUtf8(source.reader(), value.utf8);
}
}  
}  