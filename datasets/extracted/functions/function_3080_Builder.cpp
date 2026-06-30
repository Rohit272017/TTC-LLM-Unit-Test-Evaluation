#include "arolla/memory/strings_buffer.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/memory/optional_value.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/memory/simple_buffer.h"
#include "arolla/util/fingerprint.h"
namespace arolla {
StringsBuffer::Builder::Builder(int64_t max_size, RawBufferFactory* factory)
    : factory_(factory) {
  size_t initial_char_buffer_size = max_size * 16;
  DCHECK_LT(initial_char_buffer_size, std::numeric_limits<offset_type>::max());
  size_t offsets_size = max_size * sizeof(Offsets);
  InitDataPointers(
      factory->CreateRawBuffer(offsets_size + initial_char_buffer_size),
      max_size, initial_char_buffer_size);
  std::memset(offsets_.data(), 0, offsets_size);
}
StringsBuffer::ReshuffleBuilder::ReshuffleBuilder(
    int64_t max_size, const StringsBuffer& buffer,
    const OptionalValue<absl::string_view>& default_value,
    RawBufferFactory* buf_factory)
    : offsets_bldr_(max_size, buf_factory),
      old_offsets_(buffer.offsets()),
      characters_(buffer.characters()),
      base_offset_(buffer.base_offset()) {
  if (default_value.present && !default_value.value.empty()) {
    int64_t def_value_size = default_value.value.size();
    offsets_bldr_.SetNConst(
        0, max_size, {characters_.size(), def_value_size + characters_.size()});
    SimpleBuffer<char>::Builder chars_bldr(characters_.size() + def_value_size,
                                           buf_factory);
    char* data = chars_bldr.GetMutableSpan().data();
    std::memcpy(data, characters_.begin(), characters_.size());
    std::memcpy(data + characters_.size(), default_value.value.begin(),
                def_value_size);
    characters_ = std::move(chars_bldr).Build();
  } else {
    std::memset(offsets_bldr_.GetMutableSpan().begin(), 0,
                max_size * sizeof(Offsets));
  }
}
StringsBuffer StringsBuffer::Builder::Build(int64_t size) && {
  DCHECK_LE(size, offsets_.size());
  if (num_chars_ != characters_.size()) {
    ResizeCharacters(num_chars_);
  }
  SimpleBuffer<Offsets> offsets(buf_, offsets_.subspan(0, size));
  SimpleBuffer<char> characters(std::move(buf_),
                                characters_.subspan(0, num_chars_));
  return StringsBuffer(std::move(offsets), std::move(characters));
}
void StringsBuffer::Builder::ResizeCharacters(size_t new_size) {
  DCHECK_LT(new_size, std::numeric_limits<offset_type>::max());
  size_t offsets_size = offsets_.size() * sizeof(Offsets);
  InitDataPointers(factory_->ReallocRawBuffer(std::move(buf_), offsets_.begin(),
                                              offsets_size + characters_.size(),
                                              offsets_size + new_size),
                   offsets_.size(), new_size);
}
void StringsBuffer::Builder::InitDataPointers(
    std::tuple<RawBufferPtr, void*>&& buf, int64_t offsets_count,
    int64_t characters_size) {
  buf_ = std::move(std::get<0>(buf));
  void* data = std::get<1>(buf);
  offsets_ =
      absl::Span<Offsets>(reinterpret_cast<Offsets*>(data), offsets_count);
  characters_ = absl::Span<char>(
      reinterpret_cast<char*>(data) + offsets_count * sizeof(Offsets),
      characters_size);
}
StringsBuffer::StringsBuffer(SimpleBuffer<StringsBuffer::Offsets> offsets,
                             SimpleBuffer<char> characters,
                             offset_type base_offset)
    : offsets_(std::move(offsets)),
      characters_(std::move(characters)),
      base_offset_(base_offset) {
  for (int64_t i = 0; i < offsets_.size(); ++i) {
    DCHECK_LE(base_offset_, offsets_[i].start);
    DCHECK_LE(offsets_[i].start, offsets_[i].end);
    DCHECK_LE(offsets_[i].end, base_offset_ + characters_.size());
  }
}
bool StringsBuffer::operator==(const StringsBuffer& other) const {
  if (this == &other) {
    return true;
  }
  if (size() != other.size()) {
    return false;
  }
  return std::equal(begin(), end(), other.begin());
}
StringsBuffer StringsBuffer::Slice(int64_t offset, int64_t count) const& {
  if (count == 0) {
    return StringsBuffer{};
  }
  return StringsBuffer{offsets_.Slice(offset, count), characters_,
                       base_offset_};
}
StringsBuffer StringsBuffer::Slice(int64_t offset, int64_t count) && {
  if (count == 0) {
    return StringsBuffer{};
  }
  return StringsBuffer{std::move(offsets_).Slice(offset, count),
                       std::move(characters_), base_offset_};
}
StringsBuffer StringsBuffer::ShallowCopy() const {
  return StringsBuffer(offsets_.ShallowCopy(), characters_.ShallowCopy(),
                       base_offset_);
}
StringsBuffer StringsBuffer::DeepCopy(RawBufferFactory* buffer_factory) const {
  if (size() == 0) {
    return StringsBuffer{};
  }
  offset_type min_offset = offsets_[0].start;
  offset_type max_offset = offsets_[0].end;
  for (int64_t i = 1; i < size(); ++i) {
    min_offset = std::min(min_offset, offsets_[i].start);
    max_offset = std::max(max_offset, offsets_[i].end);
  }
  auto characters_slice =
      characters_.Slice(min_offset - base_offset_, max_offset - min_offset);
  return StringsBuffer(offsets_.DeepCopy(buffer_factory),
                       characters_slice.DeepCopy(buffer_factory), min_offset);
}
void FingerprintHasherTraits<StringsBuffer>::operator()(
    FingerprintHasher* hasher, const StringsBuffer& value) const {
  hasher->Combine(value.size());
  if (!value.empty()) {
    auto offsets_span = value.offsets().span();
    hasher->CombineRawBytes(offsets_span.data(),
                            offsets_span.size() * sizeof(offsets_span[0]));
    hasher->CombineSpan(value.characters().span());
  }
}
}  