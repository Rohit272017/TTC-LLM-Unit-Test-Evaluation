#include "tensorflow/core/kernels/tensor_cord.h"
#include <cstring>
#include "tensorflow/core/framework/variant.h"
namespace tensorflow {
static_assert(Variant::CanInlineType<TensorCord>(),
              "TensorCord should be inlined into Variants");
TensorCord::CordRep::~CordRep() {
  if (!is_inline_ && rep_.external.releaser) {
    rep_.external.releaser(rep_.external.arg);
  }
}
TensorCord::~TensorCord() { Cleanup(); }
void TensorCord::Encode(VariantTensorData* data) const {
  data->metadata_string().clear();
  for (auto rep : Chunks()) {
    data->metadata_string().append(rep.data(), rep.size());
  }
}
bool TensorCord::Decode(VariantTensorData data) {
  auto* str = new string(std::move(data.metadata_string()));
  Cleanup();
  chunks_.push_back(new CordRep(absl::string_view(*str), &StringReleaser, str));
  return true;
}
TensorBuffer* TensorCord::TensorBufWithRef(Tensor* tensor) {
  TensorBuffer* buf = tensor->buf_;
  buf->Ref();
  return buf;
}
void TensorCord::TensorBufReleaser(void* tensor_buffer) {
  static_cast<TensorBuffer*>(tensor_buffer)->Unref();
}
void TensorCord::StringReleaser(void* str_ptr) {
  delete static_cast<string*>(str_ptr);
}
namespace {
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) { s->resize(new_size); }
};
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};
static inline void STLStringResizeUninitialized(string* s, size_t new_size) {
  ResizeUninitializedTraits<string>::Resize(s, new_size);
}
}  
TensorCord::operator string() const {
  string out;
  STLStringResizeUninitialized(&out, size());
  char* data = const_cast<char*>(out.data());
  for (auto* rep : chunks_) {
    auto view = rep->view();
    memcpy(data, view.data(), view.size());
    data += view.size();
  }
  DCHECK_EQ(data - out.data(), size());
  return out;
}
}  