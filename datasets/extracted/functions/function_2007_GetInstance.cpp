#include "arolla/jagged_shape/dense_array/qtype/qtype.h"
#include "absl/base/no_destructor.h"
#include "arolla/dense_array/edge.h"
#include "arolla/dense_array/qtype/types.h"
#include "arolla/jagged_shape/dense_array/jagged_shape.h"
#include "arolla/jagged_shape/qtype/qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/util/init_arolla.h"
#include "arolla/util/meta.h"
namespace arolla {
namespace {
class JaggedDenseArrayShapeQType final : public JaggedShapeQType {
 public:
  static const JaggedDenseArrayShapeQType* GetInstance() {
    static absl::NoDestructor<JaggedDenseArrayShapeQType> result;
    return result.get();
  }
  JaggedDenseArrayShapeQType()
      : JaggedShapeQType(meta::type<JaggedDenseArrayShape>(),
                         "JAGGED_DENSE_ARRAY_SHAPE") {}
  QTypePtr edge_qtype() const override { return GetQType<DenseArrayEdge>(); };
};
}  
QTypePtr QTypeTraits<JaggedDenseArrayShape>::type() {
  return JaggedDenseArrayShapeQType::GetInstance();
}
AROLLA_INITIALIZER(
        .reverse_deps = {arolla::initializer_dep::kQTypes}, .init_fn = [] {
          return SetEdgeQTypeToJaggedShapeQType(
              GetQType<DenseArrayEdge>(), GetQType<JaggedDenseArrayShape>());
        })
}  