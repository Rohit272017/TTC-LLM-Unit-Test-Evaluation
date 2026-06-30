#include <cstddef>
#include <cstdlib>
#include <string>
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/string_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"
namespace tensorflow {
template <typename T>
class SubstrOp : public OpKernel {
 public:
  explicit SubstrOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string unit;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unit", &unit));
    OP_REQUIRES_OK(ctx, ParseCharUnit(unit, &unit_));
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& pos_tensor = context->input(1);
    const Tensor& len_tensor = context->input(2);
    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& pos_shape = pos_tensor.shape();
    const TensorShape& len_shape = len_tensor.shape();
    OP_REQUIRES(context, (pos_shape == len_shape),
                errors::InvalidArgument(
                    "pos and len should have the same shape, got: ",
                    pos_shape.DebugString(), " vs. ", len_shape.DebugString()));
    bool is_scalar = TensorShapeUtils::IsScalar(pos_shape);
    if (is_scalar || input_shape == pos_shape) {
      auto input = input_tensor.flat<tstring>();
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("output", input_tensor.shape(),
                                              &output_tensor));
      auto output = output_tensor->flat<tstring>();
      if (is_scalar) {
        const T pos =
            tensorflow::internal::SubtleMustCopy(pos_tensor.scalar<T>()());
        const T len =
            tensorflow::internal::SubtleMustCopy(len_tensor.scalar<T>()());
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          StringPiece in(input(i));
          T byte_pos = pos;
          T byte_len = len;
          switch (unit_) {
            case CharUnit::UTF8_CHAR:
              OP_REQUIRES(
                  context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string at index ", i));
              break;
            case CharUnit::BYTE:
              byte_pos = AdjustedPosIndex(byte_pos, in);
              OP_REQUIRES(
                  context, FastBoundsCheck(byte_pos, in.size() + 1),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string b'", in, "' at index ", i));
          }
          StringPiece sub_in = in.substr(byte_pos, byte_len);
          output(i).assign(sub_in.data(), sub_in.size());
        }
      } else {
        auto pos_flat = pos_tensor.flat<T>();
        auto len_flat = len_tensor.flat<T>();
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          StringPiece in(input(i));
          const T pos = tensorflow::internal::SubtleMustCopy(pos_flat(i));
          const T len = tensorflow::internal::SubtleMustCopy(len_flat(i));
          T byte_pos = pos;
          T byte_len = len;
          switch (unit_) {
            case CharUnit::UTF8_CHAR:
              OP_REQUIRES(
                  context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string at index ", i));
              break;
            case CharUnit::BYTE:
              byte_pos = AdjustedPosIndex(byte_pos, in);
              OP_REQUIRES(
                  context, FastBoundsCheck(byte_pos, in.size() + 1),
                  errors::InvalidArgument("pos ", pos, " out of range for ",
                                          "string b'", in, "' at index ", i));
          }
          StringPiece sub_in = in.substr(byte_pos, byte_len);
          output(i).assign(sub_in.data(), sub_in.size());
        }
      }
    } else {
      BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(pos_shape),
                   false);
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument(
                      "Incompatible shapes: ", input_shape.DebugString(),
                      " vs. ", pos_shape.DebugString()));
      TensorShape output_shape = BCast::ToShape(bcast.result_shape());
      int ndims = output_shape.dims();
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                       &output_tensor));
      switch (ndims) {
        case 1: {
          auto input = input_tensor.shaped<tstring, 1>(bcast.x_reshape());
          auto output = output_tensor->shaped<tstring, 1>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 1>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 1>(bcast.y_reshape());
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 1>::Tensor pos_bcast(
              pos_buffer.shaped<T, 1>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 1>::Tensor len_bcast(
              len_buffer.shaped<T, 1>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            StringPiece in(input(input.dimension(0) > 1 ? i : 0));
            const T pos = tensorflow::internal::SubtleMustCopy(pos_bcast(i));
            const T len = tensorflow::internal::SubtleMustCopy(len_bcast(i));
            T byte_pos = pos;
            T byte_len = len;
            switch (unit_) {
              case CharUnit::UTF8_CHAR:
                OP_REQUIRES(
                    context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                    errors::InvalidArgument("pos ", pos, " out of range for ",
                                            "string at index ", i));
                break;
              case CharUnit::BYTE:
                byte_pos = AdjustedPosIndex(byte_pos, in);
                OP_REQUIRES(
                    context, FastBoundsCheck(byte_pos, in.size() + 1),
                    errors::InvalidArgument("pos ", pos, " out of range for ",
                                            "string b'", in, "' at index ", i));
            }
            StringPiece sub_in = in.substr(byte_pos, byte_len);
            output(i).assign(sub_in.data(), sub_in.size());
          }
          break;
        }
        case 2: {
          auto input = input_tensor.shaped<tstring, 2>(bcast.x_reshape());
          auto output = output_tensor->shaped<tstring, 2>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 2>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 2>(bcast.y_reshape());
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 2>::Tensor pos_bcast(
              pos_buffer.shaped<T, 2>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 2>::Tensor len_bcast(
              len_buffer.shaped<T, 2>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            for (int j = 0; j < output_shape.dim_size(1); ++j) {
              StringPiece in(input(input.dimension(0) > 1 ? i : 0,
                                   input.dimension(1) > 1 ? j : 0));
              const T pos =
                  tensorflow::internal::SubtleMustCopy(pos_bcast(i, j));
              const T len =
                  tensorflow::internal::SubtleMustCopy(len_bcast(i, j));
              T byte_pos = pos;
              T byte_len = len;
              switch (unit_) {
                case CharUnit::UTF8_CHAR:
                  OP_REQUIRES(
                      context, UpdatePosAndLenForUtf8(in, &byte_pos, &byte_len),
                      errors::InvalidArgument("pos ", pos, " out of range for ",
                                              "string at index ", i));
                  break;
                case CharUnit::BYTE:
                  byte_pos = AdjustedPosIndex(byte_pos, in);
                  OP_REQUIRES(
                      context, FastBoundsCheck(byte_pos, in.size() + 1),
                      errors::InvalidArgument("pos ", pos, " out of range for ",
                                              "string b'", in, "' at index (",
                                              i, ", ", j, ")"));
              }
              StringPiece sub_in = in.substr(byte_pos, byte_len);
              output(i, j).assign(sub_in.data(), sub_in.size());
            }
          }
          break;
        }
        default: {
          context->SetStatus(errors::Unimplemented(
              "Substr broadcast not implemented for ", ndims, " dimensions"));
        }
      }
    }
  }
 private:
  static inline T AdjustedPosIndex(const T pos_requested, const StringPiece s) {
    if (pos_requested < 0) {
      return s.size() + pos_requested;
    }
    return pos_requested;
  }
  static inline bool UpdatePosAndLenForUtf8(const StringPiece in, T* pos,
                                            T* len) {
    if (*pos >= 0) {
      return UpdatePositivePosAndLenForUtf8(in, *pos, *len, pos, len);
    } else {
      return UpdateNegativePosAndLenForUtf8(in, *pos, *len, pos, len);
    }
  }
  static bool UpdatePositivePosAndLenForUtf8(const StringPiece in, const T pos,
                                             const T len, T* char_pos,
                                             T* char_len) {
    *char_pos = 0;
    if (!ForwardNUTF8CharPositions(in, pos, char_pos)) {
      return false;
    }
    *char_len = *char_pos;
    ForwardNUTF8CharPositions(in, len, char_len);
    *char_len = *char_len - *char_pos;
    return true;
  }
  static bool UpdateNegativePosAndLenForUtf8(const StringPiece in, const T pos,
                                             const T len, T* char_pos,
                                             T* char_len) {
    *char_len = in.size();
    T utf8_chars_to_skip = -pos - len;
    if (utf8_chars_to_skip < 0) {
      utf8_chars_to_skip = 0;
    }
    if (!BackNUTF8CharPositions(in, utf8_chars_to_skip, char_len)) {
      return false;
    }
    *char_pos = *char_len;
    if (!BackNUTF8CharPositions(in, -pos - utf8_chars_to_skip, char_pos)) {
      return false;
    }
    *char_len = *char_len - *char_pos;
    return true;
  }
  CharUnit unit_ = CharUnit::BYTE;
};
#define REGISTER_SUBSTR(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Substr").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SubstrOp<type>);
REGISTER_SUBSTR(int32);
REGISTER_SUBSTR(int64_t);
}  