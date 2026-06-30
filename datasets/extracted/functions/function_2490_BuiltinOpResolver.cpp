#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tflite_with_xnnpack_optional.h"
namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_NUMERIC_VERIFY();
TfLiteRegistration* Register_AUDIO_SPECTROGRAM();
TfLiteRegistration* Register_MFCC();
TfLiteRegistration* Register_DETECTION_POSTPROCESS();
}  
namespace builtin {
BuiltinOpResolver::BuiltinOpResolver() {
  AddBuiltin(BuiltinOperator_ABS, Register_ABS(),  1,
              5);
  AddBuiltin(BuiltinOperator_HARD_SWISH, Register_HARD_SWISH());
  AddBuiltin(BuiltinOperator_RELU, Register_RELU(),  1,
              3);
  AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
  AddBuiltin(BuiltinOperator_RELU_0_TO_1, Register_RELU_0_TO_1());
  AddBuiltin(BuiltinOperator_RELU6, Register_RELU6(),  1,
              3);
  AddBuiltin(BuiltinOperator_TANH, Register_TANH(),  1,
              3);
  AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC(),
              1,
              3);
  AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D(),
              1,
              3);
  AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D(),
              1,
              3);
  AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
  AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(),
              1,
              8);
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
              1,
              7);
  AddBuiltin(BuiltinOperator_SVDF, Register_SVDF(),
              1,
              4);
  AddBuiltin(BuiltinOperator_RNN, Register_RNN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN,
             Register_BIDIRECTIONAL_SEQUENCE_RNN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN,
             Register_UNIDIRECTIONAL_SEQUENCE_RNN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP(),
              1,
              4);
  AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
             Register_EMBEDDING_LOOKUP_SPARSE());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(),
              1,
              13);
  AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
  AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
  AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX(),
              1,
              3);
  AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(),
              1,
              4);
  AddBuiltin(BuiltinOperator_ADD, Register_ADD(),
              1,
              5);
  AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND(),
              1,
              4);
  AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND(),
              1,
              4);
  AddBuiltin(BuiltinOperator_MUL, Register_MUL(),  1,
              7);
  AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION(),
              1,
              2);
  AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
             Register_LOCAL_RESPONSE_NORMALIZATION());
  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(),  1,
              4);
  AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
             Register_BIDIRECTIONAL_SEQUENCE_LSTM(),  1,
              3);
  AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
             Register_UNIDIRECTIONAL_SEQUENCE_LSTM(),  1,
              4);
  AddBuiltin(BuiltinOperator_PAD, Register_PAD(),  1,
              4);
  AddBuiltin(BuiltinOperator_PADV2, Register_PADV2(),  1,
              4);
  AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
  AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR(),
              1,
              4);
  AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
             Register_RESIZE_NEAREST_NEIGHBOR(),
              1,
              4);
  AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
  AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH(),
              1,
              2);
  AddBuiltin(BuiltinOperator_DEPTH_TO_SPACE, Register_DEPTH_TO_SPACE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_GATHER, Register_GATHER(),
              1,
              7);
  AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE(),
              1,
              6);
  AddBuiltin(BuiltinOperator_MEAN, Register_MEAN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_DIV, Register_DIV(),
              1,
              2);
  AddBuiltin(BuiltinOperator_SUB, Register_SUB(),
              1,
              5);
  AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(),
              1,
              4);
  AddBuiltin(BuiltinOperator_SPLIT_V, Register_SPLIT_V(),
              1,
              2);
  AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE(),
              1,
              8);
  AddBuiltin(BuiltinOperator_EXP, Register_EXP(),
              1,
              2);
  AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2(),
              1,
              3);
  AddBuiltin(BuiltinOperator_LOG, Register_LOG(),
              1,
              2);
  AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX(),
              1,
              2);
  AddBuiltin(BuiltinOperator_CAST, Register_CAST(),
              1,
              6);
  AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(),
              1,
              6);
  AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
  AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM(),
              1,
              4);
  AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM(),
              1,
              4);
  AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(),
              1,
              3);
  AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(),
              1,
              2);
  AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(),
              1,
              3);
  AddBuiltin(BuiltinOperator_LESS, Register_LESS(),
              1,
              3);
  AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(),
              1,
              2);
  AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
  AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
  AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());
  AddBuiltin(BuiltinOperator_NEG, Register_NEG());
  AddBuiltin(BuiltinOperator_SELECT, Register_SELECT(),
              1,
              4);
  AddBuiltin(BuiltinOperator_SELECT_V2, Register_SELECT_V2(),
              1,
              2);
  AddBuiltin(BuiltinOperator_SLICE, Register_SLICE(),
              1,
              6);
  AddBuiltin(BuiltinOperator_SIN, Register_SIN());
  AddBuiltin(BuiltinOperator_COS, Register_COS());
  AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV(),
              1,
              5);
  AddBuiltin(BuiltinOperator_TILE, Register_TILE(),
              1,
              3);
  AddBuiltin(BuiltinOperator_SUM, Register_SUM(),
              1,
              2);
  AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD(),
              1,
              2);
  AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX(),
              1,
              3);
  AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN(),
              1,
              3);
  AddBuiltin(BuiltinOperator_REDUCE_ANY, Register_REDUCE_ANY());
  AddBuiltin(BuiltinOperator_REDUCE_ALL, Register_REDUCE_ALL());
  AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS());
  AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE(),
              1,
              3);
  AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(),
              1,
              4);
  AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(),
              1,
              3);
  AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
  AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT(),
              1,
              3);
  AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE());
  AddBuiltin(BuiltinOperator_RANK, Register_RANK());
  AddBuiltin(BuiltinOperator_POW, Register_POW());
  AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT(), 1, 2);
  AddBuiltin(BuiltinOperator_PACK, Register_PACK(),
              1,
              4);
  AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT());
  AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
  AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
  AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
  AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK(),
              1,
              4);
  AddBuiltin(BuiltinOperator_FLOOR_DIV, Register_FLOOR_DIV(),
              1,
              3);
  AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
  AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE());
  AddBuiltin(BuiltinOperator_FLOOR_MOD, Register_FLOOR_MOD(),
              1,
              2);
  AddBuiltin(BuiltinOperator_RANGE, Register_RANGE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_LEAKY_RELU, Register_LEAKY_RELU(),
              1,
              2);
  AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, Register_SQUARED_DIFFERENCE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_FILL, Register_FILL(),
              1,
              4);
  AddBuiltin(BuiltinOperator_MIRROR_PAD, Register_MIRROR_PAD(),
              1,
              3);
  AddBuiltin(BuiltinOperator_UNIQUE, Register_UNIQUE());
  AddBuiltin(BuiltinOperator_REVERSE_V2, Register_REVERSE_V2(),
              1,
              3);
  AddBuiltin(BuiltinOperator_ADD_N, Register_ADD_N());
  AddBuiltin(BuiltinOperator_GATHER_ND, Register_GATHER_ND(),
              1,
              5);
  AddBuiltin(BuiltinOperator_WHERE, Register_WHERE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_ELU, Register_ELU());
  AddBuiltin(BuiltinOperator_REVERSE_SEQUENCE, Register_REVERSE_SEQUENCE());
  AddBuiltin(BuiltinOperator_MATRIX_DIAG, Register_MATRIX_DIAG());
  AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE(),
              1,
              3);
  AddBuiltin(BuiltinOperator_MATRIX_SET_DIAG, Register_MATRIX_SET_DIAG());
  AddBuiltin(BuiltinOperator_IF, tflite::ops::builtin::Register_IF());
  AddBuiltin(BuiltinOperator_WHILE, tflite::ops::builtin::Register_WHILE());
  AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V4,
             Register_NON_MAX_SUPPRESSION_V4());
  AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V5,
             Register_NON_MAX_SUPPRESSION_V5());
  AddBuiltin(BuiltinOperator_SCATTER_ND, Register_SCATTER_ND());
  AddBuiltin(BuiltinOperator_DENSIFY, Register_DENSIFY());
  AddBuiltin(BuiltinOperator_SEGMENT_SUM, Register_SEGMENT_SUM());
  AddBuiltin(BuiltinOperator_BATCH_MATMUL, Register_BATCH_MATMUL(),
              1,
              4);
  AddBuiltin(BuiltinOperator_CUMSUM, Register_CUMSUM());
  AddBuiltin(BuiltinOperator_BROADCAST_TO, Register_BROADCAST_TO(),
              2,
              3);
  AddBuiltin(BuiltinOperator_CALL_ONCE,
             tflite::ops::builtin::Register_CALL_ONCE());
  AddBuiltin(BuiltinOperator_RFFT2D, Register_RFFT2D());
  AddBuiltin(BuiltinOperator_CONV_3D, Register_CONV_3D());
  AddBuiltin(BuiltinOperator_IMAG, Register_IMAG());
  AddBuiltin(BuiltinOperator_REAL, Register_REAL());
  AddBuiltin(BuiltinOperator_COMPLEX_ABS, Register_COMPLEX_ABS());
  AddBuiltin(BuiltinOperator_BROADCAST_ARGS, Register_BROADCAST_ARGS());
  AddBuiltin(BuiltinOperator_HASHTABLE, Register_HASHTABLE());
  AddBuiltin(BuiltinOperator_HASHTABLE_FIND, Register_HASHTABLE_FIND());
  AddBuiltin(BuiltinOperator_HASHTABLE_IMPORT, Register_HASHTABLE_IMPORT());
  AddBuiltin(BuiltinOperator_HASHTABLE_SIZE, Register_HASHTABLE_SIZE());
  AddBuiltin(BuiltinOperator_CONV_3D_TRANSPOSE, Register_CONV_3D_TRANSPOSE());
  AddBuiltin(BuiltinOperator_VAR_HANDLE, Register_VAR_HANDLE());
  AddBuiltin(BuiltinOperator_READ_VARIABLE, Register_READ_VARIABLE());
  AddBuiltin(BuiltinOperator_ASSIGN_VARIABLE, Register_ASSIGN_VARIABLE());
  AddBuiltin(BuiltinOperator_MULTINOMIAL, Register_MULTINOMIAL());
  AddBuiltin(BuiltinOperator_RANDOM_STANDARD_NORMAL,
             Register_RANDOM_STANDARD_NORMAL());
  AddBuiltin(BuiltinOperator_BUCKETIZE, Register_BUCKETIZE());
  AddBuiltin(BuiltinOperator_RANDOM_UNIFORM, Register_RANDOM_UNIFORM());
  AddBuiltin(BuiltinOperator_GELU, Register_GELU(),
              1,
              2);
  AddBuiltin(BuiltinOperator_DYNAMIC_UPDATE_SLICE,
             Register_DYNAMIC_UPDATE_SLICE(),
              1,
              2);
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_PROD,
             Register_UNSORTED_SEGMENT_PROD());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MAX,
             Register_UNSORTED_SEGMENT_MAX());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MIN,
             Register_UNSORTED_SEGMENT_MIN());
  AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_SUM,
             Register_UNSORTED_SEGMENT_SUM());
  AddBuiltin(BuiltinOperator_ATAN2, Register_ATAN2());
  AddBuiltin(BuiltinOperator_SIGN, Register_SIGN(),
              1,
              2);
  AddBuiltin(BuiltinOperator_BITCAST, Register_BITCAST());
  AddBuiltin(BuiltinOperator_BITWISE_XOR, Register_BITWISE_XOR());
  AddBuiltin(BuiltinOperator_RIGHT_SHIFT, Register_RIGHT_SHIFT());
  AddBuiltin(BuiltinOperator_STABLEHLO_SCATTER, Register_STABLEHLO_SCATTER());
  AddBuiltin(BuiltinOperator_DILATE, Register_DILATE());
  AddBuiltin(BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR,
             Register_STABLEHLO_RNG_BIT_GENERATOR());
  AddBuiltin(BuiltinOperator_REDUCE_WINDOW, Register_REDUCE_WINDOW());
  AddBuiltin(BuiltinOperator_STABLEHLO_REDUCE_WINDOW,
             Register_STABLEHLO_REDUCE_WINDOW());
  AddBuiltin(BuiltinOperator_STABLEHLO_GATHER, Register_STABLEHLO_GATHER());
  AddBuiltin(BuiltinOperator_STABLEHLO_ADD, Register_STABLEHLO_ADD());
  AddBuiltin(BuiltinOperator_STABLEHLO_AND, Register_STABLEHLO_AND());
  AddBuiltin(BuiltinOperator_STABLEHLO_MULTIPLY, Register_STABLEHLO_MULTIPLY());
  AddBuiltin(BuiltinOperator_STABLEHLO_MAXIMUM, Register_STABLEHLO_MAXIMUM());
  AddBuiltin(BuiltinOperator_STABLEHLO_MINIMUM, Register_STABLEHLO_MINIMUM());
  AddBuiltin(BuiltinOperator_STABLEHLO_SHIFT_LEFT,
             Register_STABLEHLO_SHIFT_LEFT());
  AddBuiltin(BuiltinOperator_STABLEHLO_PAD, Register_STABLEHLO_PAD());
  AddBuiltin(BuiltinOperator_STABLEHLO_COMPOSITE,
             Register_STABLEHLO_COMPOSITE());
  AddCustom("NumericVerify", tflite::ops::custom::Register_NUMERIC_VERIFY());
  AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
  AddCustom("AudioSpectrogram",
            tflite::ops::custom::Register_AUDIO_SPECTROGRAM());
  AddCustom("TFLite_Detection_PostProcess",
            tflite::ops::custom::Register_DETECTION_POSTPROCESS());
  may_directly_contain_user_defined_ops_ = false;
  delegate_creators_.push_back([](TfLiteContext* context) {
    return tflite::MaybeCreateXNNPACKDelegate(context,
                                              XNNPackQS8Options::default_value);
  });
}
BuiltinOpResolverWithXNNPACK::BuiltinOpResolverWithXNNPACK(
    bool enable_xnnpack_unsigned_quantized) {
  delegate_creators_.clear();
  XNNPackQS8Options xnnpack_qs8_options = enable_xnnpack_unsigned_quantized
                                              ? XNNPackQS8Options::enabled
                                              : XNNPackQS8Options::disabled;
  delegate_creators_.push_back([xnnpack_qs8_options](TfLiteContext* context) {
    return tflite::MaybeCreateXNNPACKDelegate(context, xnnpack_qs8_options);
  });
}
}  
}  
}  