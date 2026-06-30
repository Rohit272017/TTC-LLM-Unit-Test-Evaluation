#include <ctype.h>
#include <vector>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace {
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor->type, kTfLiteString);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, output_tensor->type, kTfLiteString);
  return kTfLiteOk;
}
bool ShouldIncludeCurrentNgram(const TfLiteSkipGramParams* params, int size) {
  if (size <= 0) {
    return false;
  }
  if (params->include_all_ngrams) {
    return size <= params->ngram_size;
  } else {
    return size == params->ngram_size;
  }
}
bool ShouldStepInRecursion(const TfLiteSkipGramParams* params,
                           const std::vector<int>& stack, int stack_idx,
                           int num_words) {
  if (stack_idx < params->ngram_size && stack[stack_idx] + 1 < num_words) {
    if (stack_idx == 0) {
      return true;
    }
    if (stack[stack_idx] - stack[stack_idx - 1] <= params->max_skip_size) {
      return true;
    }
  }
  return false;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSkipGramParams*>(node->builtin_data);
  std::vector<StringRef> words;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  tflite::StringRef strref = tflite::GetString(input, 0);
  int prev_idx = 0;
  for (size_t i = 1; i < strref.len; i++) {
    if (isspace(*(strref.str + i))) {
      if (i > prev_idx && !isspace(*(strref.str + prev_idx))) {
        words.push_back({strref.str + prev_idx, i - prev_idx});
      }
      prev_idx = i + 1;
    }
  }
  if (strref.len > prev_idx) {
    words.push_back({strref.str + prev_idx, strref.len - prev_idx});
  }
  tflite::DynamicBuffer buf;
  if (words.size() < params->ngram_size) {
    buf.WriteToTensorAsVector(GetOutput(context, node, 0));
    return kTfLiteOk;
  }
  std::vector<int> stack(params->ngram_size, 0);
  int stack_idx = 1;
  int num_words = words.size();
  while (stack_idx >= 0) {
    if (ShouldStepInRecursion(params, stack, stack_idx, num_words)) {
      stack[stack_idx]++;
      stack_idx++;
      if (stack_idx < params->ngram_size) {
        stack[stack_idx] = stack[stack_idx - 1];
      }
    } else {
      if (ShouldIncludeCurrentNgram(params, stack_idx)) {
        std::vector<StringRef> gram(stack_idx);
        for (int i = 0; i < stack_idx; i++) {
          gram[i] = words[stack[i]];
        }
        buf.AddJoinedString(gram, ' ');
      }
      stack_idx--;
    }
  }
  buf.WriteToTensorAsVector(GetOutput(context, node, 0));
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_SKIP_GRAM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}
}  
}  
}  