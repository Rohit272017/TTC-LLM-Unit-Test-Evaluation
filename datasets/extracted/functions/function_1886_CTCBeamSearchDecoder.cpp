#ifndef TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_SEARCH_H_
#define TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_SEARCH_H_
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include "Eigen/Core"  
#include "tensorflow/lite/kernels/ctc/ctc_beam_entry.h"
#include "tensorflow/lite/kernels/ctc/ctc_beam_scorer.h"
#include "tensorflow/lite/kernels/ctc/ctc_decoder.h"
#include "tensorflow/lite/kernels/ctc/ctc_loss_util.h"
#include "tensorflow/lite/kernels/ctc/top_n.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
namespace tflite {
namespace custom {
namespace ctc {
template <typename CTCBeamState = ctc_beam_search::EmptyBeamState,
          typename CTCBeamComparer =
              ctc_beam_search::BeamComparer<CTCBeamState>>
class CTCBeamSearchDecoder : public CTCDecoder {
  typedef ctc_beam_search::BeamEntry<CTCBeamState> BeamEntry;
  typedef ctc_beam_search::BeamRoot<CTCBeamState> BeamRoot;
  typedef ctc_beam_search::BeamProbability BeamProbability;
 public:
  typedef BaseBeamScorer<CTCBeamState> DefaultBeamScorer;
  CTCBeamSearchDecoder(int num_classes, int beam_width,
                       BaseBeamScorer<CTCBeamState>* scorer, int batch_size = 1,
                       bool merge_repeated = false)
      : CTCDecoder(num_classes, batch_size, merge_repeated),
        beam_width_(beam_width),
        leaves_(beam_width),
        beam_scorer_(scorer) {
    Reset();
  }
  ~CTCBeamSearchDecoder() override {}
  bool Decode(const CTCDecoder::SequenceLength& seq_len,
              const std::vector<CTCDecoder::Input>& input,
              std::vector<CTCDecoder::Output>* output,
              CTCDecoder::ScoreOutput* scores) override;
  template <typename Vector>
  void Step(const Vector& raw_input);
  template <typename Vector>
  float GetTopK(const int K, const Vector& input,
                std::vector<float>* top_k_logits,
                std::vector<int>* top_k_indices);
  BaseBeamScorer<CTCBeamState>* GetBeamScorer() const { return beam_scorer_; }
  void SetLabelSelectionParameters(int label_selection_size,
                                   float label_selection_margin) {
    label_selection_size_ = label_selection_size;
    label_selection_margin_ = label_selection_margin;
  }
  void Reset();
  bool TopPaths(int n, std::vector<std::vector<int>>* paths,
                std::vector<float>* log_probs, bool merge_repeated) const;
 private:
  int beam_width_;
  int label_selection_size_ = 0;       
  float label_selection_margin_ = -1;  
  gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
  std::unique_ptr<BeamRoot> beam_root_;
  BaseBeamScorer<CTCBeamState>* beam_scorer_;
  CTCBeamSearchDecoder(const CTCBeamSearchDecoder&) = delete;
  void operator=(const CTCBeamSearchDecoder&) = delete;
};
template <typename CTCBeamState, typename CTCBeamComparer>
bool CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Decode(
    const CTCDecoder::SequenceLength& seq_len,
    const std::vector<CTCDecoder::Input>& input,
    std::vector<CTCDecoder::Output>* output, ScoreOutput* scores) {
  std::vector<std::vector<int>> beams;
  std::vector<float> beam_log_probabilities;
  int top_n = output->size();
  if (std::any_of(output->begin(), output->end(),
                  [this](const CTCDecoder::Output& output) -> bool {
                    return output.size() < this->batch_size_;
                  })) {
    return false;
  }
  if (scores->rows() < batch_size_ || scores->cols() < top_n) {
    return false;
  }
  for (int b = 0; b < batch_size_; ++b) {
    int seq_len_b = seq_len[b];
    Reset();
    for (int t = 0; t < seq_len_b; ++t) {
      Step(input[t].row(b));
    }  
    std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
    leaves_.Reset();
    for (int i = 0; i < branches->size(); ++i) {
      BeamEntry* entry = (*branches)[i];
      beam_scorer_->ExpandStateEnd(&entry->state);
      entry->newp.total +=
          beam_scorer_->GetStateEndExpansionScore(entry->state);
      leaves_.push(entry);
    }
    bool status =
        TopPaths(top_n, &beams, &beam_log_probabilities, merge_repeated_);
    if (!status) {
      return status;
    }
    TFLITE_DCHECK_EQ(top_n, beam_log_probabilities.size());
    TFLITE_DCHECK_EQ(beams.size(), beam_log_probabilities.size());
    for (int i = 0; i < top_n; ++i) {
      (*output)[i][b].swap(beams[i]);
      (*scores)(b, i) = -beam_log_probabilities[i];
    }
  }  
  return true;
}
template <typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
float CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::GetTopK(
    const int K, const Vector& input, std::vector<float>* top_k_logits,
    std::vector<int>* top_k_indices) {
  TFLITE_DCHECK_EQ(num_classes_, input.size());
  top_k_logits->clear();
  top_k_indices->clear();
  top_k_logits->resize(K, -INFINITY);
  top_k_indices->resize(K, -1);
  for (int j = 0; j < num_classes_ - 1; ++j) {
    const float logit = input(j);
    if (logit > (*top_k_logits)[K - 1]) {
      int k = K - 1;
      while (k > 0 && logit > (*top_k_logits)[k - 1]) {
        (*top_k_logits)[k] = (*top_k_logits)[k - 1];
        (*top_k_indices)[k] = (*top_k_indices)[k - 1];
        k--;
      }
      (*top_k_logits)[k] = logit;
      (*top_k_indices)[k] = j;
    }
  }
  return std::max((*top_k_logits)[0], input(num_classes_ - 1));
}
template <typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Step(
    const Vector& raw_input) {
  std::vector<float> top_k_logits;
  std::vector<int> top_k_indices;
  const bool top_k =
      (label_selection_size_ > 0 && label_selection_size_ < raw_input.size());
  const int max_classes = top_k ? label_selection_size_ : (num_classes_ - 1);
  float max_coeff;
  if (top_k) {
    max_coeff = GetTopK(label_selection_size_, raw_input, &top_k_logits,
                        &top_k_indices);
  } else {
    max_coeff = raw_input.maxCoeff();
  }
  float logsumexp = 0.0;
  for (int j = 0; j < raw_input.size(); ++j) {
    logsumexp += Eigen::numext::exp(raw_input(j) - max_coeff);
  }
  logsumexp = Eigen::numext::log(logsumexp);
  float norm_offset = max_coeff + logsumexp;
  const float label_selection_input_min =
      (label_selection_margin_ >= 0) ? (max_coeff - label_selection_margin_)
                                     : -std::numeric_limits<float>::infinity();
  TFLITE_DCHECK_EQ(num_classes_, raw_input.size());
  std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
  leaves_.Reset();
  for (BeamEntry* b : *branches) {
    b->oldp = b->newp;
  }
  for (BeamEntry* b : *branches) {
    if (b->parent != nullptr) {  
      if (b->parent->Active()) {
        float previous = (b->label == b->parent->label) ? b->parent->oldp.blank
                                                        : b->parent->oldp.total;
        b->newp.label =
            LogSumExp(b->newp.label,
                      beam_scorer_->GetStateExpansionScore(b->state, previous));
      }
      b->newp.label += raw_input(b->label) - norm_offset;
    }
    b->newp.blank = b->oldp.total + raw_input(blank_index_) - norm_offset;
    b->newp.total = LogSumExp(b->newp.blank, b->newp.label);
    leaves_.push(b);
  }
  for (BeamEntry* b : *branches) {
    auto is_candidate = [this](const BeamProbability& prob) {
      return (prob.total > kLogZero &&
              (leaves_.size() < beam_width_ ||
               prob.total > leaves_.peek_bottom()->newp.total));
    };
    if (!is_candidate(b->oldp)) {
      continue;
    }
    for (int ind = 0; ind < max_classes; ind++) {
      const int label = top_k ? top_k_indices[ind] : ind;
      const float logit = top_k ? top_k_logits[ind] : raw_input(ind);
      if (logit < label_selection_input_min) {
        continue;
      }
      BeamEntry& c = b->GetChild(label);
      if (!c.Active()) {
        c.newp.blank = kLogZero;
        beam_scorer_->ExpandState(b->state, b->label, &c.state, c.label);
        float previous = (c.label == b->label) ? b->oldp.blank : b->oldp.total;
        c.newp.label = logit - norm_offset +
                       beam_scorer_->GetStateExpansionScore(c.state, previous);
        c.newp.total = c.newp.label;
        if (is_candidate(c.newp)) {
          if (leaves_.size() == beam_width_) {
            BeamEntry* bottom = leaves_.peek_bottom();
            bottom->newp.Reset();
          }
          leaves_.push(&c);
        } else {
          c.oldp.Reset();
          c.newp.Reset();
        }
      }
    }
  }  
}
template <typename CTCBeamState, typename CTCBeamComparer>
void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Reset() {
  leaves_.Reset();
  beam_root_.reset(new BeamRoot(nullptr, -1));
  beam_root_->RootEntry()->newp.total = 0.0;  
  beam_root_->RootEntry()->newp.blank = 0.0;  
  leaves_.push(beam_root_->RootEntry());
  beam_scorer_->InitializeState(&beam_root_->RootEntry()->state);
}
template <typename CTCBeamState, typename CTCBeamComparer>
bool CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::TopPaths(
    int n, std::vector<std::vector<int>>* paths, std::vector<float>* log_probs,
    bool merge_repeated) const {
  TFLITE_DCHECK(paths);
  TFLITE_DCHECK(log_probs);
  paths->clear();
  log_probs->clear();
  if (n > beam_width_) {
    return false;
  }
  if (n > leaves_.size()) {
    return false;
  }
  gtl::TopN<BeamEntry*, CTCBeamComparer> top_branches(n);
  for (auto it = leaves_.unsorted_begin(); it != leaves_.unsorted_end(); ++it) {
    top_branches.push(*it);
  }
  std::unique_ptr<std::vector<BeamEntry*>> branches(top_branches.Extract());
  for (int i = 0; i < n; ++i) {
    BeamEntry* e((*branches)[i]);
    paths->push_back(e->LabelSeq(merge_repeated));
    log_probs->push_back(e->newp.total);
  }
  return true;
}
}  
}  
}  
#endif  