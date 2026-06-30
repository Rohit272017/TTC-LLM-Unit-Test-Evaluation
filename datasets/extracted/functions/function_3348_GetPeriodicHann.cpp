#include "tensorflow/lite/kernels/internal/spectrogram.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "third_party/fft2d/fft.h"
namespace tflite {
namespace internal {
using std::complex;
namespace {
void GetPeriodicHann(int window_length, std::vector<double>* window) {
  const double pi = std::atan(1.0) * 4.0;
  window->resize(window_length);
  for (int i = 0; i < window_length; ++i) {
    (*window)[i] = 0.5 - 0.5 * cos((2.0 * pi * i) / window_length);
  }
}
}  
bool Spectrogram::Initialize(int window_length, int step_length) {
  std::vector<double> window;
  GetPeriodicHann(window_length, &window);
  return Initialize(window, step_length);
}
inline int Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int log = 0;
  uint32_t value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}
inline int Log2Ceiling(uint32_t n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  
    return floor;
  else
    return floor + 1;
}
inline uint32_t NextPowerOfTwo(uint32_t value) {
  int exponent = Log2Ceiling(value);
  return 1 << exponent;
}
bool Spectrogram::Initialize(const std::vector<double>& window,
                             int step_length) {
  window_length_ = window.size();
  window_ = window;  
  if (window_length_ < 2) {
    initialized_ = false;
    return false;
  }
  step_length_ = step_length;
  if (step_length_ < 1) {
    initialized_ = false;
    return false;
  }
  fft_length_ = NextPowerOfTwo(window_length_);
  output_frequency_channels_ = 1 + fft_length_ / 2;
  fft_input_output_.assign(fft_length_ + 2, 0.0);
  int half_fft_length = fft_length_ / 2;
  fft_double_working_area_.assign(half_fft_length, 0.0);
  fft_integer_working_area_.assign(2 + static_cast<int>(sqrt(half_fft_length)),
                                   0);
  fft_integer_working_area_[0] = 0;
  input_queue_.clear();
  samples_to_next_step_ = window_length_;
  initialized_ = true;
  return true;
}
template <class InputSample, class OutputSample>
bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<complex<OutputSample>>>* output) {
  if (!initialized_) {
    return false;
  }
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    ProcessCoreFFT();  
    output->resize(output->size() + 1);
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      spectrogram_slice[i] = complex<OutputSample>(
          fft_input_output_[2 * i], fft_input_output_[2 * i + 1]);
    }
  }
  return true;
}
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<float>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<float>& input,
    std::vector<std::vector<complex<double>>>*);
template bool Spectrogram::ComputeComplexSpectrogram(
    const std::vector<double>& input,
    std::vector<std::vector<complex<double>>>*);
template <class InputSample, class OutputSample>
bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<InputSample>& input,
    std::vector<std::vector<OutputSample>>* output) {
  if (!initialized_) {
    return false;
  }
  output->clear();
  int input_start = 0;
  while (GetNextWindowOfSamples(input, &input_start)) {
    ProcessCoreFFT();  
    output->resize(output->size() + 1);
    auto& spectrogram_slice = output->back();
    spectrogram_slice.resize(output_frequency_channels_);
    for (int i = 0; i < output_frequency_channels_; ++i) {
      const double re = fft_input_output_[2 * i];
      const double im = fft_input_output_[2 * i + 1];
      spectrogram_slice[i] = re * re + im * im;
    }
  }
  return true;
}
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<float>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<float>& input, std::vector<std::vector<double>>*);
template bool Spectrogram::ComputeSquaredMagnitudeSpectrogram(
    const std::vector<double>& input, std::vector<std::vector<double>>*);
template <class InputSample>
bool Spectrogram::GetNextWindowOfSamples(const std::vector<InputSample>& input,
                                         int* input_start) {
  auto input_it = input.begin() + *input_start;
  int input_remaining = input.end() - input_it;
  if (samples_to_next_step_ > input_remaining) {
    input_queue_.insert(input_queue_.end(), input_it, input.end());
    *input_start += input_remaining;  
    samples_to_next_step_ -= input_remaining;
    return false;  
  } else {
    input_queue_.insert(input_queue_.end(), input_it,
                        input_it + samples_to_next_step_);
    *input_start += samples_to_next_step_;
    input_queue_.erase(
        input_queue_.begin(),
        input_queue_.begin() + input_queue_.size() - window_length_);
    samples_to_next_step_ = step_length_;  
    return true;  
  }
}
void Spectrogram::ProcessCoreFFT() {
  for (int j = 0; j < window_length_; ++j) {
    fft_input_output_[j] = input_queue_[j] * window_[j];
  }
  for (int j = window_length_; j < fft_length_; ++j) {
    fft_input_output_[j] = 0.0;
  }
  const int kForwardFFT = 1;  
  rdft(fft_length_, kForwardFFT, &fft_input_output_[0],
       &fft_integer_working_area_[0], &fft_double_working_area_[0]);
  fft_input_output_[fft_length_] = fft_input_output_[1];
  fft_input_output_[fft_length_ + 1] = 0;
  fft_input_output_[1] = 0;
}
}  
}  