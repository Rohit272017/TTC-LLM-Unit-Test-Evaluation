#include "tensorflow/c/eager/c_api_experimental_reader.h"
#include "tensorflow/c/eager/tfe_monitoring_reader_internal.h"
template <typename... LabelType>
int64_t TFE_MonitoringCounterReader::Read(const LabelType&... labels) {
  return counter->Read(labels...);
}
TFE_MonitoringCounterReader* TFE_MonitoringNewCounterReader(const char* name) {
  auto* result = new TFE_MonitoringCounterReader(name);
  return result;
}
int64_t TFE_MonitoringReadCounter0(TFE_MonitoringCounterReader* cell_reader) {
  int64_t result = cell_reader->Read();
  return result;
}
int64_t TFE_MonitoringReadCounter1(TFE_MonitoringCounterReader* cell_reader,
                                   const char* label) {
  int64_t result = cell_reader->Read(label);
  return result;
}