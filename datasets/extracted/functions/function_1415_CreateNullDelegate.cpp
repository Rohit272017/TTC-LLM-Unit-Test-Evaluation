#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
namespace tflite {
namespace tools {
TfLiteDelegatePtr CreateNullDelegate() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteOpaqueDelegate*) {});
}
void ProvidedDelegateList::AddAllDelegateParams() const {
  for (const auto& provider : providers_) {
    params_->Merge(provider->DefaultParams());
  }
}
void ProvidedDelegateList::AppendCmdlineFlags(std::vector<Flag>& flags) const {
  for (const auto& provider : providers_) {
    auto delegate_flags = provider->CreateFlags(params_);
    flags.insert(flags.end(), delegate_flags.begin(), delegate_flags.end());
  }
}
void ProvidedDelegateList::RemoveCmdlineFlag(std::vector<Flag>& flags,
                                             const std::string& name) const {
  decltype(flags.begin()) it;
  for (it = flags.begin(); it < flags.end();) {
    if (it->GetFlagName() == name) {
      it = flags.erase(it);
    } else {
      ++it;
    }
  }
}
std::vector<ProvidedDelegateList::ProvidedDelegate>
ProvidedDelegateList::CreateAllRankedDelegates(const ToolParams& params) const {
  std::vector<ProvidedDelegateList::ProvidedDelegate> delegates;
  for (const auto& provider : providers_) {
    auto ptr_rank = provider->CreateRankedTfLiteDelegate(params);
    if (ptr_rank.first == nullptr) continue;
    static bool already_logged = false;
    if (!already_logged) {
      TFLITE_LOG(INFO) << provider->GetName() << " delegate created.";
#ifndef NDEBUG
      provider->LogParams(params, false);
#endif
      already_logged = true;
    }
    ProvidedDelegateList::ProvidedDelegate info;
    info.provider = provider.get();
    info.delegate = std::move(ptr_rank.first);
    info.rank = ptr_rank.second;
    delegates.emplace_back(std::move(info));
  }
  std::sort(delegates.begin(), delegates.end(),
            [](const ProvidedDelegateList::ProvidedDelegate& a,
               const ProvidedDelegateList::ProvidedDelegate& b) {
              return a.rank < b.rank;
            });
  return delegates;
}
}  
}  