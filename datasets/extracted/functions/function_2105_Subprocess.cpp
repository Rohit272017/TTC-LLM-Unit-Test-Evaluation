#ifndef TENSORSTORE_INTERNAL_SUBPROCESS_H_
#define TENSORSTORE_INTERNAL_SUBPROCESS_H_
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
namespace internal {
class Subprocess;
struct SubprocessOptions {
  std::string executable;         
  std::vector<std::string> args;  
  std::optional<absl::flat_hash_map<std::string, std::string>> env;
  struct Inherit {};
  struct Ignore {};
  struct Redirect {
    std::string filename;
  };
  std::variant<Ignore, Redirect> stdin_action = Ignore{};
  std::variant<Inherit, Ignore, Redirect> stdout_action = Inherit{};
  std::variant<Inherit, Ignore, Redirect> stderr_action = Inherit{};
};
Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options);
class Subprocess {
 public:
  Subprocess(const Subprocess&) = default;
  Subprocess& operator=(const Subprocess&) = default;
  Subprocess(Subprocess&&) = default;
  Subprocess& operator=(Subprocess&&) = default;
  ~Subprocess();
  absl::Status Kill(int signal = 9) const;
  Result<int> Join(bool block = true) const;
 private:
  friend Result<Subprocess> SpawnSubprocess(const SubprocessOptions& options);
  struct Impl;
  Subprocess(std::shared_ptr<Subprocess::Impl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<Subprocess::Impl> impl_;
};
}  
}  
#endif  