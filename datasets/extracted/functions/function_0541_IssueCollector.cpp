#ifndef THIRD_PARTY_CEL_CPP_RUNTIME_INTERNAL_ISSUE_COLLECTOR_H_
#define THIRD_PARTY_CEL_CPP_RUNTIME_INTERNAL_ISSUE_COLLECTOR_H_
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "runtime/runtime_issue.h"
namespace cel::runtime_internal {
class IssueCollector {
 public:
  explicit IssueCollector(RuntimeIssue::Severity severity_limit)
      : severity_limit_(severity_limit) {}
  IssueCollector(const IssueCollector&) = delete;
  IssueCollector& operator=(const IssueCollector&) = delete;
  IssueCollector(IssueCollector&&) = default;
  IssueCollector& operator=(IssueCollector&&) = default;
  absl::Status AddIssue(RuntimeIssue issue) {
    issues_.push_back(std::move(issue));
    if (issues_.back().severity() >= severity_limit_) {
      return issues_.back().ToStatus();
    }
    return absl::OkStatus();
  }
  absl::Span<const RuntimeIssue> issues() const { return issues_; }
  std::vector<RuntimeIssue> ExtractIssues() { return std::move(issues_); }
 private:
  RuntimeIssue::Severity severity_limit_;
  std::vector<RuntimeIssue> issues_;
};
}  
#endif  