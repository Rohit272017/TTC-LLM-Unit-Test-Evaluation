#include "tsl/platform/mutex.h"
#include <time.h>
#include <cstdint>
#include "nsync_cv.h"       
#include "nsync_mu.h"       
#include "nsync_mu_wait.h"  
#include "nsync_time.h"     
namespace tsl {
static_assert(sizeof(nsync::nsync_mu) <= sizeof(internal::MuData),
              "tsl::internal::MuData needs to be bigger");
static inline nsync::nsync_mu *mu_cast(internal::MuData *mu) {
  return reinterpret_cast<nsync::nsync_mu *>(mu);
}
static inline const nsync::nsync_mu *mu_cast(const internal::MuData *mu) {
  return reinterpret_cast<const nsync::nsync_mu *>(mu);
}
mutex::mutex() { nsync::nsync_mu_init(mu_cast(&mu_)); }
void mutex::lock() { nsync::nsync_mu_lock(mu_cast(&mu_)); }
bool mutex::try_lock() { return nsync::nsync_mu_trylock(mu_cast(&mu_)) != 0; };
void mutex::unlock() { nsync::nsync_mu_unlock(mu_cast(&mu_)); }
void mutex::assert_held() const TF_ASSERT_EXCLUSIVE_LOCK() {
  nsync::nsync_mu_assert_held(mu_cast(&mu_));
}
void mutex::lock_shared() { nsync::nsync_mu_rlock(mu_cast(&mu_)); }
bool mutex::try_lock_shared() {
  return nsync::nsync_mu_rtrylock(mu_cast(&mu_)) != 0;
};
void mutex::unlock_shared() { nsync::nsync_mu_runlock(mu_cast(&mu_)); }
void mutex::assert_held_shared() const TF_ASSERT_SHARED_LOCK() {
  nsync::nsync_mu_rassert_held(mu_cast(&mu_));
}
static int EvaluateCondition(const void *vcond) {
  return static_cast<int>(static_cast<const Condition *>(vcond)->Eval());
}
void mutex::Await(const Condition &cond) {
  nsync::nsync_mu_wait(mu_cast(&mu_), &EvaluateCondition, &cond, nullptr);
}
bool mutex::AwaitWithDeadline(const Condition &cond, uint64_t abs_deadline_ns) {
  time_t seconds = abs_deadline_ns / (1000 * 1000 * 1000);
  nsync::nsync_time abs_time = nsync::nsync_time_s_ns(
      seconds, abs_deadline_ns - seconds * (1000 * 1000 * 1000));
  return nsync::nsync_mu_wait_with_deadline(mu_cast(&mu_), &EvaluateCondition,
                                            &cond, nullptr, abs_time,
                                            nullptr) == 0;
}
static_assert(sizeof(nsync::nsync_cv) <= sizeof(internal::CVData),
              "tsl::internal::CVData needs to be bigger");
static inline nsync::nsync_cv *cv_cast(internal::CVData *cv) {
  return reinterpret_cast<nsync::nsync_cv *>(cv);
}
condition_variable::condition_variable() {
  nsync::nsync_cv_init(cv_cast(&cv_));
}
void condition_variable::wait(mutex_lock &lock) {
  nsync::nsync_cv_wait(cv_cast(&cv_), mu_cast(&lock.mutex()->mu_));
}
void condition_variable::notify_one() { nsync::nsync_cv_signal(cv_cast(&cv_)); }
void condition_variable::notify_all() {
  nsync::nsync_cv_broadcast(cv_cast(&cv_));
}
namespace internal {
std::cv_status wait_until_system_clock(
    CVData *cv_data, MuData *mu_data,
    const std::chrono::system_clock::time_point timeout_time) {
  int r = nsync::nsync_cv_wait_with_deadline(cv_cast(cv_data), mu_cast(mu_data),
                                             timeout_time, nullptr);
  return r ? std::cv_status::timeout : std::cv_status::no_timeout;
}
}  
}  