#include "tensorflow/core/kernels/checkpoint_callback_manager.h"
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tsl/platform/regexp.h"
namespace tensorflow {
namespace checkpoint {
const absl::string_view kCheckpointCallbackManagerResourceName =
    "checkpoint_callback_manager";
namespace {
const absl::string_view kCheckpointFileRegex = "^part-[0-9]*-of-[0-9]*";
const absl::string_view kCheckpointTempDirRegex = "-[0-9]*_temp$";
const absl::string_view kCheckpointDirRegex = "-[0-9]*$";
const absl::string_view kCheckpointTempDirSuffix = "_temp";
void TriggerSaveCallbackIfFileNotExist(absl::string_view checkpoint_id,
                                       absl::string_view checkpoint_dir,
                                       absl::string_view file_extension,
                                       SaveCallback callback) {
  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));
  if (Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  LOG(INFO) << "Calling a save callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  absl::StatusOr<std::string> save_content = callback(checkpoint_id);
  if (!save_content.ok()) {
    LOG(WARNING) << save_content.status();
    return;
  }
  if (save_content->empty()) {
    return;
  }
  Status write_status =
      WriteStringToFile(Env::Default(), file_path, *save_content);
  if (!write_status.ok()) {
    LOG(WARNING) << write_status;
  } else {
    LOG(INFO) << "A CheckpointCallbackManager has been written to "
              << file_path;
  }
}
void TriggerRestoreCallbackIfFileExists(absl::string_view checkpoint_id,
                                        absl::string_view checkpoint_dir,
                                        absl::string_view file_extension,
                                        RestoreCallback callback) {
  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));
  if (!Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  std::string payload;
  Status read_status = ReadFileToString(Env::Default(), file_path, &payload);
  if (!read_status.ok()) {
    LOG(WARNING) << "Failed to read: " << read_status;
    return;
  }
  LOG(INFO) << "Calling a restore callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  Status callback_status = callback(checkpoint_id, payload);
  if (!callback_status.ok()) {
    LOG(WARNING) << callback_status;
  }
}
}  
absl::StatusOr<std::pair<std::string, std::string>>
CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
    absl::string_view prefix) {
  for (absl::string_view path = prefix;; path = io::Dirname(path)) {
    absl::string_view basename = io::Basename(path);
    if (basename.empty()) break;
    if (RE2::PartialMatch(basename, kCheckpointFileRegex)) continue;
    if (RE2::PartialMatch(basename, kCheckpointTempDirRegex)) {
      return std::make_pair(
          std::string(basename.substr(
              0, basename.length() - kCheckpointTempDirSuffix.length())),
          std::string(io::Dirname(path)));
    }
    if (RE2::PartialMatch(basename, kCheckpointDirRegex)) {
      return std::make_pair(std::string(basename),
                            std::string(io::Dirname(path)));
    }
  }
  return errors::NotFound(
      absl::StrCat("Failed to find a checkpoint id. prefix = ", prefix));
}
Status CheckpointCallbackManager::RegisterSaveCallback(
    absl::string_view file_extension, SaveCallback callback) {
  SaveCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!save_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }
    if (!last_saved_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = save_callbacks_[file_extension];
      checkpoint_id = last_saved_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_saved_checkpoint_id_and_dir_.second;
    }
  }
  if (lazy_callback != nullptr) {
    TriggerSaveCallbackIfFileNotExist(checkpoint_id, checkpoint_dir,
                                      file_extension, lazy_callback);
  }
  return absl::OkStatus();
}
bool CheckpointCallbackManager::DoesSaveCallbackExist(
    absl::string_view file_extension) {
  tf_shared_lock l(mu_);
  return save_callbacks_.contains(file_extension);
}
Status CheckpointCallbackManager::RegisterRestoreCallback(
    absl::string_view file_extension, RestoreCallback callback) {
  RestoreCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!restore_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }
    if (!last_restored_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = restore_callbacks_[file_extension];
      checkpoint_id = last_restored_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_restored_checkpoint_id_and_dir_.second;
    }
  }
  if (lazy_callback != nullptr) {
    TriggerRestoreCallbackIfFileExists(checkpoint_id, checkpoint_dir,
                                       file_extension, lazy_callback);
  }
  return absl::OkStatus();
}
bool CheckpointCallbackManager::DoesRestoreCallbackExist(
    absl::string_view file_extension) {
  tf_shared_lock l(mu_);
  return restore_callbacks_.contains(file_extension);
}
void CheckpointCallbackManager::Save(absl::string_view prefix) {
  absl::StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }
  absl::flat_hash_map<std::string, SaveCallback> copy_of_save_callbacks;
  {
    mutex_lock l(mu_);
    last_saved_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_save_callbacks = save_callbacks_;
  }
  for (const auto& name_and_callback : copy_of_save_callbacks) {
    TriggerSaveCallbackIfFileNotExist(id_and_dir->first, id_and_dir->second,
                                      name_and_callback.first,
                                      name_and_callback.second);
  }
}
void CheckpointCallbackManager::Restore(absl::string_view prefix) {
  absl::StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }
  absl::flat_hash_map<std::string, RestoreCallback> copy_of_restore_callbacks;
  {
    mutex_lock l(mu_);
    if (*id_and_dir == last_restored_checkpoint_id_and_dir_) {
      return;
    }
    last_restored_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_restore_callbacks = restore_callbacks_;
  }
  for (const auto& name_and_callback : copy_of_restore_callbacks) {
    TriggerRestoreCallbackIfFileExists(id_and_dir->first, id_and_dir->second,
                                       name_and_callback.first,
                                       name_and_callback.second);
  }
}
}  
}  