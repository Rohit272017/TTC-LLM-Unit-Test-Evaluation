#include "tensorflow/core/lib/db/sqlite.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
extern "C" int sqlite3_snapfn_init(sqlite3*, const char**, const void*);
namespace tensorflow {
namespace {
absl::StatusCode GetTfErrorCode(int code) {
  switch (code & 0xff) {
    case SQLITE_OK:    
    case SQLITE_ROW:   
    case SQLITE_DONE:  
      return absl::StatusCode::kOk;
    case SQLITE_ABORT:  
      return absl::StatusCode::kAborted;
    case SQLITE_READONLY:  
    case SQLITE_MISMATCH:  
      return absl::StatusCode::kFailedPrecondition;
    case SQLITE_MISUSE:    
    case SQLITE_INTERNAL:  
      return absl::StatusCode::kInternal;
    case SQLITE_RANGE:  
      return absl::StatusCode::kOutOfRange;
    case SQLITE_CANTOPEN:    
    case SQLITE_CONSTRAINT:  
    case SQLITE_NOTFOUND:    
    case SQLITE_NOTADB:      
      return absl::StatusCode::kInvalidArgument;
    case SQLITE_CORRUPT:  
      return absl::StatusCode::kDataLoss;
    case SQLITE_AUTH:  
    case SQLITE_PERM:  
      return absl::StatusCode::kPermissionDenied;
    case SQLITE_FULL:    
    case SQLITE_TOOBIG:  
    case SQLITE_NOLFS:   
      return absl::StatusCode::kResourceExhausted;
    case SQLITE_BUSY:      
    case SQLITE_LOCKED:    
    case SQLITE_PROTOCOL:  
    case SQLITE_NOMEM:     
      return absl::StatusCode::kUnavailable;
    case SQLITE_INTERRUPT:  
      return absl::StatusCode::kCancelled;
    case SQLITE_ERROR:   
    case SQLITE_IOERR:   
    case SQLITE_SCHEMA:  
    default:
      return absl::StatusCode::kUnknown;
  }
}
template <typename... Args>
Status PrintfStatus(int rc, const char* fmt, Args&&... args) {
  return {GetTfErrorCode(rc),
          strings::Printf(fmt, std::forward<Args>(args)...)};
}
sqlite3_stmt* PrepareRawOrDie(sqlite3* db, const char* sql) {
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  CHECK_EQ(SQLITE_OK, rc) << sql;
  return stmt;
}
Status SetPragma(Sqlite* db, const char* pragma, const StringPiece& value) {
  if (value.empty()) return absl::OkStatus();
  for (auto p = value.begin(); p < value.end(); ++p) {
    if (!(('0' <= *p && *p <= '9') || ('A' <= *p && *p <= 'Z') ||
          ('a' <= *p && *p <= 'z') || *p == '-')) {
      return errors::InvalidArgument("Illegal pragma character");
    }
  }
  SqliteStatement stmt;
  TF_RETURN_IF_ERROR(  
      db->Prepare(strings::StrCat("PRAGMA ", pragma, "=", value), &stmt));
  bool unused_done;
  return stmt.Step(&unused_done);
}
const StringPiece GetEnv(const char* var) {
  const char* val = std::getenv(var);
  return (val == nullptr) ? StringPiece() : StringPiece(val);
}
Status EnvPragma(Sqlite* db, const char* pragma, const char* var) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(SetPragma(db, pragma, GetEnv(var)), "getenv(",
                                  var, ")");
  return absl::OkStatus();
}
}  
Status Sqlite::Open(const string& path, int flags, Sqlite** db) {
  flags |= SQLITE_OPEN_PRIVATECACHE;
  flags |= SQLITE_OPEN_URI;
  sqlite3* sqlite = nullptr;
  int rc = sqlite3_open_v2(path.c_str(), &sqlite, flags, nullptr);
  if (rc != SQLITE_OK) {
    *db = nullptr;
    return PrintfStatus(rc, "Sqlite::Open(%s) failed: %s", path.c_str(),
                        sqlite3_errstr(rc));
  }
  CHECK_EQ(SQLITE_OK, sqlite3_extended_result_codes(sqlite, 1));
  CHECK_EQ(SQLITE_OK, sqlite3_snapfn_init(sqlite, nullptr, nullptr));
  sqlite3_stmt* begin = PrepareRawOrDie(sqlite, "BEGIN");
  sqlite3_stmt* commit = PrepareRawOrDie(sqlite, "COMMIT");
  sqlite3_stmt* rollback = PrepareRawOrDie(sqlite, "ROLLBACK");
  *db = new Sqlite(sqlite, begin, commit, rollback);
  Status s = absl::OkStatus();
  s.Update(SetPragma(*db, "page_size", "4096"));
  s.Update(EnvPragma(*db, "secure_delete", "TF_SQLITE_SECURE_DELETE"));
  s.Update(EnvPragma(*db, "page_size", "TF_SQLITE_PAGE_SIZE"));
  s.Update(EnvPragma(*db, "journal_mode", "TF_SQLITE_JOURNAL_MODE"));
  s.Update(EnvPragma(*db, "synchronous", "TF_SQLITE_SYNCHRONOUS"));
  s.Update(EnvPragma(*db, "mmap_size", "TF_SQLITE_MMAP_SIZE"));
  s.Update(EnvPragma(*db, "locking_mode", "TF_SQLITE_LOCKING_MODE"));
  s.Update(EnvPragma(*db, "cache_size", "TF_SQLITE_CACHE_SIZE"));
  s.Update(EnvPragma(*db, "auto_vacuum", "TF_SQLITE_AUTO_VACUUM"));
  DCHECK((*db)->RefCountIsOne());
  if (!s.ok()) {
    (*db)->Unref();
    *db = nullptr;
  }
  return s;
}
Sqlite::~Sqlite() {
  sqlite3_finalize(rollback_);
  sqlite3_finalize(commit_);
  sqlite3_finalize(begin_);
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_));
}
Status Sqlite::Prepare(const StringPiece& sql, SqliteStatement* stmt) {
  SqliteLock lock(*this);
  sqlite3_stmt* ps = nullptr;
  int rc = sqlite3_prepare_v2(db_, sql.data(), static_cast<int>(sql.size()),
                              &ps, nullptr);
  if (rc != SQLITE_OK) {
    *stmt = SqliteStatement();
    return PrintfStatus(rc, "Prepare() failed: [%d] %s: %.*s", rc, errmsg(),
                        sql.size(), sql.data());
  }
  *stmt = SqliteStatement(this, ps);
  return absl::OkStatus();
}
Status SqliteStatement::Step(bool* is_done) {
  DCHECK(stmt_ != nullptr);
  if (TF_PREDICT_FALSE(bind_error_ != SQLITE_OK)) {
    *is_done = true;
    return PrintfStatus(bind_error_, "Bind(%d) failed: %s: %s",
                        bind_error_parameter_, sqlite3_errstr(bind_error_),
                        sql());
  }
  SqliteLock lock(*db_);
  int rc = sqlite3_step(stmt_);
  switch (rc) {
    case SQLITE_ROW:
      *is_done = false;
      return absl::OkStatus();
    case SQLITE_DONE:
      *is_done = true;
      return absl::OkStatus();
    default:
      *is_done = true;
      return PrintfStatus(rc, "Step() failed: [%d] %s: %s", rc, db_->errmsg(),
                          sql());
  }
}
bool SqliteStatement::StepOrDie() {
  bool is_done;
  TF_CHECK_OK(Step(&is_done));
  return !is_done;
}
Status SqliteStatement::StepOnce() {
  bool is_done;
  TF_RETURN_IF_ERROR(Step(&is_done));
  if (TF_PREDICT_FALSE(is_done)) {
    return errors::Internal("No rows returned: ", sql());
  }
  return absl::OkStatus();
}
const SqliteStatement& SqliteStatement::StepOnceOrDie() {
  TF_CHECK_OK(StepOnce());
  return *this;
}
Status SqliteStatement::StepAndReset() {
  bool is_done;
  Status s = Step(&is_done);
  if (TF_PREDICT_FALSE(s.ok() && !is_done)) {
    s = errors::Internal("Unexpected row: ", sql());
  }
  Reset();
  return s;
}
void SqliteStatement::StepAndResetOrDie() { TF_CHECK_OK(StepAndReset()); }
void SqliteStatement::Reset() {
  if (TF_PREDICT_TRUE(stmt_ != nullptr)) {
    sqlite3_reset(stmt_);
    sqlite3_clear_bindings(stmt_);
  }
  bind_error_ = SQLITE_OK;
  size_ = 0;
}
SqliteTransaction::SqliteTransaction(Sqlite& db) : db_(&db) {
  sqlite3_mutex_enter(sqlite3_db_mutex(db_->db_));
  CHECK(!db_->is_in_transaction_);
  db_->is_in_transaction_ = true;
  Begin();
}
SqliteTransaction::~SqliteTransaction() {
  sqlite3_step(db_->rollback_);
  sqlite3_reset(db_->rollback_);
  sqlite3_reset(db_->begin_);
  db_->is_in_transaction_ = false;
  sqlite3_mutex_leave(sqlite3_db_mutex(db_->db_));
}
void SqliteTransaction::Begin() {
  if (sqlite3_step(db_->begin_) != SQLITE_DONE) {
    LOG(FATAL) << "BEGIN failed: " << sqlite3_errmsg(db_->db_);
  }
}
Status SqliteTransaction::Commit() {
  int rc = sqlite3_step(db_->commit_);
  if (rc != SQLITE_DONE) {
    return PrintfStatus(rc, "COMMIT failed: [%d] %s", rc,
                        sqlite3_errmsg(db_->db_));
  }
  sqlite3_reset(db_->commit_);
  sqlite3_reset(db_->begin_);
  Begin();
  return absl::OkStatus();
}
}  