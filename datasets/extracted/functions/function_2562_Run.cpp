#include "tensorflow/core/summary/schema.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace {
Status Run(Sqlite* db, const char* sql) {
  SqliteStatement stmt;
  TF_RETURN_IF_ERROR(db->Prepare(sql, &stmt));
  return stmt.StepAndReset();
}
}  
Status SetupTensorboardSqliteDb(Sqlite* db) {
  TF_RETURN_IF_ERROR(
      db->PrepareOrDie(strings::StrCat("PRAGMA application_id=",
                                       kTensorboardSqliteApplicationId))
          .StepAndReset());
  db->PrepareOrDie("PRAGMA user_version=0").StepAndResetOrDie();
  Status s;
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Ids (
      id INTEGER PRIMARY KEY
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Descriptions (
      id INTEGER PRIMARY KEY,
      description TEXT
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Tensors (
      rowid INTEGER PRIMARY KEY,
      series INTEGER,
      step INTEGER,
      dtype INTEGER,
      computed_time REAL,
      shape TEXT,
      data BLOB
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS
      TensorSeriesStepIndex
    ON
      Tensors (series, step)
    WHERE
      series IS NOT NULL
      AND step IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS TensorStrings (
      rowid INTEGER PRIMARY KEY,
      tensor_rowid INTEGER NOT NULL,
      idx INTEGER NOT NULL,
      data BLOB
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS TensorStringIndex
    ON TensorStrings (tensor_rowid, idx)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Tags (
      rowid INTEGER PRIMARY KEY,
      run_id INTEGER,
      tag_id INTEGER NOT NULL,
      inserted_time DOUBLE,
      tag_name TEXT,
      display_name TEXT,
      plugin_name TEXT,
      plugin_data BLOB
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS TagIdIndex
    ON Tags (tag_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS
      TagRunNameIndex
    ON
      Tags (run_id, tag_name)
    WHERE
      run_id IS NOT NULL
      AND tag_name IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Runs (
      rowid INTEGER PRIMARY KEY,
      experiment_id INTEGER,
      run_id INTEGER NOT NULL,
      inserted_time REAL,
      started_time REAL,
      finished_time REAL,
      run_name TEXT
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS RunIdIndex
    ON Runs (run_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS RunNameIndex
    ON Runs (experiment_id, run_name)
    WHERE run_name IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Experiments (
      rowid INTEGER PRIMARY KEY,
      user_id INTEGER,
      experiment_id INTEGER NOT NULL,
      inserted_time REAL,
      started_time REAL,
      is_watching INTEGER,
      experiment_name TEXT
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS ExperimentIdIndex
    ON Experiments (experiment_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS ExperimentNameIndex
    ON Experiments (user_id, experiment_name)
    WHERE experiment_name IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Users (
      rowid INTEGER PRIMARY KEY,
      user_id INTEGER NOT NULL,
      inserted_time REAL,
      user_name TEXT,
      email TEXT
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserIdIndex
    ON Users (user_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserNameIndex
    ON Users (user_name)
    WHERE user_name IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS UserEmailIndex
    ON Users (email)
    WHERE email IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Graphs (
      rowid INTEGER PRIMARY KEY,
      run_id INTEGER,
      graph_id INTEGER NOT NULL,
      inserted_time REAL,
      graph_def BLOB
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS GraphIdIndex
    ON Graphs (graph_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS GraphRunIndex
    ON Graphs (run_id)
    WHERE run_id IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS Nodes (
      rowid INTEGER PRIMARY KEY,
      graph_id INTEGER NOT NULL,
      node_id INTEGER NOT NULL,
      node_name TEXT,
      op TEXT,
      device TEXT,
      node_def BLOB
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeIdIndex
    ON Nodes (graph_id, node_id)
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeNameIndex
    ON Nodes (graph_id, node_name)
    WHERE node_name IS NOT NULL
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE TABLE IF NOT EXISTS NodeInputs (
      rowid INTEGER PRIMARY KEY,
      graph_id INTEGER NOT NULL,
      node_id INTEGER NOT NULL,
      idx INTEGER NOT NULL,
      input_node_id INTEGER NOT NULL,
      input_node_idx INTEGER,
      is_control INTEGER
    )
  )sql"));
  s.Update(Run(db, R"sql(
    CREATE UNIQUE INDEX IF NOT EXISTS NodeInputsIndex
    ON NodeInputs (graph_id, node_id, idx)
  )sql"));
  return s;
}
}  