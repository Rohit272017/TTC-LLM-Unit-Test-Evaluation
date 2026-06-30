#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class FakeInlineRewrite : public InlineRewrite {
 public:
  FakeInlineRewrite(std::string expected_input,
                    RewriteStatus status,
                    std::string replacement)
      : expected_input_(std::move(expected_input)),
        status_(status),
        replacement_(std::move(replacement)) {}

  RewriteStatus Rewrite(absl::string_view input,
                        std::string* output) override {
    last_input_ = std::string(input);
    ++call_count_;

    if (input != expected_input_) {
      return RewriteStatus::NOT_RECOGNIZED;
    }

    if (status_ == RewriteStatus::SUCCESS) {
      output->append(replacement_);
    }

    if (status_ == RewriteStatus::ERROR) {
      output->append(replacement_);
    }

    return status_;
  }

  int call_count() const { return call_count_; }
  const std::string& last_input() const { return last_input_; }

 private:
  std::string expected_input_;
  RewriteStatus status_;
  std::string replacement_;
  int call_count_ = 0;
  std::string last_input_;
};

class AlwaysNotRecognizedRewrite : public InlineRewrite {
 public:
  RewriteStatus Rewrite(absl::string_view input,
                        std::string* output) override {
    last_input = std::string(input);
    ++call_count;
    return RewriteStatus::NOT_RECOGNIZED;
  }

  int call_count = 0;
  std::string last_input;
};

TEST(TextPreprocessorTest, ECP_InputWithoutInlineBlockIsCopiedUnchanged) {
  TextPreprocessor preprocessor('$', false);
  std::string output;

  absl::Status status = preprocessor.Rewrite("plain text", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "plain text");
}

TEST(TextPreprocessorTest, BVA_EmptyInputProducesEmptyOutput) {
  TextPreprocessor preprocessor('$', false);
  std::string output = "unchanged";

  absl::Status status = preprocessor.Rewrite("", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "");
}

TEST(TextPreprocessorTest, BVA_SingleCharacterInputWithoutDelimiter) {
  TextPreprocessor preprocessor('$', false);
  std::string output;

  absl::Status status = preprocessor.Rewrite("a", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "a");
}

TEST(TextPreprocessorTest, ECP_SingleKnownInlineBlockIsRewritten) {
  TextPreprocessor preprocessor('$', false);
  auto rewrite = std::make_unique<FakeInlineRewrite>(
      "VALUE", RewriteStatus::SUCCESS, "42");
  FakeInlineRewrite* rewrite_ptr = rewrite.get();
  preprocessor.AddRewrite(std::move(rewrite));

  std::string output;
  absl::Status status = preprocessor.Rewrite("x = $VALUE$;", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "x = 42;");
  EXPECT_EQ(rewrite_ptr->call_count(), 1);
  EXPECT_EQ(rewrite_ptr->last_input(), "VALUE");
}

TEST(TextPreprocessorTest, ECP_MultipleKnownInlineBlocksAreRewritten) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "one"));
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "B", RewriteStatus::SUCCESS, "two"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$ + $B$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "one + two");
}

TEST(TextPreprocessorTest, BVA_EmptyInlineBlockIsPassedToRewrite) {
  TextPreprocessor preprocessor('$', false);
  auto rewrite = std::make_unique<FakeInlineRewrite>(
      "", RewriteStatus::SUCCESS, "EMPTY");
  FakeInlineRewrite* rewrite_ptr = rewrite.get();
  preprocessor.AddRewrite(std::move(rewrite));

  std::string output;
  absl::Status status = preprocessor.Rewrite("before $$ after", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "before EMPTY after");
  EXPECT_EQ(rewrite_ptr->call_count(), 1);
  EXPECT_EQ(rewrite_ptr->last_input(), "");
}

TEST(TextPreprocessorTest, BVA_InlineBlockAtBeginning) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$ suffix", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "X suffix");
}

TEST(TextPreprocessorTest, BVA_InlineBlockAtEnd) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("prefix $A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "prefix X");
}

TEST(TextPreprocessorTest, BVA_OnlyInlineBlock) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "X");
}

TEST(TextPreprocessorTest, Invalid_UnclosedInlineBlockReturnsNotFound) {
  TextPreprocessor preprocessor('$', false);
  std::string output = "old";

  absl::Status status = preprocessor.Rewrite("prefix $A", &output);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(output, "old");
}

TEST(TextPreprocessorTest, Invalid_SingleDelimiterOnlyReturnsNotFound) {
  TextPreprocessor preprocessor('$', false);
  std::string output = "old";

  absl::Status status = preprocessor.Rewrite("$", &output);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(output, "old");
}

TEST(TextPreprocessorTest, Invalid_UnknownRewriteReturnsNotFoundWhenNotKept) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<AlwaysNotRecognizedRewrite>());

  std::string output = "old";
  absl::Status status = preprocessor.Rewrite("prefix $UNKNOWN$ suffix", &output);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(output, "old");
}

TEST(TextPreprocessorTest, ECP_UnknownRewriteIsPreservedWhenKeepUnknownEnabled) {
  TextPreprocessor preprocessor('$', true);
  auto rewrite = std::make_unique<AlwaysNotRecognizedRewrite>();
  AlwaysNotRecognizedRewrite* rewrite_ptr = rewrite.get();
  preprocessor.AddRewrite(std::move(rewrite));

  std::string output;
  absl::Status status = preprocessor.Rewrite("prefix $UNKNOWN$ suffix", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "prefix $UNKNOWN$ suffix");
  EXPECT_EQ(rewrite_ptr->call_count, 1);
  EXPECT_EQ(rewrite_ptr->last_input, "UNKNOWN");
}

TEST(TextPreprocessorTest, ECP_ErrorRewriteReturnsInternalError) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "BAD", RewriteStatus::ERROR, "partial"));

  std::string output = "old";
  absl::Status status = preprocessor.Rewrite("before $BAD$ after", &output);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(output, "old");
}

TEST(TextPreprocessorTest, Edge_FirstRecognizedRewriteStopsFurtherProcessing) {
  TextPreprocessor preprocessor('$', false);

  auto first = std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "first");
  FakeInlineRewrite* first_ptr = first.get();

  auto second = std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "second");
  FakeInlineRewrite* second_ptr = second.get();

  preprocessor.AddRewrite(std::move(first));
  preprocessor.AddRewrite(std::move(second));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "first");
  EXPECT_EQ(first_ptr->call_count(), 1);
  EXPECT_EQ(second_ptr->call_count(), 0);
}

TEST(TextPreprocessorTest, Edge_NotRecognizedRewriteFallsThroughToNextRewrite) {
  TextPreprocessor preprocessor('$', false);

  auto first = std::make_unique<AlwaysNotRecognizedRewrite>();
  AlwaysNotRecognizedRewrite* first_ptr = first.get();

  auto second = std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "second");
  FakeInlineRewrite* second_ptr = second.get();

  preprocessor.AddRewrite(std::move(first));
  preprocessor.AddRewrite(std::move(second));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "second");
  EXPECT_EQ(first_ptr->call_count, 1);
  EXPECT_EQ(second_ptr->call_count(), 1);
}

TEST(TextPreprocessorTest, Edge_AdjacentInlineBlocksAreBothProcessed) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "1"));
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "B", RewriteStatus::SUCCESS, "2"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$$B$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "12");
}

TEST(TextPreprocessorTest, Edge_InlineBlockContentMayContainWhitespace) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      " A B ", RewriteStatus::SUCCESS, "ok"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("x$ A B $y", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "xoky");
}

TEST(TextPreprocessorTest, Edge_InlineBlockContentMayContainOtherDelimiterCharacters) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A#B", RewriteStatus::SUCCESS, "ok"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A#B$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "ok");
}

TEST(TextPreprocessorTest, Edge_DifferentInlineDelimiterIsSupported) {
  TextPreprocessor preprocessor('@', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("before @A@ after $A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "before X after $A$");
}

TEST(TextPreprocessorTest, Edge_OnlyConfiguredDelimiterIsRecognized) {
  TextPreprocessor preprocessor('#', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output;
  absl::Status status = preprocessor.Rewrite("$A$ #A#", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "$A$ X");
}

TEST(TextPreprocessorTest, Edge_RewriteCanAppendLongReplacement) {
  std::string replacement(4096, 'x');
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "LONG", RewriteStatus::SUCCESS, replacement));

  std::string output;
  absl::Status status = preprocessor.Rewrite("a$LONG$b", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "a" + replacement + "b");
}

TEST(TextPreprocessorTest, Edge_LongInputWithoutInlineBlockIsCopied) {
  std::string input(4096, 'a');
  TextPreprocessor preprocessor('$', false);
  std::string output;

  absl::Status status = preprocessor.Rewrite(input, &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, input);
}

TEST(TextPreprocessorTest, Edge_OutputIsOverwrittenOnSuccess) {
  TextPreprocessor preprocessor('$', false);
  preprocessor.AddRewrite(std::make_unique<FakeInlineRewrite>(
      "A", RewriteStatus::SUCCESS, "X"));

  std::string output = "old content";
  absl::Status status = preprocessor.Rewrite("$A$", &output);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output, "X");
}

TEST(TextPreprocessorTest, Edge_OutputIsNotOverwrittenOnFailure) {
  TextPreprocessor preprocessor('$', false);

  std::string output = "old content";
  absl::Status status = preprocessor.Rewrite("$UNKNOWN$", &output);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(output, "old content");
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite