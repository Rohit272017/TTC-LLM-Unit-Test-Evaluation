#include "phonenumbers/phonenumbermatcher.h"
#ifndef I18N_PHONENUMBERS_USE_ICU_REGEXP
#error phonenumbermatcher depends on ICU \
    (i.e. I18N_PHONENUMBERS_USE_ICU_REGEXP must be set)
#endif  
#include <ctype.h>
#include <stddef.h>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unicode/uchar.h>
#include "phonenumbers/alternate_format.h"
#include "phonenumbers/base/logging.h"
#include "phonenumbers/base/memory/scoped_ptr.h"
#include "phonenumbers/base/memory/singleton.h"
#include "phonenumbers/callback.h"
#include "phonenumbers/default_logger.h"
#include "phonenumbers/encoding_utils.h"
#include "phonenumbers/normalize_utf8.h"
#include "phonenumbers/phonemetadata.pb.h"
#include "phonenumbers/phonenumber.pb.h"
#include "phonenumbers/phonenumbermatch.h"
#include "phonenumbers/phonenumberutil.h"
#include "phonenumbers/regexp_adapter.h"
#include "phonenumbers/regexp_adapter_icu.h"
#include "phonenumbers/regexp_cache.h"
#include "phonenumbers/stringutil.h"
#include "phonenumbers/utf/unicodetext.h"
#ifdef I18N_PHONENUMBERS_USE_RE2
#include "phonenumbers/regexp_adapter_re2.h"
#endif  
using std::map;
using std::numeric_limits;
using std::string;
namespace i18n {
namespace phonenumbers {
namespace {
string Limit(int lower, int upper) {
  DCHECK_GE(lower, 0);
  DCHECK_GT(upper, 0);
  DCHECK_LT(lower, upper);
  return StrCat("{", lower, ",", upper, "}");
}
bool IsInvalidPunctuationSymbol(char32 character) {
  return character == '%' || u_charType(character) == U_CURRENCY_SYMBOL;
}
bool ContainsOnlyValidXChars(const PhoneNumber& number, const string& candidate,
                             const PhoneNumberUtil& util) {
  size_t found;
  found = candidate.find_first_of("xX");
  while (found != string::npos && found < candidate.length() - 1) {
    char next_char = candidate[found + 1];
    if (next_char == 'x' || next_char == 'X') {
      ++found;
      if (util.IsNumberMatchWithOneString(
              number, candidate.substr(found, candidate.length() - found))
          != PhoneNumberUtil::NSN_MATCH) {
        return false;
      }
    } else {
      string normalized_extension(candidate.substr(found,
                                                   candidate.length() - found));
      util.NormalizeDigitsOnly(&normalized_extension);
      if (normalized_extension != number.extension()) {
        return false;
      }
    }
    found = candidate.find_first_of("xX", found + 1);
  }
  return true;
}
bool AllNumberGroupsRemainGrouped(
    const PhoneNumberUtil& util,
    const PhoneNumber& number,
    const string& normalized_candidate,
    const std::vector<string>& formatted_number_groups) {
  size_t from_index = 0;
  if (number.country_code_source() != PhoneNumber::FROM_DEFAULT_COUNTRY) {
    string country_code = SimpleItoa(number.country_code());
    from_index = normalized_candidate.find(country_code) + country_code.size();
  }
  for (size_t i = 0; i < formatted_number_groups.size(); ++i) {
    from_index = normalized_candidate.find(formatted_number_groups.at(i),
                                           from_index);
    if (from_index == string::npos) {
      return false;
    }
    from_index += formatted_number_groups.at(i).length();
    if (i == 0 && from_index < normalized_candidate.length()) {
      string region;
      util.GetRegionCodeForCountryCode(number.country_code(), &region);
      string ndd_prefix;
      util.GetNddPrefixForRegion(region, true, &ndd_prefix);
      if (!ndd_prefix.empty() && isdigit(normalized_candidate.at(from_index))) {
        string national_significant_number;
        util.GetNationalSignificantNumber(number, &national_significant_number);
        return HasPrefixString(normalized_candidate.substr(
            from_index - formatted_number_groups.at(i).length()),
            national_significant_number);
        }
      }
    }
    return normalized_candidate.substr(from_index)
        .find(number.extension()) != string::npos;
}
bool LoadAlternateFormats(PhoneMetadataCollection* alternate_formats) {
#if defined(I18N_PHONENUMBERS_USE_ALTERNATE_FORMATS)
  if (!alternate_formats->ParseFromArray(alternate_format_get(),
                                         alternate_format_size())) {
    LOG(ERROR) << "Could not parse binary data.";
    return false;
  }
  return true;
#else
  return false;
#endif
}
}  
class PhoneNumberMatcherRegExps : public Singleton<PhoneNumberMatcherRegExps> {
 private:
  friend class Singleton<PhoneNumberMatcherRegExps>;
  string opening_parens_;
  string closing_parens_;
  string non_parens_;
  string bracket_pair_limit_;
  string leading_maybe_matched_bracket_;
  string bracket_pairs_;
  string lead_limit_;
  string punctuation_limit_;
  int digit_block_limit_;
  string block_limit_;
  string punctuation_;
  string digit_sequence_;
  string lead_class_chars_;
  string lead_class_;
 public:
  scoped_ptr<const AbstractRegExpFactory> regexp_factory_for_pattern_;
  scoped_ptr<const AbstractRegExpFactory> regexp_factory_;
  mutable RegExpCache regexp_cache_;
  scoped_ptr<const RegExp> pub_pages_;
  scoped_ptr<const RegExp> slash_separated_dates_;
  scoped_ptr<const RegExp> time_stamps_;
  scoped_ptr<const RegExp> time_stamps_suffix_;
  scoped_ptr<const RegExp> matching_brackets_;
  scoped_ptr<std::vector<const RegExp*> > inner_matches_;
  scoped_ptr<const RegExp> capture_up_to_second_number_start_pattern_;
  scoped_ptr<const RegExp> capturing_ascii_digits_pattern_;
  scoped_ptr<const RegExp> lead_class_pattern_;
  scoped_ptr<const RegExp> pattern_;
  PhoneNumberMatcherRegExps()
      : opening_parens_("(\\[\xEF\xBC\x88\xEF\xBC\xBB" ),
        closing_parens_(")\\]\xEF\xBC\x89\xEF\xBC\xBD" ),
        non_parens_(StrCat("[^", opening_parens_, closing_parens_, "]")),
        bracket_pair_limit_(Limit(0, 3)),
        leading_maybe_matched_bracket_(StrCat(
            "(?:[", opening_parens_, "])?",
            "(?:", non_parens_, "+[", closing_parens_, "])?")),
        bracket_pairs_(StrCat(
            "(?:[", opening_parens_, "]", non_parens_, "+",
            "[", closing_parens_, "])", bracket_pair_limit_)),
        lead_limit_(Limit(0, 2)),
        punctuation_limit_(Limit(0, 4)),
        digit_block_limit_(PhoneNumberUtil::kMaxLengthForNsn +
                           PhoneNumberUtil::kMaxLengthCountryCode),
        block_limit_(Limit(0, digit_block_limit_)),
        punctuation_(StrCat("[", PhoneNumberUtil::kValidPunctuation, "]",
                            punctuation_limit_)),
        digit_sequence_(StrCat("\\p{Nd}", Limit(1, digit_block_limit_))),
        lead_class_chars_(StrCat(opening_parens_, PhoneNumberUtil::kPlusChars)),
        lead_class_(StrCat("[", lead_class_chars_, "]")),
        regexp_factory_for_pattern_(new ICURegExpFactory()),
#ifdef I18N_PHONENUMBERS_USE_RE2
        regexp_factory_(new RE2RegExpFactory()),
#else
        regexp_factory_(new ICURegExpFactory()),
#endif  
        regexp_cache_(*regexp_factory_, 32),
        pub_pages_(regexp_factory_->CreateRegExp(
            "\\d{1,5}-+\\d{1,5}\\s{0,4}\\(\\d{1,4}")),
        slash_separated_dates_(regexp_factory_->CreateRegExp(
            "(?:(?:[0-3]?\\d/[01]?\\d)|"
            "(?:[01]?\\d/[0-3]?\\d))/(?:[12]\\d)?\\d{2}")),
        time_stamps_(regexp_factory_->CreateRegExp(
            "[12]\\d{3}[-/]?[01]\\d[-/]?[0-3]\\d +[0-2]\\d$")),
        time_stamps_suffix_(regexp_factory_->CreateRegExp(":[0-5]\\d")),
        matching_brackets_(regexp_factory_->CreateRegExp(
            StrCat(leading_maybe_matched_bracket_, non_parens_, "+",
                   bracket_pairs_, non_parens_, "*"))),
        inner_matches_(new std::vector<const RegExp*>()),
        capture_up_to_second_number_start_pattern_(
            regexp_factory_->CreateRegExp(
                PhoneNumberUtil::kCaptureUpToSecondNumberStart)),
        capturing_ascii_digits_pattern_(
            regexp_factory_->CreateRegExp("(\\d+)")),
        lead_class_pattern_(regexp_factory_->CreateRegExp(lead_class_)),
        pattern_(regexp_factory_for_pattern_->CreateRegExp(StrCat(
            "((?:", lead_class_, punctuation_, ")", lead_limit_,
            digit_sequence_, "(?:", punctuation_, digit_sequence_, ")",
            block_limit_, "(?i)(?:",
            PhoneNumberUtil::GetInstance()->GetExtnPatternsForMatching(),
            ")?)"))) {
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp("/+(.*)"));
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp("(\\([^(]*)"));
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp("(?:\\p{Z}-|-\\p{Z})\\p{Z}*(.+)"));
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp(
            "[\xE2\x80\x92-\xE2\x80\x95\xEF\xBC\x8D]" 
            "\\p{Z}*(.+)"));
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp("\\.+\\p{Z}*([^.]+)"));
    inner_matches_->push_back(
        regexp_factory_->CreateRegExp("\\p{Z}+(\\P{Z}+)"));
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(PhoneNumberMatcherRegExps);
};
class AlternateFormats : public Singleton<AlternateFormats> {
 public:
  PhoneMetadataCollection format_data_;
  map<int, const PhoneMetadata*> calling_code_to_alternate_formats_map_;
  AlternateFormats()
      : format_data_(),
        calling_code_to_alternate_formats_map_() {
    if (!LoadAlternateFormats(&format_data_)) {
      LOG(DFATAL) << "Could not parse compiled-in metadata.";
      return;
    }
    for (RepeatedPtrField<PhoneMetadata>::const_iterator it =
             format_data_.metadata().begin();
         it != format_data_.metadata().end();
         ++it) {
      calling_code_to_alternate_formats_map_.insert(
          std::make_pair(it->country_code(), &*it));
    }
  }
  const PhoneMetadata* GetAlternateFormatsForCountry(int country_calling_code)
      const {
    map<int, const PhoneMetadata*>::const_iterator it =
        calling_code_to_alternate_formats_map_.find(country_calling_code);
    if (it != calling_code_to_alternate_formats_map_.end()) {
      return it->second;
    }
    return NULL;
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(AlternateFormats);
};
PhoneNumberMatcher::PhoneNumberMatcher(const PhoneNumberUtil& util,
                                       const string& text,
                                       const string& region_code,
                                       PhoneNumberMatcher::Leniency leniency,
                                       int max_tries)
    : reg_exps_(PhoneNumberMatcherRegExps::GetInstance()),
      alternate_formats_(AlternateFormats::GetInstance()),
      phone_util_(util),
      text_(text),
      preferred_region_(region_code),
      leniency_(leniency),
      max_tries_(max_tries),
      state_(NOT_READY),
      last_match_(NULL),
      search_index_(0),
      is_input_valid_utf8_(true) {
  is_input_valid_utf8_ = IsInputUtf8(); 
}
PhoneNumberMatcher::PhoneNumberMatcher(const string& text,
                                       const string& region_code)
    : reg_exps_(PhoneNumberMatcherRegExps::GetInstance()),
      alternate_formats_(NULL),  
      phone_util_(*PhoneNumberUtil::GetInstance()),
      text_(text),
      preferred_region_(region_code),
      leniency_(VALID),
      max_tries_(numeric_limits<int>::max()),
      state_(NOT_READY),
      last_match_(NULL),
      search_index_(0),
      is_input_valid_utf8_(true) {
  is_input_valid_utf8_ =  IsInputUtf8();
}
PhoneNumberMatcher::~PhoneNumberMatcher() {
}
bool PhoneNumberMatcher::IsInputUtf8() {
  UnicodeText number_as_unicode;
  number_as_unicode.PointToUTF8(text_.c_str(), text_.size());
  return number_as_unicode.UTF8WasValid();
}
bool PhoneNumberMatcher::IsLatinLetter(char32 letter) {
  if (!u_isalpha(letter) && (u_charType(letter) != U_NON_SPACING_MARK)) {
    return false;
  }
  UBlockCode block = ublock_getCode(letter);
  return ((block == UBLOCK_BASIC_LATIN) ||
      (block == UBLOCK_LATIN_1_SUPPLEMENT) ||
      (block == UBLOCK_LATIN_EXTENDED_A) ||
      (block == UBLOCK_LATIN_EXTENDED_ADDITIONAL) ||
      (block == UBLOCK_LATIN_EXTENDED_B) ||
      (block == UBLOCK_COMBINING_DIACRITICAL_MARKS));
}
bool PhoneNumberMatcher::ParseAndVerify(const string& candidate, int offset,
                                        PhoneNumberMatch* match) {
  DCHECK(match);
  if (!reg_exps_->matching_brackets_->FullMatch(candidate) ||
      reg_exps_->pub_pages_->PartialMatch(candidate)) {
    return false;
  }
  if (leniency_ >= VALID) {
    scoped_ptr<RegExpInput> candidate_input(
        reg_exps_->regexp_factory_->CreateInput(candidate));
    if (offset > 0 &&
        !reg_exps_->lead_class_pattern_->Consume(candidate_input.get())) {
      char32 previous_char;
      const char* previous_char_ptr =
          EncodingUtils::BackUpOneUTF8Character(text_.c_str(),
                                                text_.c_str() + offset);
      EncodingUtils::DecodeUTF8Char(previous_char_ptr, &previous_char);
      if (IsInvalidPunctuationSymbol(previous_char) ||
          IsLatinLetter(previous_char)) {
        return false;
      }
    }
    size_t lastCharIndex = offset + candidate.length();
    if (lastCharIndex < text_.length()) {
      char32 next_char;
      const char* next_char_ptr =
          EncodingUtils::AdvanceOneUTF8Character(
              text_.c_str() + lastCharIndex - 1);
      EncodingUtils::DecodeUTF8Char(next_char_ptr, &next_char);
      if (IsInvalidPunctuationSymbol(next_char) || IsLatinLetter(next_char)) {
        return false;
      }
    }
  }
  PhoneNumber number;
  if (phone_util_.ParseAndKeepRawInput(candidate, preferred_region_, &number) !=
      PhoneNumberUtil::NO_PARSING_ERROR) {
    return false;
  }
  if (VerifyAccordingToLeniency(leniency_, number, candidate)) {
    match->set_start(offset);
    match->set_raw_string(candidate);
    number.clear_country_code_source();
    number.clear_preferred_domestic_carrier_code();
    number.clear_raw_input();
    match->set_number(number);
    return true;
  }
  return false;
}
bool PhoneNumberMatcher::VerifyAccordingToLeniency(
    Leniency leniency, const PhoneNumber& number,
    const string& candidate) const {
  switch (leniency) {
    case PhoneNumberMatcher::POSSIBLE:
      return phone_util_.IsPossibleNumber(number);
    case PhoneNumberMatcher::VALID:
      if (!phone_util_.IsValidNumber(number) ||
          !ContainsOnlyValidXChars(number, candidate, phone_util_)) {
        return false;
      }
      return IsNationalPrefixPresentIfRequired(number);
    case PhoneNumberMatcher::STRICT_GROUPING: {
      if (!phone_util_.IsValidNumber(number) ||
          !ContainsOnlyValidXChars(number, candidate, phone_util_) ||
          ContainsMoreThanOneSlashInNationalNumber(
              number, candidate, phone_util_) ||
          !IsNationalPrefixPresentIfRequired(number)) {
        return false;
      }
      ResultCallback4<bool, const PhoneNumberUtil&, const PhoneNumber&,
                      const string&, const std::vector<string>&>* callback =
          NewPermanentCallback(&AllNumberGroupsRemainGrouped);
      bool is_valid = CheckNumberGroupingIsValid(number, candidate, callback);
      delete(callback);
      return is_valid;
    }
    case PhoneNumberMatcher::EXACT_GROUPING: {
      if (!phone_util_.IsValidNumber(number) ||
          !ContainsOnlyValidXChars(number, candidate, phone_util_) ||
          ContainsMoreThanOneSlashInNationalNumber(
              number, candidate, phone_util_) ||
          !IsNationalPrefixPresentIfRequired(number)) {
        return false;
      }
      ResultCallback4<bool, const PhoneNumberUtil&, const PhoneNumber&,
                      const string&, const std::vector<string>&>* callback =
          NewPermanentCallback(
              this, &PhoneNumberMatcher::AllNumberGroupsAreExactlyPresent);
      bool is_valid = CheckNumberGroupingIsValid(number, candidate, callback);
      delete(callback);
      return is_valid;
    }
    default:
      LOG(ERROR) << "No implementation defined for verification for leniency "
                 << static_cast<int>(leniency);
      return false;
  }
}
bool PhoneNumberMatcher::ExtractInnerMatch(const string& candidate, int offset,
                                           PhoneNumberMatch* match) {
  DCHECK(match);
  for (std::vector<const RegExp*>::const_iterator regex =
           reg_exps_->inner_matches_->begin();
           regex != reg_exps_->inner_matches_->end(); regex++) {
    scoped_ptr<RegExpInput> candidate_input(
        reg_exps_->regexp_factory_->CreateInput(candidate));
    bool is_first_match = true;
    string group;
    while ((*regex)->FindAndConsume(candidate_input.get(), &group) &&
           max_tries_ > 0) {
      int group_start_index = static_cast<int>(candidate.length() -
          candidate_input->ToString().length() - group.length());
      if (is_first_match) {
        string first_group_only = candidate.substr(0, group_start_index);
        phone_util_.TrimUnwantedEndChars(&first_group_only);
        bool success = ParseAndVerify(first_group_only, offset, match);
        if (success) {
          return true;
        }
        --max_tries_;
        is_first_match = false;
      }
      phone_util_.TrimUnwantedEndChars(&group);
      bool success = ParseAndVerify(group, offset + group_start_index, match);
      if (success) {
        return true;
      }
      --max_tries_;
    }
  }
  return false;
}
bool PhoneNumberMatcher::ExtractMatch(const string& candidate, int offset,
                                      PhoneNumberMatch* match) {
  DCHECK(match);
  if (reg_exps_->slash_separated_dates_->PartialMatch(candidate)) {
    return false;
  }
  if (reg_exps_->time_stamps_->PartialMatch(candidate)) {
    scoped_ptr<RegExpInput> following_text(
        reg_exps_->regexp_factory_->CreateInput(
            text_.substr(offset + candidate.size())));
    if (reg_exps_->time_stamps_suffix_->Consume(following_text.get())) {
      return false;
    }
  }
  if (ParseAndVerify(candidate, offset, match)) {
    return true;
  }
  return ExtractInnerMatch(candidate, offset, match);
}
bool PhoneNumberMatcher::HasNext() {
  if (!is_input_valid_utf8_) {
    state_ = DONE;
    return false;
  }
  if (state_ == NOT_READY) {
    PhoneNumberMatch temp_match;
    if (!Find(search_index_, &temp_match)) {
      state_ = DONE;
    } else {
      last_match_.reset(new PhoneNumberMatch(temp_match.start(),
                                             temp_match.raw_string(),
                                             temp_match.number()));
      search_index_ = last_match_->end();
      state_ = READY;
    }
  }
  return state_ == READY;
}
bool PhoneNumberMatcher::Next(PhoneNumberMatch* match) {
  DCHECK(match);
  if (!HasNext()) {
    return false;
  }
  match->CopyFrom(*last_match_);
  state_ = NOT_READY;
  last_match_.reset(NULL);
  return true;
}
bool PhoneNumberMatcher::Find(int index, PhoneNumberMatch* match) {
  DCHECK(match);
  scoped_ptr<RegExpInput> text(
      reg_exps_->regexp_factory_for_pattern_->CreateInput(text_.substr(index)));
  string candidate;
  while ((max_tries_ > 0) &&
         reg_exps_->pattern_->FindAndConsume(text.get(), &candidate)) {
    int start = static_cast<int>(text_.length() - text->ToString().length() - candidate.length());
    reg_exps_->capture_up_to_second_number_start_pattern_->
        PartialMatch(candidate, &candidate);
    if (ExtractMatch(candidate, start, match)) {
      return true;
    }
    index = static_cast<int>(start + candidate.length());
    --max_tries_;
  }
  return false;
}
bool PhoneNumberMatcher::CheckNumberGroupingIsValid(
    const PhoneNumber& phone_number,
    const string& candidate,
    ResultCallback4<bool, const PhoneNumberUtil&, const PhoneNumber&,
                    const string&, const std::vector<string>&>* checker) const {
  DCHECK(checker);
  string normalized_candidate =
      NormalizeUTF8::NormalizeDecimalDigits(candidate);
  std::vector<string> formatted_number_groups;
  GetNationalNumberGroups(phone_number, &formatted_number_groups);
  if (checker->Run(phone_util_, phone_number, normalized_candidate,
                   formatted_number_groups)) {
    return true;
  }
  const PhoneMetadata* alternate_formats =
    alternate_formats_->GetAlternateFormatsForCountry(
        phone_number.country_code());
  if (alternate_formats) {
    string national_significant_number;
    phone_util_.GetNationalSignificantNumber(phone_number,
                                             &national_significant_number);
    for (RepeatedPtrField<NumberFormat>::const_iterator it =
             alternate_formats->number_format().begin();
         it != alternate_formats->number_format().end(); ++it) {
      if (it->leading_digits_pattern_size() > 0) {
        std::unique_ptr<RegExpInput> nsn_input(
            reg_exps_->regexp_factory_->CreateInput(
                national_significant_number));
        if (!reg_exps_->regexp_cache_.GetRegExp(
                it->leading_digits_pattern(0)).Consume(nsn_input.get())) {
          continue;
        }
      }
      formatted_number_groups.clear();
      GetNationalNumberGroupsForPattern(phone_number, &*it,
                                        &formatted_number_groups);
      if (checker->Run(phone_util_, phone_number, normalized_candidate,
                       formatted_number_groups)) {
        return true;
      }
    }
  }
  return false;
}
void PhoneNumberMatcher::GetNationalNumberGroups(
    const PhoneNumber& number,
    std::vector<string>* digit_blocks) const {
  string rfc3966_format;
  phone_util_.Format(number, PhoneNumberUtil::RFC3966, &rfc3966_format);
  size_t end_index = rfc3966_format.find(';');
  if (end_index == string::npos) {
    end_index = rfc3966_format.length();
  }
  size_t start_index = rfc3966_format.find('-') + 1;
  SplitStringUsing(rfc3966_format.substr(start_index,
                                         end_index - start_index),
                   '-', digit_blocks);
}
void PhoneNumberMatcher::GetNationalNumberGroupsForPattern(
    const PhoneNumber& number,
    const NumberFormat* formatting_pattern,
    std::vector<string>* digit_blocks) const {
  string rfc3966_format;
  string national_significant_number;
  phone_util_.GetNationalSignificantNumber(number,
                                           &national_significant_number);
  phone_util_.FormatNsnUsingPattern(national_significant_number,
                                    *formatting_pattern,
                                    PhoneNumberUtil::RFC3966,
                                    &rfc3966_format);
  SplitStringUsing(rfc3966_format, '-', digit_blocks);
}
bool PhoneNumberMatcher::IsNationalPrefixPresentIfRequired(
    const PhoneNumber& number) const {
  if (number.country_code_source() != PhoneNumber::FROM_DEFAULT_COUNTRY) {
    return true;
  }
  string phone_number_region;
  phone_util_.GetRegionCodeForCountryCode(
      number.country_code(), &phone_number_region);
  const PhoneMetadata* metadata =
      phone_util_.GetMetadataForRegion(phone_number_region);
  if (!metadata) {
    return true;
  }
  string national_number;
  phone_util_.GetNationalSignificantNumber(number, &national_number);
  const NumberFormat* format_rule =
      phone_util_.ChooseFormattingPatternForNumber(metadata->number_format(),
                                                   national_number);
  if (format_rule && !format_rule->national_prefix_formatting_rule().empty()) {
    if (format_rule->national_prefix_optional_when_formatting()) {
      return true;
    }
    if (phone_util_.FormattingRuleHasFirstGroupOnly(
        format_rule->national_prefix_formatting_rule())) {
      return true;
    }
    string raw_input_copy(number.raw_input());
    phone_util_.NormalizeDigitsOnly(&raw_input_copy);
    return phone_util_.MaybeStripNationalPrefixAndCarrierCode(
        *metadata,
        &raw_input_copy,
        NULL);  
  }
  return true;
}
bool PhoneNumberMatcher::AllNumberGroupsAreExactlyPresent(
    const PhoneNumberUtil& util,
    const PhoneNumber& phone_number,
    const string& normalized_candidate,
    const std::vector<string>& formatted_number_groups) const {
  const scoped_ptr<RegExpInput> candidate_number(
      reg_exps_->regexp_factory_->CreateInput(normalized_candidate));
  std::vector<string> candidate_groups;
  string digit_block;
  while (reg_exps_->capturing_ascii_digits_pattern_->FindAndConsume(
             candidate_number.get(),
             &digit_block)) {
    candidate_groups.push_back(digit_block);
  }
  int candidate_number_group_index = static_cast<int>(
      phone_number.has_extension() ? candidate_groups.size() - 2
                                   : candidate_groups.size() - 1);
  string national_significant_number;
  util.GetNationalSignificantNumber(phone_number,
                                    &national_significant_number);
  if (candidate_groups.size() == 1 ||
      candidate_groups.at(candidate_number_group_index).find(
          national_significant_number) != string::npos) {
    return true;
  }
  for (int formatted_number_group_index =
           static_cast<int>(formatted_number_groups.size() - 1);
       formatted_number_group_index > 0 &&
       candidate_number_group_index >= 0;
       --formatted_number_group_index, --candidate_number_group_index) {
    if (candidate_groups.at(candidate_number_group_index) !=
        formatted_number_groups.at(formatted_number_group_index)) {
      return false;
    }
  }
  return (candidate_number_group_index >= 0 &&
          HasSuffixString(candidate_groups.at(candidate_number_group_index),
                          formatted_number_groups.at(0)));
}
bool PhoneNumberMatcher::ContainsMoreThanOneSlashInNationalNumber(
    const PhoneNumber& number,
    const string& candidate,
    const PhoneNumberUtil& util) {
  size_t first_slash_in_body = candidate.find('/');
  if (first_slash_in_body == string::npos) {
    return false;
  }
  size_t second_slash_in_body = candidate.find('/', first_slash_in_body + 1);
  if (second_slash_in_body == string::npos) {
    return false;
  }
  if (number.country_code_source() == PhoneNumber::FROM_NUMBER_WITH_PLUS_SIGN ||
      number.country_code_source() ==
          PhoneNumber::FROM_NUMBER_WITHOUT_PLUS_SIGN) {
    string normalized_country_code =
        candidate.substr(0, first_slash_in_body);
    util.NormalizeDigitsOnly(&normalized_country_code);
    if (normalized_country_code == SimpleItoa(number.country_code())) {
      return candidate.find('/', second_slash_in_body + 1) != string::npos;
    }
  }
  return true;
}
}  
}  