#include "phonenumbers/phonenumberutil.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iterator>
#include <map>
#include <utility>
#include <vector>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include "phonenumbers/asyoutypeformatter.h"
#include "phonenumbers/base/basictypes.h"
#include "phonenumbers/base/logging.h"
#include "phonenumbers/base/memory/singleton.h"
#include "phonenumbers/default_logger.h"
#include "phonenumbers/encoding_utils.h"
#include "phonenumbers/matcher_api.h"
#include "phonenumbers/metadata.h"
#include "phonenumbers/normalize_utf8.h"
#include "phonenumbers/phonemetadata.pb.h"
#include "phonenumbers/phonenumber.h"
#include "phonenumbers/phonenumber.pb.h"
#include "phonenumbers/regex_based_matcher.h"
#include "phonenumbers/regexp_adapter.h"
#include "phonenumbers/regexp_cache.h"
#include "phonenumbers/regexp_factory.h"
#include "phonenumbers/region_code.h"
#include "phonenumbers/stl_util.h"
#include "phonenumbers/stringutil.h"
#include "phonenumbers/utf/unicodetext.h"
#include "phonenumbers/utf/utf.h"
namespace i18n {
namespace phonenumbers {
using google::protobuf::RepeatedField;
using gtl::OrderByFirst;
const size_t PhoneNumberUtil::kMinLengthForNsn;
const size_t PhoneNumberUtil::kMaxLengthForNsn;
const size_t PhoneNumberUtil::kMaxLengthCountryCode;
const int PhoneNumberUtil::kNanpaCountryCode;
const char PhoneNumberUtil::kPlusChars[] = "+\xEF\xBC\x8B";  
const char PhoneNumberUtil::kValidPunctuation[] =
    "-x\xE2\x80\x90-\xE2\x80\x95\xE2\x88\x92\xE3\x83\xBC\xEF\xBC\x8D-\xEF\xBC"
    "\x8F \xC2\xA0\xC2\xAD\xE2\x80\x8B\xE2\x81\xA0\xE3\x80\x80()\xEF\xBC\x88"
    "\xEF\xBC\x89\xEF\xBC\xBB\xEF\xBC\xBD.\\[\\]/~\xE2\x81\x93\xE2\x88\xBC";
const char PhoneNumberUtil::kCaptureUpToSecondNumberStart[] = "(.*)[\\\\/] *x";
const char PhoneNumberUtil::kRegionCodeForNonGeoEntity[] = "001";
namespace {
const char kPlusSign[] = "+";
const char kStarSign[] = "*";
const char kRfc3966ExtnPrefix[] = ";ext=";
const char kRfc3966Prefix[] = "tel:";
const char kRfc3966PhoneContext[] = ";phone-context=";
const char kRfc3966IsdnSubaddress[] = ";isub=";
const char kRfc3966VisualSeparator[] = "[\\-\\.\\(\\)]?";
const char kDigits[] = "\\p{Nd}";
const char kValidAlpha[] = "a-z";
const char kValidAlphaInclUppercase[] = "A-Za-z";
const char kDefaultExtnPrefix[] = " ext. ";
const char kPossibleSeparatorsBetweenNumberAndExtLabel[] =
    "[ \xC2\xA0\\t,]*";
const char kPossibleCharsAfterExtLabel[] =
    "[:\\.\xEF\xBC\x8E]?[ \xC2\xA0\\t,-]*";
const char kOptionalExtSuffix[] = "#?";
bool LoadCompiledInMetadata(PhoneMetadataCollection* metadata) {
  if (!metadata->ParseFromArray(metadata_get(), metadata_size())) {
    LOG(ERROR) << "Could not parse binary data.";
    return false;
  }
  return true;
}
const PhoneNumberDesc* GetNumberDescByType(
    const PhoneMetadata& metadata,
    PhoneNumberUtil::PhoneNumberType type) {
  switch (type) {
    case PhoneNumberUtil::PREMIUM_RATE:
      return &metadata.premium_rate();
    case PhoneNumberUtil::TOLL_FREE:
      return &metadata.toll_free();
    case PhoneNumberUtil::MOBILE:
      return &metadata.mobile();
    case PhoneNumberUtil::FIXED_LINE:
    case PhoneNumberUtil::FIXED_LINE_OR_MOBILE:
      return &metadata.fixed_line();
    case PhoneNumberUtil::SHARED_COST:
      return &metadata.shared_cost();
    case PhoneNumberUtil::VOIP:
      return &metadata.voip();
    case PhoneNumberUtil::PERSONAL_NUMBER:
      return &metadata.personal_number();
    case PhoneNumberUtil::PAGER:
      return &metadata.pager();
    case PhoneNumberUtil::UAN:
      return &metadata.uan();
    case PhoneNumberUtil::VOICEMAIL:
      return &metadata.voicemail();
    default:
      return &metadata.general_desc();
  }
}
void PrefixNumberWithCountryCallingCode(
    int country_calling_code, PhoneNumberUtil::PhoneNumberFormat number_format,
    std::string* formatted_number) {
  switch (number_format) {
    case PhoneNumberUtil::E164:
      formatted_number->insert(0, StrCat(kPlusSign, country_calling_code));
      return;
    case PhoneNumberUtil::INTERNATIONAL:
      formatted_number->insert(0, StrCat(kPlusSign, country_calling_code, " "));
      return;
    case PhoneNumberUtil::RFC3966:
      formatted_number->insert(0, StrCat(kRfc3966Prefix, kPlusSign,
                                         country_calling_code, "-"));
      return;
    case PhoneNumberUtil::NATIONAL:
    default:
      return;
  }
}
bool IsNationalNumberSuffixOfTheOther(const PhoneNumber& first_number,
                                      const PhoneNumber& second_number) {
  const std::string& first_number_national_number =
    SimpleItoa(static_cast<uint64>(first_number.national_number()));
  const std::string& second_number_national_number =
    SimpleItoa(static_cast<uint64>(second_number.national_number()));
  return HasSuffixString(first_number_national_number,
                         second_number_national_number) ||
         HasSuffixString(second_number_national_number,
                         first_number_national_number);
}
char32 ToUnicodeCodepoint(const char* unicode_char) {
  char32 codepoint;
  EncodingUtils::DecodeUTF8Char(unicode_char, &codepoint);
  return codepoint;
}
std::string ExtnDigits(int max_length) {
  return StrCat("([", kDigits, "]{1,", max_length, "})");
}
std::string CreateExtnPattern(bool for_parsing) {
  int ext_limit_after_explicit_label = 20;
  int ext_limit_after_likely_label = 15;
  int ext_limit_after_ambiguous_char = 9;
  int ext_limit_when_not_sure = 6;
  std::string explicit_ext_labels =
      "(?:e?xt(?:ensi(?:o\xCC\x81?|\xC3\xB3))?n?|(?:\xEF\xBD\x85)?"
      "\xEF\xBD\x98\xEF\xBD\x94(?:\xEF\xBD\x8E)?|\xD0\xB4\xD0\xBE\xD0\xB1|"
      "anexo)";
  std::string ambiguous_ext_labels =
      "(?:[x\xEF\xBD\x98#\xEF\xBC\x83~\xEF\xBD\x9E]|int|"
      "\xEF\xBD\x89\xEF\xBD\x8E\xEF\xBD\x94)";
  std::string ambiguous_separator = "[- ]+";
  std::string rfc_extn = StrCat(
      kRfc3966ExtnPrefix, ExtnDigits(ext_limit_after_explicit_label));
  std::string explicit_extn = StrCat(
      kPossibleSeparatorsBetweenNumberAndExtLabel, explicit_ext_labels,
      kPossibleCharsAfterExtLabel, ExtnDigits(ext_limit_after_explicit_label),
      kOptionalExtSuffix);
  std::string ambiguous_extn = StrCat(
      kPossibleSeparatorsBetweenNumberAndExtLabel, ambiguous_ext_labels,
      kPossibleCharsAfterExtLabel, ExtnDigits(ext_limit_after_ambiguous_char),
      kOptionalExtSuffix);
  std::string american_style_extn_with_suffix = StrCat(
      ambiguous_separator, ExtnDigits(ext_limit_when_not_sure), "#");
  std::string extension_pattern =
      StrCat(rfc_extn, "|", explicit_extn, "|", ambiguous_extn, "|",
             american_style_extn_with_suffix);
  if (for_parsing) {
    std::string auto_dialling_and_ext_labels_found = "(?:,{2}|;)";
    std::string possible_separators_number_extLabel_no_comma =
        "[ \xC2\xA0\\t]*";
    std::string auto_dialling_extn = StrCat(
      possible_separators_number_extLabel_no_comma,
      auto_dialling_and_ext_labels_found, kPossibleCharsAfterExtLabel,
      ExtnDigits(ext_limit_after_likely_label),
      kOptionalExtSuffix);
    std::string only_commas_extn = StrCat(
      possible_separators_number_extLabel_no_comma,
      "(?:,)+", kPossibleCharsAfterExtLabel,
      ExtnDigits(ext_limit_after_ambiguous_char),
      kOptionalExtSuffix);
    return StrCat(extension_pattern, "|",
                  auto_dialling_extn, "|",
                  only_commas_extn);
  }
  return extension_pattern;
}
void NormalizeHelper(const std::map<char32, char>& normalization_replacements,
                     bool remove_non_matches,
                     string* number) {
  DCHECK(number);
  UnicodeText number_as_unicode;
  number_as_unicode.PointToUTF8(number->data(), static_cast<int>(number->size()));
  if (!number_as_unicode.UTF8WasValid()) {
    number->clear();
    return;
  }
  std::string normalized_number;
  char unicode_char[5];
  for (UnicodeText::const_iterator it = number_as_unicode.begin();
       it != number_as_unicode.end();
       ++it) {
    std::map<char32, char>::const_iterator found_glyph_pair =
        normalization_replacements.find(*it);
    if (found_glyph_pair != normalization_replacements.end()) {
      normalized_number.push_back(found_glyph_pair->second);
    } else if (!remove_non_matches) {
      int char_len = it.get_utf8(unicode_char);
      normalized_number.append(unicode_char, char_len);
    }
  }
  number->assign(normalized_number);
}
bool DescHasPossibleNumberData(const PhoneNumberDesc& desc) {
  return desc.possible_length_size() != 1 || desc.possible_length(0) != -1;
}
bool DescHasData(const PhoneNumberDesc& desc) {
  return desc.has_example_number() || DescHasPossibleNumberData(desc) ||
         desc.has_national_number_pattern();
}
void GetSupportedTypesForMetadata(
    const PhoneMetadata& metadata,
    std::set<PhoneNumberUtil::PhoneNumberType>* types) {
  DCHECK(types);
  for (int i = 0; i <= static_cast<int>(PhoneNumberUtil::kMaxNumberType); ++i) {
    PhoneNumberUtil::PhoneNumberType type =
        static_cast<PhoneNumberUtil::PhoneNumberType>(i);
    if (type == PhoneNumberUtil::FIXED_LINE_OR_MOBILE ||
        type == PhoneNumberUtil::UNKNOWN) {
      continue;
    }
    if (DescHasData(*GetNumberDescByType(metadata, type))) {
      types->insert(type);
    }
  }
}
PhoneNumberUtil::ValidationResult TestNumberLength(
    const std::string& number, const PhoneMetadata& metadata,
    PhoneNumberUtil::PhoneNumberType type) {
  const PhoneNumberDesc* desc_for_type = GetNumberDescByType(metadata, type);
  RepeatedField<int> possible_lengths =
      desc_for_type->possible_length_size() == 0
          ? metadata.general_desc().possible_length()
          : desc_for_type->possible_length();
  RepeatedField<int> local_lengths =
      desc_for_type->possible_length_local_only();
  if (type == PhoneNumberUtil::FIXED_LINE_OR_MOBILE) {
    const PhoneNumberDesc* fixed_line_desc =
        GetNumberDescByType(metadata, PhoneNumberUtil::FIXED_LINE);
    if (!DescHasPossibleNumberData(*fixed_line_desc)) {
      return TestNumberLength(number, metadata, PhoneNumberUtil::MOBILE);
    } else {
      const PhoneNumberDesc* mobile_desc =
          GetNumberDescByType(metadata, PhoneNumberUtil::MOBILE);
      if (DescHasPossibleNumberData(*mobile_desc)) {
        possible_lengths.MergeFrom(
            mobile_desc->possible_length_size() == 0
            ? metadata.general_desc().possible_length()
            : mobile_desc->possible_length());
        std::sort(possible_lengths.begin(), possible_lengths.end());
        if (local_lengths.size() == 0) {
          local_lengths = mobile_desc->possible_length_local_only();
        } else {
          local_lengths.MergeFrom(mobile_desc->possible_length_local_only());
          std::sort(local_lengths.begin(), local_lengths.end());
        }
      }
    }
  }
  if (possible_lengths.Get(0) == -1) {
    return PhoneNumberUtil::INVALID_LENGTH;
  }
  int actual_length = static_cast<int>(number.length());
  if (std::find(local_lengths.begin(), local_lengths.end(), actual_length) !=
      local_lengths.end()) {
    return PhoneNumberUtil::IS_POSSIBLE_LOCAL_ONLY;
  }
  int minimum_length = possible_lengths.Get(0);
  if (minimum_length == actual_length) {
    return PhoneNumberUtil::IS_POSSIBLE;
  } else if (minimum_length > actual_length) {
    return PhoneNumberUtil::TOO_SHORT;
  } else if (*(possible_lengths.end() - 1) < actual_length) {
    return PhoneNumberUtil::TOO_LONG;
  }
  return std::find(possible_lengths.begin() + 1, possible_lengths.end(),
                   actual_length) != possible_lengths.end()
             ? PhoneNumberUtil::IS_POSSIBLE
             : PhoneNumberUtil::INVALID_LENGTH;
}
PhoneNumberUtil::ValidationResult TestNumberLength(
    const std::string& number, const PhoneMetadata& metadata) {
  return TestNumberLength(number, metadata, PhoneNumberUtil::UNKNOWN);
}
void CopyCoreFieldsOnly(const PhoneNumber& number, PhoneNumber* pruned_number) {
  pruned_number->set_country_code(number.country_code());
  pruned_number->set_national_number(number.national_number());
  if (!number.extension().empty()) {
    pruned_number->set_extension(number.extension());
  }
  if (number.italian_leading_zero()) {
    pruned_number->set_italian_leading_zero(true);
    pruned_number->set_number_of_leading_zeros(
        number.number_of_leading_zeros());
  }
}
bool IsMatch(const MatcherApi& matcher_api,
             const std::string& number, const PhoneNumberDesc& desc) {
  return matcher_api.MatchNationalNumber(number, desc, false);
}
}  
void PhoneNumberUtil::SetLogger(Logger* logger) {
  logger_.reset(logger);
  Logger::set_logger_impl(logger_.get());
}
class PhoneNumberRegExpsAndMappings {
 private:
  void InitializeMapsAndSets() {
    diallable_char_mappings_.insert(std::make_pair('+', '+'));
    diallable_char_mappings_.insert(std::make_pair('*', '*'));
    diallable_char_mappings_.insert(std::make_pair('#', '#'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("-"), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xEF\xBC\x8D" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x90" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x91" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x92" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x93" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x94" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x80\x95" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x88\x92" ), '-'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("/"), '/'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xEF\xBC\x8F" ), '/'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint(" "), ' '));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE3\x80\x80" ), ' '));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xE2\x81\xA0"), ' '));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("."), '.'));
    all_plus_number_grouping_symbols_.insert(
        std::make_pair(ToUnicodeCodepoint("\xEF\xBC\x8E" ), '.'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("A"), '2'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("B"), '2'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("C"), '2'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("D"), '3'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("E"), '3'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("F"), '3'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("G"), '4'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("H"), '4'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("I"), '4'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("J"), '5'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("K"), '5'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("L"), '5'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("M"), '6'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("N"), '6'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("O"), '6'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("P"), '7'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("Q"), '7'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("R"), '7'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("S"), '7'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("T"), '8'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("U"), '8'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("V"), '8'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("W"), '9'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("X"), '9'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("Y"), '9'));
    alpha_mappings_.insert(std::make_pair(ToUnicodeCodepoint("Z"), '9'));
    std::map<char32, char> lower_case_mappings;
    std::map<char32, char> alpha_letters;
    for (std::map<char32, char>::const_iterator it = alpha_mappings_.begin();
         it != alpha_mappings_.end();
         ++it) {
      if (it->first < 128) {
        char letter_as_upper = static_cast<char>(it->first);
        char32 letter_as_lower = static_cast<char32>(tolower(letter_as_upper));
        lower_case_mappings.insert(std::make_pair(letter_as_lower, it->second));
        alpha_letters.insert(std::make_pair(letter_as_lower, letter_as_upper));
        alpha_letters.insert(std::make_pair(it->first, letter_as_upper));
      }
    }
    alpha_mappings_.insert(lower_case_mappings.begin(),
                           lower_case_mappings.end());
    alpha_phone_mappings_.insert(alpha_mappings_.begin(),
                                 alpha_mappings_.end());
    all_plus_number_grouping_symbols_.insert(alpha_letters.begin(),
                                             alpha_letters.end());
    for (char c = '0'; c <= '9'; ++c) {
      diallable_char_mappings_.insert(std::make_pair(c, c));
      alpha_phone_mappings_.insert(std::make_pair(c, c));
      all_plus_number_grouping_symbols_.insert(std::make_pair(c, c));
    }
    mobile_token_mappings_.insert(std::make_pair(54, '9'));
    countries_without_national_prefix_with_area_codes_.insert(52);  
    geo_mobile_countries_without_mobile_area_codes_.insert(86);  
    geo_mobile_countries_.insert(52);  
    geo_mobile_countries_.insert(54);  
    geo_mobile_countries_.insert(55);  
    geo_mobile_countries_.insert(62);
    geo_mobile_countries_.insert(
        geo_mobile_countries_without_mobile_area_codes_.begin(),
        geo_mobile_countries_without_mobile_area_codes_.end());
  }
  const std::string valid_phone_number_;
  const std::string extn_patterns_for_parsing_;
  const std::string rfc3966_phone_digit_;
  const std::string alphanum_;
  const std::string rfc3966_domainlabel_;
  const std::string rfc3966_toplabel_;
 public:
  scoped_ptr<const AbstractRegExpFactory> regexp_factory_;
  scoped_ptr<RegExpCache> regexp_cache_;
  std::map<char32, char> diallable_char_mappings_;
  std::map<char32, char> alpha_mappings_;
  std::map<char32, char> alpha_phone_mappings_;
  std::map<char32, char> all_plus_number_grouping_symbols_;
  std::map<int, char> mobile_token_mappings_;
  std::set<int> countries_without_national_prefix_with_area_codes_;
  std::set<int> geo_mobile_countries_without_mobile_area_codes_;
  std::set<int> geo_mobile_countries_;
  scoped_ptr<const RegExp> single_international_prefix_;
  scoped_ptr<const RegExp> digits_pattern_;
  scoped_ptr<const RegExp> capturing_digit_pattern_;
  scoped_ptr<const RegExp> capturing_ascii_digits_pattern_;
  scoped_ptr<const RegExp> valid_start_char_pattern_;
  scoped_ptr<const RegExp> capture_up_to_second_number_start_pattern_;
  scoped_ptr<const RegExp> unwanted_end_char_pattern_;
  scoped_ptr<const RegExp> separator_pattern_;
  const std::string extn_patterns_for_matching_;
  scoped_ptr<const RegExp> extn_pattern_;
  scoped_ptr<const RegExp> valid_phone_number_pattern_;
  scoped_ptr<const RegExp> valid_alpha_phone_pattern_;
  scoped_ptr<const RegExp> first_group_capturing_pattern_;
  scoped_ptr<const RegExp> carrier_code_pattern_;
  scoped_ptr<const RegExp> plus_chars_pattern_;
  std::unique_ptr<const RegExp> rfc3966_global_number_digits_pattern_;
  std::unique_ptr<const RegExp> rfc3966_domainname_pattern_;
  PhoneNumberRegExpsAndMappings()
      : valid_phone_number_(
            StrCat(kDigits, "{", PhoneNumberUtil::kMinLengthForNsn, "}|[",
                   PhoneNumberUtil::kPlusChars, "]*(?:[",
                   PhoneNumberUtil::kValidPunctuation, kStarSign, "]*",
                   kDigits, "){3,}[", PhoneNumberUtil::kValidPunctuation,
                   kStarSign, kValidAlpha, kDigits, "]*")),
        extn_patterns_for_parsing_(CreateExtnPattern( true)),
        rfc3966_phone_digit_(
            StrCat("(", kDigits, "|", kRfc3966VisualSeparator, ")")),
        alphanum_(StrCat(kValidAlphaInclUppercase, kDigits)),
        rfc3966_domainlabel_(
            StrCat("[", alphanum_, "]+((\\-)*[", alphanum_, "])*")),
        rfc3966_toplabel_(StrCat("[", kValidAlphaInclUppercase,
                                 "]+((\\-)*[", alphanum_, "])*")),
        regexp_factory_(new RegExpFactory()),
        regexp_cache_(new RegExpCache(*regexp_factory_.get(), 128)),
        diallable_char_mappings_(),
        alpha_mappings_(),
        alpha_phone_mappings_(),
        all_plus_number_grouping_symbols_(),
        mobile_token_mappings_(),
        countries_without_national_prefix_with_area_codes_(),
        geo_mobile_countries_without_mobile_area_codes_(),
        geo_mobile_countries_(),
        single_international_prefix_(regexp_factory_->CreateRegExp(
            "[\\d]+(?:[~\xE2\x81\x93\xE2\x88\xBC\xEF\xBD\x9E][\\d]+)?")),
        digits_pattern_(
            regexp_factory_->CreateRegExp(StrCat("[", kDigits, "]*"))),
        capturing_digit_pattern_(
            regexp_factory_->CreateRegExp(StrCat("([", kDigits, "])"))),
        capturing_ascii_digits_pattern_(
            regexp_factory_->CreateRegExp("(\\d+)")),
        valid_start_char_pattern_(regexp_factory_->CreateRegExp(
            StrCat("[", PhoneNumberUtil::kPlusChars, kDigits, "]"))),
        capture_up_to_second_number_start_pattern_(
            regexp_factory_->CreateRegExp(
                PhoneNumberUtil::kCaptureUpToSecondNumberStart)),
        unwanted_end_char_pattern_(
            regexp_factory_->CreateRegExp("[^\\p{N}\\p{L}#]")),
        separator_pattern_(regexp_factory_->CreateRegExp(
            StrCat("[", PhoneNumberUtil::kValidPunctuation, "]+"))),
        extn_patterns_for_matching_(
            CreateExtnPattern( false)),
        extn_pattern_(regexp_factory_->CreateRegExp(
            StrCat("(?i)(?:", extn_patterns_for_parsing_, ")$"))),
        valid_phone_number_pattern_(regexp_factory_->CreateRegExp(
            StrCat("(?i)", valid_phone_number_,
                   "(?:", extn_patterns_for_parsing_, ")?"))),
        valid_alpha_phone_pattern_(regexp_factory_->CreateRegExp(
            StrCat("(?i)(?:.*?[", kValidAlpha, "]){3}"))),
        first_group_capturing_pattern_(
            regexp_factory_->CreateRegExp("(\\$\\d)")),
        carrier_code_pattern_(regexp_factory_->CreateRegExp("\\$CC")),
        plus_chars_pattern_(regexp_factory_->CreateRegExp(
            StrCat("[", PhoneNumberUtil::kPlusChars, "]+"))),
        rfc3966_global_number_digits_pattern_(regexp_factory_->CreateRegExp(
            StrCat("^\\", kPlusSign, rfc3966_phone_digit_, "*", kDigits,
                   rfc3966_phone_digit_, "*$"))),
        rfc3966_domainname_pattern_(regexp_factory_->CreateRegExp(StrCat(
            "^(", rfc3966_domainlabel_, "\\.)*", rfc3966_toplabel_, "\\.?$"))) {
    InitializeMapsAndSets();
  }
  PhoneNumberRegExpsAndMappings(const PhoneNumberRegExpsAndMappings&) = delete;
  PhoneNumberRegExpsAndMappings& operator=(
      const PhoneNumberRegExpsAndMappings&) = delete;
};
PhoneNumberUtil::PhoneNumberUtil()
    : logger_(Logger::set_logger_impl(new NullLogger())),
      matcher_api_(new RegexBasedMatcher()),
      reg_exps_(new PhoneNumberRegExpsAndMappings),
      country_calling_code_to_region_code_map_(
          new std::vector<IntRegionsPair>()),
      nanpa_regions_(new absl::node_hash_set<std::string>()),
      region_to_metadata_map_(new absl::node_hash_map<std::string, PhoneMetadata>()),
      country_code_to_non_geographical_metadata_map_(
          new absl::node_hash_map<int, PhoneMetadata>) {
  Logger::set_logger_impl(logger_.get());
  PhoneMetadataCollection metadata_collection;
  if (!LoadCompiledInMetadata(&metadata_collection)) {
    LOG(DFATAL) << "Could not parse compiled-in metadata.";
    return;
  }
  std::map<int, std::list<std::string>* > country_calling_code_to_region_map;
  for (RepeatedPtrField<PhoneMetadata>::const_iterator it =
           metadata_collection.metadata().begin();
       it != metadata_collection.metadata().end();
       ++it) {
    const std::string& region_code = it->id();
    if (region_code == RegionCode::GetUnknown()) {
      continue;
    }
    int country_calling_code = it->country_code();
    if (kRegionCodeForNonGeoEntity == region_code) {
      country_code_to_non_geographical_metadata_map_->insert(
          std::make_pair(country_calling_code, *it));
    } else {
      region_to_metadata_map_->insert(std::make_pair(region_code, *it));
    }
    std::map<int, std::list<std::string>* >::iterator calling_code_in_map =
        country_calling_code_to_region_map.find(country_calling_code);
    if (calling_code_in_map != country_calling_code_to_region_map.end()) {
      if (it->main_country_for_code()) {
        calling_code_in_map->second->push_front(region_code);
      } else {
        calling_code_in_map->second->push_back(region_code);
      }
    } else {
      std::list<std::string>* list_with_region_code =
          new std::list<std::string>();
      list_with_region_code->push_back(region_code);
      country_calling_code_to_region_map.insert(
          std::make_pair(country_calling_code, list_with_region_code));
    }
    if (country_calling_code == kNanpaCountryCode) {
        nanpa_regions_->insert(region_code);
    }
  }
  country_calling_code_to_region_code_map_->insert(
      country_calling_code_to_region_code_map_->begin(),
      country_calling_code_to_region_map.begin(),
      country_calling_code_to_region_map.end());
  std::sort(country_calling_code_to_region_code_map_->begin(),
            country_calling_code_to_region_code_map_->end(), OrderByFirst());
}
PhoneNumberUtil::~PhoneNumberUtil() {
  gtl::STLDeleteContainerPairSecondPointers(
      country_calling_code_to_region_code_map_->begin(),
      country_calling_code_to_region_code_map_->end());
}
void PhoneNumberUtil::GetSupportedRegions(std::set<std::string>* regions)
    const {
  DCHECK(regions);
  for (absl::node_hash_map<std::string, PhoneMetadata>::const_iterator it =
       region_to_metadata_map_->begin(); it != region_to_metadata_map_->end();
       ++it) {
    regions->insert(it->first);
  }
}
void PhoneNumberUtil::GetSupportedGlobalNetworkCallingCodes(
    std::set<int>* calling_codes) const {
  DCHECK(calling_codes);
  for (absl::node_hash_map<int, PhoneMetadata>::const_iterator it =
           country_code_to_non_geographical_metadata_map_->begin();
       it != country_code_to_non_geographical_metadata_map_->end(); ++it) {
    calling_codes->insert(it->first);
  }
}
void PhoneNumberUtil::GetSupportedCallingCodes(
    std::set<int>* calling_codes) const {
  DCHECK(calling_codes);
  for (std::vector<IntRegionsPair>::const_iterator it =
           country_calling_code_to_region_code_map_->begin();
       it != country_calling_code_to_region_code_map_->end(); ++it) {
    calling_codes->insert(it->first);
  }
}
void PhoneNumberUtil::GetSupportedTypesForRegion(
    const std::string& region_code,
    std::set<PhoneNumberType>* types) const {
  DCHECK(types);
  if (!IsValidRegionCode(region_code)) {
    LOG(WARNING) << "Invalid or unknown region code provided: " << region_code;
    return;
  }
  const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
  GetSupportedTypesForMetadata(*metadata, types);
}
void PhoneNumberUtil::GetSupportedTypesForNonGeoEntity(
    int country_calling_code,
    std::set<PhoneNumberType>* types) const {
  DCHECK(types);
  const PhoneMetadata* metadata =
      GetMetadataForNonGeographicalRegion(country_calling_code);
  if (metadata == NULL) {
    LOG(WARNING) << "Unknown country calling code for a non-geographical "
                 << "entity provided: "
                 << country_calling_code;
    return;
  }
  GetSupportedTypesForMetadata(*metadata, types);
}
PhoneNumberUtil* PhoneNumberUtil::GetInstance() {
  return Singleton<PhoneNumberUtil>::GetInstance();
}
const std::string& PhoneNumberUtil::GetExtnPatternsForMatching() const {
  return reg_exps_->extn_patterns_for_matching_;
}
bool PhoneNumberUtil::StartsWithPlusCharsPattern(
    const std::string& number) const {
  const scoped_ptr<RegExpInput> number_string_piece(
      reg_exps_->regexp_factory_->CreateInput(number));
  return reg_exps_->plus_chars_pattern_->Consume(number_string_piece.get());
}
bool PhoneNumberUtil::ContainsOnlyValidDigits(const std::string& s) const {
  return reg_exps_->digits_pattern_->FullMatch(s);
}
void PhoneNumberUtil::TrimUnwantedEndChars(std::string* number) const {
  DCHECK(number);
  UnicodeText number_as_unicode;
  number_as_unicode.PointToUTF8(number->data(), static_cast<int>(number->size()));
  if (!number_as_unicode.UTF8WasValid()) {
    number->clear();
    return;
  }
  char current_char[5];
  int len;
  UnicodeText::const_reverse_iterator reverse_it(number_as_unicode.end());
  for (; reverse_it.base() != number_as_unicode.begin(); ++reverse_it) {
    len = reverse_it.get_utf8(current_char);
    current_char[len] = '\0';
    if (!reg_exps_->unwanted_end_char_pattern_->FullMatch(current_char)) {
      break;
    }
  }
  number->assign(UnicodeText::UTF8Substring(number_as_unicode.begin(),
                                            reverse_it.base()));
}
bool PhoneNumberUtil::IsFormatEligibleForAsYouTypeFormatter(
    const std::string& format) const {
  const RegExp& eligible_format_pattern = reg_exps_->regexp_cache_->GetRegExp(
      StrCat("[", kValidPunctuation, "]*", "\\$1",
             "[", kValidPunctuation, "]*", "(\\$\\d",
             "[", kValidPunctuation, "]*)*"));
  return eligible_format_pattern.FullMatch(format);
}
bool PhoneNumberUtil::FormattingRuleHasFirstGroupOnly(
    const std::string& national_prefix_formatting_rule) const {
  const RegExp& first_group_only_prefix_pattern =
      reg_exps_->regexp_cache_->GetRegExp("\\(?\\$1\\)?");
  return national_prefix_formatting_rule.empty() ||
      first_group_only_prefix_pattern.FullMatch(
          national_prefix_formatting_rule);
}
void PhoneNumberUtil::GetNddPrefixForRegion(
    const std::string& region_code, bool strip_non_digits,
  std::string* national_prefix) const {
  DCHECK(national_prefix);
  const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
  if (!metadata) {
    LOG(WARNING) << "Invalid or unknown region code (" << region_code
                 << ") provided.";
    return;
  }
  national_prefix->assign(metadata->national_prefix());
  if (strip_non_digits) {
    strrmm(national_prefix, "~");
  }
}
bool PhoneNumberUtil::IsValidRegionCode(const std::string& region_code) const {
  return (region_to_metadata_map_->find(region_code) !=
          region_to_metadata_map_->end());
}
bool PhoneNumberUtil::HasValidCountryCallingCode(
    int country_calling_code) const {
  IntRegionsPair target_pair;
  target_pair.first = country_calling_code;
  return (std::binary_search(country_calling_code_to_region_code_map_->begin(),
                             country_calling_code_to_region_code_map_->end(),
                             target_pair, OrderByFirst()));
}
const PhoneMetadata* PhoneNumberUtil::GetMetadataForRegion(
    const std::string& region_code) const {
  absl::node_hash_map<std::string, PhoneMetadata>::const_iterator it =
      region_to_metadata_map_->find(region_code);
  if (it != region_to_metadata_map_->end()) {
    return &it->second;
  }
  return NULL;
}
const PhoneMetadata* PhoneNumberUtil::GetMetadataForNonGeographicalRegion(
    int country_calling_code) const {
  absl::node_hash_map<int, PhoneMetadata>::const_iterator it =
      country_code_to_non_geographical_metadata_map_->find(
          country_calling_code);
  if (it != country_code_to_non_geographical_metadata_map_->end()) {
    return &it->second;
  }
  return NULL;
}
void PhoneNumberUtil::Format(const PhoneNumber& number,
                             PhoneNumberFormat number_format,
                             std::string* formatted_number) const {
  DCHECK(formatted_number);
  if (number.national_number() == 0) {
    const std::string& raw_input = number.raw_input();
    if (!raw_input.empty()) {
      formatted_number->assign(raw_input);
      return;
    }
  }
  int country_calling_code = number.country_code();
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  if (number_format == E164) {
    formatted_number->assign(national_significant_number);
    PrefixNumberWithCountryCallingCode(country_calling_code, E164,
                                       formatted_number);
    return;
  }
  if (!HasValidCountryCallingCode(country_calling_code)) {
    formatted_number->assign(national_significant_number);
    return;
  }
  std::string region_code;
  GetRegionCodeForCountryCode(country_calling_code, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_calling_code, region_code);
  FormatNsn(national_significant_number, *metadata, number_format,
            formatted_number);
  MaybeAppendFormattedExtension(number, *metadata, number_format,
                                formatted_number);
  PrefixNumberWithCountryCallingCode(country_calling_code, number_format,
                                     formatted_number);
}
void PhoneNumberUtil::FormatByPattern(
    const PhoneNumber& number,
    PhoneNumberFormat number_format,
    const RepeatedPtrField<NumberFormat>& user_defined_formats,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  int country_calling_code = number.country_code();
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  if (!HasValidCountryCallingCode(country_calling_code)) {
    formatted_number->assign(national_significant_number);
    return;
  }
  std::string region_code;
  GetRegionCodeForCountryCode(country_calling_code, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_calling_code, region_code);
  const NumberFormat* formatting_pattern =
      ChooseFormattingPatternForNumber(user_defined_formats,
                                       national_significant_number);
  if (!formatting_pattern) {
    formatted_number->assign(national_significant_number);
  } else {
    NumberFormat num_format_copy;
    num_format_copy.MergeFrom(*formatting_pattern);
    std::string national_prefix_formatting_rule(
        formatting_pattern->national_prefix_formatting_rule());
    if (!national_prefix_formatting_rule.empty()) {
      const std::string& national_prefix = metadata->national_prefix();
      if (!national_prefix.empty()) {
        GlobalReplaceSubstring("$NP", national_prefix,
                            &national_prefix_formatting_rule);
        GlobalReplaceSubstring("$FG", "$1", &national_prefix_formatting_rule);
        num_format_copy.set_national_prefix_formatting_rule(
            national_prefix_formatting_rule);
      } else {
        num_format_copy.clear_national_prefix_formatting_rule();
      }
    }
    FormatNsnUsingPattern(national_significant_number, num_format_copy,
                          number_format, formatted_number);
  }
  MaybeAppendFormattedExtension(number, *metadata, NATIONAL, formatted_number);
  PrefixNumberWithCountryCallingCode(country_calling_code, number_format,
                                     formatted_number);
}
void PhoneNumberUtil::FormatNationalNumberWithCarrierCode(
    const PhoneNumber& number, const std::string& carrier_code,
    std::string* formatted_number) const {
  int country_calling_code = number.country_code();
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  if (!HasValidCountryCallingCode(country_calling_code)) {
    formatted_number->assign(national_significant_number);
    return;
  }
  std::string region_code;
  GetRegionCodeForCountryCode(country_calling_code, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_calling_code, region_code);
  FormatNsnWithCarrier(national_significant_number, *metadata, NATIONAL,
                       carrier_code, formatted_number);
  MaybeAppendFormattedExtension(number, *metadata, NATIONAL, formatted_number);
  PrefixNumberWithCountryCallingCode(country_calling_code, NATIONAL,
                                     formatted_number);
}
const PhoneMetadata* PhoneNumberUtil::GetMetadataForRegionOrCallingCode(
      int country_calling_code, const std::string& region_code) const {
  return kRegionCodeForNonGeoEntity == region_code
      ? GetMetadataForNonGeographicalRegion(country_calling_code)
      : GetMetadataForRegion(region_code);
}
void PhoneNumberUtil::FormatNationalNumberWithPreferredCarrierCode(
    const PhoneNumber& number,
    const std::string& fallback_carrier_code,
    std::string* formatted_number) const {
  FormatNationalNumberWithCarrierCode(
      number,
      !number.preferred_domestic_carrier_code().empty()
          ? number.preferred_domestic_carrier_code()
          : fallback_carrier_code,
      formatted_number);
}
void PhoneNumberUtil::FormatNumberForMobileDialing(
    const PhoneNumber& number,
    const std::string& calling_from,
    bool with_formatting,
    std::string* formatted_number) const {
  int country_calling_code = number.country_code();
  if (!HasValidCountryCallingCode(country_calling_code)) {
    formatted_number->assign(number.has_raw_input() ? number.raw_input() : "");
    return;
  }
  formatted_number->assign("");
  PhoneNumber number_no_extension(number);
  number_no_extension.clear_extension();
  std::string region_code;
  GetRegionCodeForCountryCode(country_calling_code, &region_code);
  PhoneNumberType number_type = GetNumberType(number_no_extension);
  bool is_valid_number = (number_type != UNKNOWN);
  if (calling_from == region_code) {
    bool is_fixed_line_or_mobile =
        (number_type == FIXED_LINE) || (number_type == MOBILE) ||
        (number_type == FIXED_LINE_OR_MOBILE);
    if ((region_code == "BR") && (is_fixed_line_or_mobile)) {
      if (!number_no_extension.preferred_domestic_carrier_code().empty()) {
        FormatNationalNumberWithPreferredCarrierCode(number_no_extension, "",
                                                     formatted_number);
      } else {
        formatted_number->assign("");
      }
    } else if (country_calling_code == kNanpaCountryCode) {
      const PhoneMetadata* region_metadata = GetMetadataForRegion(calling_from);
      std::string national_number;
      GetNationalSignificantNumber(number_no_extension, &national_number);
      if (CanBeInternationallyDialled(number_no_extension) &&
          TestNumberLength(national_number, *region_metadata) != TOO_SHORT) {
        Format(number_no_extension, INTERNATIONAL, formatted_number);
      } else {
        Format(number_no_extension, NATIONAL, formatted_number);
      }
    } else {
      if ((region_code == kRegionCodeForNonGeoEntity ||
           ((region_code == "MX" ||
             region_code == "CL" ||
             region_code == "UZ") &&
            is_fixed_line_or_mobile)) &&
          CanBeInternationallyDialled(number_no_extension)) {
        Format(number_no_extension, INTERNATIONAL, formatted_number);
      } else {
        Format(number_no_extension, NATIONAL, formatted_number);
      }
    }
  } else if (is_valid_number &&
      CanBeInternationallyDialled(number_no_extension)) {
    with_formatting
        ? Format(number_no_extension, INTERNATIONAL, formatted_number)
        : Format(number_no_extension, E164, formatted_number);
    return;
  }
  if (!with_formatting) {
    NormalizeDiallableCharsOnly(formatted_number);
  }
}
void PhoneNumberUtil::FormatOutOfCountryCallingNumber(
    const PhoneNumber& number, const std::string& calling_from,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  if (!IsValidRegionCode(calling_from)) {
    VLOG(1) << "Trying to format number from invalid region " << calling_from
            << ". International formatting applied.";
    Format(number, INTERNATIONAL, formatted_number);
    return;
  }
  int country_code = number.country_code();
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  if (!HasValidCountryCallingCode(country_code)) {
    formatted_number->assign(national_significant_number);
    return;
  }
  if (country_code == kNanpaCountryCode) {
    if (IsNANPACountry(calling_from)) {
      Format(number, NATIONAL, formatted_number);
      formatted_number->insert(0, StrCat(country_code, " "));
      return;
    }
  } else if (country_code == GetCountryCodeForValidRegion(calling_from)) {
    Format(number, NATIONAL, formatted_number);
    return;
  }
  const PhoneMetadata* metadata_calling_from =
      GetMetadataForRegion(calling_from);
  const std::string& international_prefix =
      metadata_calling_from->international_prefix();
  std::string international_prefix_for_formatting;
  if (metadata_calling_from->has_preferred_international_prefix()) {
    international_prefix_for_formatting =
        metadata_calling_from->preferred_international_prefix();
  } else if (reg_exps_->single_international_prefix_->FullMatch(
                 international_prefix)) {
    international_prefix_for_formatting = international_prefix;
  }
  std::string region_code;
  GetRegionCodeForCountryCode(country_code, &region_code);
  const PhoneMetadata* metadata_for_region =
      GetMetadataForRegionOrCallingCode(country_code, region_code);
  FormatNsn(national_significant_number, *metadata_for_region, INTERNATIONAL,
            formatted_number);
  MaybeAppendFormattedExtension(number, *metadata_for_region, INTERNATIONAL,
                                formatted_number);
  if (!international_prefix_for_formatting.empty()) {
    formatted_number->insert(
        0, StrCat(international_prefix_for_formatting, " ", country_code, " "));
  } else {
    PrefixNumberWithCountryCallingCode(country_code, INTERNATIONAL,
                                       formatted_number);
  }
}
void PhoneNumberUtil::FormatInOriginalFormat(
    const PhoneNumber& number, const std::string& region_calling_from,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  if (number.has_raw_input() && !HasFormattingPatternForNumber(number)) {
    formatted_number->assign(number.raw_input());
    return;
  }
  if (!number.has_country_code_source()) {
    Format(number, NATIONAL, formatted_number);
    return;
  }
  switch (number.country_code_source()) {
    case PhoneNumber::FROM_NUMBER_WITH_PLUS_SIGN:
      Format(number, INTERNATIONAL, formatted_number);
      break;
    case PhoneNumber::FROM_NUMBER_WITH_IDD:
      FormatOutOfCountryCallingNumber(number, region_calling_from,
                                      formatted_number);
      break;
    case PhoneNumber::FROM_NUMBER_WITHOUT_PLUS_SIGN:
      Format(number, INTERNATIONAL, formatted_number);
      formatted_number->erase(formatted_number->begin());
      break;
    case PhoneNumber::FROM_DEFAULT_COUNTRY:
    default:
      std::string region_code;
      GetRegionCodeForCountryCode(number.country_code(), &region_code);
      std::string national_prefix;
      GetNddPrefixForRegion(region_code, true ,
                            &national_prefix);
      if (national_prefix.empty()) {
        Format(number, NATIONAL, formatted_number);
        break;
      }
      if (RawInputContainsNationalPrefix(number.raw_input(), national_prefix,
                                         region_code)) {
        Format(number, NATIONAL, formatted_number);
        break;
      }
      const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
      std::string national_number;
      GetNationalSignificantNumber(number, &national_number);
      const NumberFormat* format_rule =
          ChooseFormattingPatternForNumber(metadata->number_format(),
                                           national_number);
      if (!format_rule) {
        Format(number, NATIONAL, formatted_number);
        break;
      }
      std::string candidate_national_prefix_rule(
          format_rule->national_prefix_formatting_rule());
      if (!candidate_national_prefix_rule.empty()) {
        size_t index_of_first_group = candidate_national_prefix_rule.find("$1");
        if (index_of_first_group == std::string::npos) {
          LOG(ERROR) << "First group missing in national prefix rule: "
              << candidate_national_prefix_rule;
          Format(number, NATIONAL, formatted_number);
          break;
        }
        candidate_national_prefix_rule.erase(index_of_first_group);
        NormalizeDigitsOnly(&candidate_national_prefix_rule);
      }
      if (candidate_national_prefix_rule.empty()) {
        Format(number, NATIONAL, formatted_number);
        break;
      }
      RepeatedPtrField<NumberFormat> number_formats;
      NumberFormat* number_format = number_formats.Add();
      number_format->MergeFrom(*format_rule);
      number_format->clear_national_prefix_formatting_rule();
      FormatByPattern(number, NATIONAL, number_formats, formatted_number);
      break;
  }
  if (!formatted_number->empty() && !number.raw_input().empty()) {
    std::string normalized_formatted_number(*formatted_number);
    NormalizeDiallableCharsOnly(&normalized_formatted_number);
    std::string normalized_raw_input(number.raw_input());
    NormalizeDiallableCharsOnly(&normalized_raw_input);
    if (normalized_formatted_number != normalized_raw_input) {
      formatted_number->assign(number.raw_input());
    }
  }
}
bool PhoneNumberUtil::RawInputContainsNationalPrefix(
    const std::string& raw_input, const std::string& national_prefix,
    const std::string& region_code) const {
  std::string normalized_national_number(raw_input);
  NormalizeDigitsOnly(&normalized_national_number);
  if (HasPrefixString(normalized_national_number, national_prefix)) {
    PhoneNumber number_without_national_prefix;
    if (Parse(normalized_national_number.substr(national_prefix.length()),
              region_code, &number_without_national_prefix)
        == NO_PARSING_ERROR) {
      return IsValidNumber(number_without_national_prefix);
    }
  }
  return false;
}
bool PhoneNumberUtil::HasFormattingPatternForNumber(
    const PhoneNumber& number) const {
  int country_calling_code = number.country_code();
  std::string region_code;
  GetRegionCodeForCountryCode(country_calling_code, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_calling_code, region_code);
  if (!metadata) {
    return false;
  }
  std::string national_number;
  GetNationalSignificantNumber(number, &national_number);
  const NumberFormat* format_rule =
      ChooseFormattingPatternForNumber(metadata->number_format(),
                                       national_number);
  return format_rule;
}
void PhoneNumberUtil::FormatOutOfCountryKeepingAlphaChars(
    const PhoneNumber& number, const std::string& calling_from,
    std::string* formatted_number) const {
  if (number.raw_input().empty()) {
    FormatOutOfCountryCallingNumber(number, calling_from, formatted_number);
    return;
  }
  int country_code = number.country_code();
  if (!HasValidCountryCallingCode(country_code)) {
    formatted_number->assign(number.raw_input());
    return;
  }
  std::string raw_input_copy(number.raw_input());
  NormalizeHelper(reg_exps_->all_plus_number_grouping_symbols_, true,
                  &raw_input_copy);
  std::string national_number;
  GetNationalSignificantNumber(number, &national_number);
  if (national_number.length() > 3) {
    size_t first_national_number_digit =
        raw_input_copy.find(national_number.substr(0, 3));
    if (first_national_number_digit != std::string::npos) {
      raw_input_copy = raw_input_copy.substr(first_national_number_digit);
    }
  }
  const PhoneMetadata* metadata = GetMetadataForRegion(calling_from);
  if (country_code == kNanpaCountryCode) {
    if (IsNANPACountry(calling_from)) {
      StrAppend(formatted_number, country_code, " ", raw_input_copy);
      return;
    }
  } else if (metadata &&
             country_code == GetCountryCodeForValidRegion(calling_from)) {
    const NumberFormat* formatting_pattern =
        ChooseFormattingPatternForNumber(metadata->number_format(),
                                         national_number);
    if (!formatting_pattern) {
      formatted_number->assign(raw_input_copy);
      return;
    }
    NumberFormat new_format;
    new_format.MergeFrom(*formatting_pattern);
    new_format.set_pattern("(\\d+)(.*)");
    new_format.set_format("$1$2");
    FormatNsnUsingPattern(raw_input_copy, new_format, NATIONAL,
                          formatted_number);
    return;
  }
  std::string international_prefix_for_formatting;
  if (metadata) {
    const std::string& international_prefix = metadata->international_prefix();
    international_prefix_for_formatting =
        reg_exps_->single_international_prefix_->FullMatch(international_prefix)
        ? international_prefix
        : metadata->preferred_international_prefix();
  }
  if (!international_prefix_for_formatting.empty()) {
    StrAppend(formatted_number, international_prefix_for_formatting, " ",
              country_code, " ", raw_input_copy);
  } else {
    if (!IsValidRegionCode(calling_from)) {
      VLOG(1) << "Trying to format number from invalid region " << calling_from
              << ". International formatting applied.";
    }
    formatted_number->assign(raw_input_copy);
    PrefixNumberWithCountryCallingCode(country_code, INTERNATIONAL,
                                       formatted_number);
  }
}
const NumberFormat* PhoneNumberUtil::ChooseFormattingPatternForNumber(
    const RepeatedPtrField<NumberFormat>& available_formats,
    const std::string& national_number) const {
  for (RepeatedPtrField<NumberFormat>::const_iterator
       it = available_formats.begin(); it != available_formats.end(); ++it) {
    int size = it->leading_digits_pattern_size();
    if (size > 0) {
      const scoped_ptr<RegExpInput> number_copy(
          reg_exps_->regexp_factory_->CreateInput(national_number));
      if (!reg_exps_->regexp_cache_->GetRegExp(
              it->leading_digits_pattern(size - 1)).Consume(
                  number_copy.get())) {
        continue;
      }
    }
    const RegExp& pattern_to_match(
        reg_exps_->regexp_cache_->GetRegExp(it->pattern()));
    if (pattern_to_match.FullMatch(national_number)) {
      return &(*it);
    }
  }
  return NULL;
}
void PhoneNumberUtil::FormatNsnUsingPatternWithCarrier(
    const std::string& national_number, const NumberFormat& formatting_pattern,
    PhoneNumberUtil::PhoneNumberFormat number_format,
    const std::string& carrier_code, std::string* formatted_number) const {
  DCHECK(formatted_number);
  std::string number_format_rule(formatting_pattern.format());
  if (number_format == PhoneNumberUtil::NATIONAL &&
      carrier_code.length() > 0 &&
      formatting_pattern.domestic_carrier_code_formatting_rule().length() > 0) {
    std::string carrier_code_formatting_rule =
        formatting_pattern.domestic_carrier_code_formatting_rule();
    reg_exps_->carrier_code_pattern_->Replace(&carrier_code_formatting_rule,
                                              carrier_code);
    reg_exps_->first_group_capturing_pattern_->
        Replace(&number_format_rule, carrier_code_formatting_rule);
  } else {
    std::string national_prefix_formatting_rule =
        formatting_pattern.national_prefix_formatting_rule();
    if (number_format == PhoneNumberUtil::NATIONAL &&
        national_prefix_formatting_rule.length() > 0) {
      reg_exps_->first_group_capturing_pattern_->Replace(
          &number_format_rule, national_prefix_formatting_rule);
    }
  }
  formatted_number->assign(national_number);
  const RegExp& pattern_to_match(
      reg_exps_->regexp_cache_->GetRegExp(formatting_pattern.pattern()));
  pattern_to_match.GlobalReplace(formatted_number, number_format_rule);
  if (number_format == RFC3966) {
    const scoped_ptr<RegExpInput> number(
        reg_exps_->regexp_factory_->CreateInput(*formatted_number));
    if (reg_exps_->separator_pattern_->Consume(number.get())) {
      formatted_number->assign(number->ToString());
    }
    reg_exps_->separator_pattern_->GlobalReplace(formatted_number, "-");
  }
}
void PhoneNumberUtil::FormatNsnUsingPattern(
    const std::string& national_number, const NumberFormat& formatting_pattern,
    PhoneNumberUtil::PhoneNumberFormat number_format,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  FormatNsnUsingPatternWithCarrier(national_number, formatting_pattern,
                                   number_format, "", formatted_number);
}
void PhoneNumberUtil::FormatNsn(const std::string& number,
                                const PhoneMetadata& metadata,
                                PhoneNumberFormat number_format,
                                std::string* formatted_number) const {
  DCHECK(formatted_number);
  FormatNsnWithCarrier(number, metadata, number_format, "", formatted_number);
}
void PhoneNumberUtil::FormatNsnWithCarrier(
    const std::string& number, const PhoneMetadata& metadata,
    PhoneNumberFormat number_format, const std::string& carrier_code,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  const RepeatedPtrField<NumberFormat> available_formats =
      (metadata.intl_number_format_size() == 0 || number_format == NATIONAL)
      ? metadata.number_format()
      : metadata.intl_number_format();
  const NumberFormat* formatting_pattern =
      ChooseFormattingPatternForNumber(available_formats, number);
  if (!formatting_pattern) {
    formatted_number->assign(number);
  } else {
    FormatNsnUsingPatternWithCarrier(number, *formatting_pattern, number_format,
                                     carrier_code, formatted_number);
  }
}
void PhoneNumberUtil::MaybeAppendFormattedExtension(
    const PhoneNumber& number,
    const PhoneMetadata& metadata,
    PhoneNumberFormat number_format,
    std::string* formatted_number) const {
  DCHECK(formatted_number);
  if (number.has_extension() && number.extension().length() > 0) {
    if (number_format == RFC3966) {
      StrAppend(formatted_number, kRfc3966ExtnPrefix, number.extension());
    } else {
      if (metadata.has_preferred_extn_prefix()) {
        StrAppend(formatted_number, metadata.preferred_extn_prefix(),
                  number.extension());
      } else {
        StrAppend(formatted_number, kDefaultExtnPrefix, number.extension());
      }
    }
  }
}
bool PhoneNumberUtil::IsNANPACountry(const std::string& region_code) const {
  return nanpa_regions_->find(region_code) != nanpa_regions_->end();
}
void PhoneNumberUtil::GetRegionCodesForCountryCallingCode(
    int country_calling_code,
    std::list<std::string>* region_codes) const {
  DCHECK(region_codes);
  IntRegionsPair target_pair;
  target_pair.first = country_calling_code;
  typedef std::vector<IntRegionsPair>::const_iterator ConstIterator;
  std::pair<ConstIterator, ConstIterator> range =
      std::equal_range(country_calling_code_to_region_code_map_->begin(),
                       country_calling_code_to_region_code_map_->end(),
                       target_pair, OrderByFirst());
  if (range.first != range.second) {
    region_codes->insert(region_codes->begin(),
                         range.first->second->begin(),
                         range.first->second->end());
  }
}
void PhoneNumberUtil::GetRegionCodeForCountryCode(
    int country_calling_code,
    std::string* region_code) const {
  DCHECK(region_code);
  std::list<std::string> region_codes;
  GetRegionCodesForCountryCallingCode(country_calling_code, &region_codes);
  *region_code = (region_codes.size() > 0) ?
      region_codes.front() : RegionCode::GetUnknown();
}
void PhoneNumberUtil::GetRegionCodeForNumber(
    const PhoneNumber& number, std::string* region_code) const {
  DCHECK(region_code);
  int country_calling_code = number.country_code();
  std::list<std::string> region_codes;
  GetRegionCodesForCountryCallingCode(country_calling_code, &region_codes);
  if (region_codes.size() == 0) {
    VLOG(1) << "Missing/invalid country calling code ("
            << country_calling_code << ")";
    *region_code = RegionCode::GetUnknown();
    return;
  }
  if (region_codes.size() == 1) {
    *region_code = region_codes.front();
  } else {
    GetRegionCodeForNumberFromRegionList(number, region_codes, region_code);
  }
}
void PhoneNumberUtil::GetRegionCodeForNumberFromRegionList(
    const PhoneNumber& number,
    const std::list<std::string>& region_codes,
    std::string* region_code) const {
  DCHECK(region_code);
  std::string national_number;
  GetNationalSignificantNumber(number, &national_number);
  for (std::list<std::string>::const_iterator it = region_codes.begin();
       it != region_codes.end(); ++it) {
    const PhoneMetadata* metadata = GetMetadataForRegion(*it);
    if (metadata->has_leading_digits()) {
      const scoped_ptr<RegExpInput> number(
          reg_exps_->regexp_factory_->CreateInput(national_number));
      if (reg_exps_->regexp_cache_->
              GetRegExp(metadata->leading_digits()).Consume(number.get())) {
        *region_code = *it;
        return;
      }
    } else if (GetNumberTypeHelper(national_number, *metadata) != UNKNOWN) {
      *region_code = *it;
      return;
    }
  }
  *region_code = RegionCode::GetUnknown();
}
int PhoneNumberUtil::GetCountryCodeForRegion(const std::string& region_code) const {
  if (!IsValidRegionCode(region_code)) {
    LOG(WARNING) << "Invalid or unknown region code (" << region_code
                 << ") provided.";
    return 0;
  }
  return GetCountryCodeForValidRegion(region_code);
}
int PhoneNumberUtil::GetCountryCodeForValidRegion(
    const std::string& region_code) const {
  const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
  return metadata->country_code();
}
bool PhoneNumberUtil::GetExampleNumber(const std::string& region_code,
                                       PhoneNumber* number) const {
  DCHECK(number);
  return GetExampleNumberForType(region_code, FIXED_LINE, number);
}
bool PhoneNumberUtil::GetInvalidExampleNumber(const std::string& region_code,
                                              PhoneNumber* number) const {
  DCHECK(number);
  if (!IsValidRegionCode(region_code)) {
    LOG(WARNING) << "Invalid or unknown region code (" << region_code
                 << ") provided.";
    return false;
  }
  const PhoneMetadata* region_metadata = GetMetadataForRegion(region_code);
  const PhoneNumberDesc* desc =
      GetNumberDescByType(*region_metadata, FIXED_LINE);
  if (!desc->has_example_number()) {
    return false;
  }
  const std::string& example_number = desc->example_number();
  for (size_t phone_number_length = example_number.length() - 1;
       phone_number_length >= kMinLengthForNsn;
       phone_number_length--) {
    std::string number_to_try = example_number.substr(0, phone_number_length);
    PhoneNumber possibly_valid_number;
    Parse(number_to_try, region_code, &possibly_valid_number);
    if (!IsValidNumber(possibly_valid_number)) {
      number->MergeFrom(possibly_valid_number);
      return true;
    }
  }
  return false;
}
bool PhoneNumberUtil::GetExampleNumberForType(
    const std::string& region_code,
    PhoneNumberUtil::PhoneNumberType type,
    PhoneNumber* number) const {
  DCHECK(number);
  if (!IsValidRegionCode(region_code)) {
    LOG(WARNING) << "Invalid or unknown region code (" << region_code
                 << ") provided.";
    return false;
  }
  const PhoneMetadata* region_metadata = GetMetadataForRegion(region_code);
  const PhoneNumberDesc* desc = GetNumberDescByType(*region_metadata, type);
  if (desc && desc->has_example_number()) {
    ErrorType success = Parse(desc->example_number(), region_code, number);
    if (success == NO_PARSING_ERROR) {
      return true;
    } else {
      LOG(ERROR) << "Error parsing example number ("
                 << static_cast<int>(success) << ")";
    }
  }
  return false;
}
bool PhoneNumberUtil::GetExampleNumberForType(
    PhoneNumberUtil::PhoneNumberType type,
    PhoneNumber* number) const {
  DCHECK(number);
  std::set<std::string> regions;
  GetSupportedRegions(&regions);
  for (const std::string& region_code : regions) {
    if (GetExampleNumberForType(region_code, type, number)) {
      return true;
    }
  }
  std::set<int> global_network_calling_codes;
  GetSupportedGlobalNetworkCallingCodes(&global_network_calling_codes);
  for (std::set<int>::const_iterator it = global_network_calling_codes.begin();
       it != global_network_calling_codes.end(); ++it) {
    int country_calling_code = *it;
    const PhoneMetadata* metadata =
        GetMetadataForNonGeographicalRegion(country_calling_code);
    const PhoneNumberDesc* desc = GetNumberDescByType(*metadata, type);
    if (desc->has_example_number()) {
      ErrorType success = Parse(StrCat(kPlusSign,
                                       country_calling_code,
                                       desc->example_number()),
                                RegionCode::GetUnknown(), number);
      if (success == NO_PARSING_ERROR) {
        return true;
      } else {
        LOG(ERROR) << "Error parsing example number ("
                   << static_cast<int>(success) << ")";
      }
    }
  }
  return false;
}
bool PhoneNumberUtil::GetExampleNumberForNonGeoEntity(
    int country_calling_code, PhoneNumber* number) const {
  DCHECK(number);
  const PhoneMetadata* metadata =
      GetMetadataForNonGeographicalRegion(country_calling_code);
  if (metadata) {
    const int kNumberTypes = 7;
    PhoneNumberDesc types[kNumberTypes] = {
        metadata->mobile(), metadata->toll_free(), metadata->shared_cost(),
        metadata->voip(), metadata->voicemail(), metadata->uan(),
        metadata->premium_rate()};
    for (int i = 0; i < kNumberTypes; ++i) {
      if (types[i].has_example_number()) {
        ErrorType success = Parse(StrCat(kPlusSign,
                                         SimpleItoa(country_calling_code),
                                         types[i].example_number()),
                                  RegionCode::GetUnknown(), number);
        if (success == NO_PARSING_ERROR) {
          return true;
        } else {
          LOG(ERROR) << "Error parsing example number ("
                     << static_cast<int>(success) << ")";
        }
      }
    }
  } else {
    LOG(WARNING) << "Invalid or unknown country calling code provided: "
                 << country_calling_code;
  }
  return false;
}
PhoneNumberUtil::ErrorType PhoneNumberUtil::Parse(
    absl::string_view number_to_parse, const std::string& default_region,
    PhoneNumber* number) const {
  DCHECK(number);
  return ParseHelper(number_to_parse, default_region, false, true, number);
}
PhoneNumberUtil::ErrorType PhoneNumberUtil::ParseAndKeepRawInput(
    absl::string_view number_to_parse, const std::string& default_region,
    PhoneNumber* number) const {
  DCHECK(number);
  return ParseHelper(number_to_parse, default_region, true, true, number);
}
bool PhoneNumberUtil::CheckRegionForParsing(
    const std::string& number_to_parse,
    const std::string& default_region) const {
  if (!IsValidRegionCode(default_region) && !number_to_parse.empty()) {
    const scoped_ptr<RegExpInput> number(
        reg_exps_->regexp_factory_->CreateInput(number_to_parse));
    if (!reg_exps_->plus_chars_pattern_->Consume(number.get())) {
      return false;
    }
  }
  return true;
}
absl::optional<absl::string_view> PhoneNumberUtil::ExtractPhoneContext(
    const absl::string_view number_to_extract_from,
    const size_t index_of_phone_context) const {
  if (index_of_phone_context == std::string::npos) {
    return absl::nullopt;
  }
  size_t phone_context_start =
      index_of_phone_context + strlen(kRfc3966PhoneContext);
  if (phone_context_start >= number_to_extract_from.length()) {
    return "";
  }
  size_t phone_context_end =
      number_to_extract_from.find(';', phone_context_start);
  if (phone_context_end != std::string::npos) {
    return number_to_extract_from.substr(
        phone_context_start, phone_context_end - phone_context_start);
  } else {
    return number_to_extract_from.substr(phone_context_start);
  }
}
bool PhoneNumberUtil::IsPhoneContextValid(
    const absl::optional<absl::string_view> phone_context) const {
  if (!phone_context.has_value()) {
    return true;
  }
  if (phone_context.value().empty()) {
    return false;
  }
  return reg_exps_->rfc3966_global_number_digits_pattern_->FullMatch(
      std::string{phone_context.value()}) ||
      reg_exps_->rfc3966_domainname_pattern_->FullMatch(
          std::string{phone_context.value()});
}
PhoneNumberUtil::ErrorType PhoneNumberUtil::BuildNationalNumberForParsing(
    absl::string_view number_to_parse, std::string* national_number) const {
  size_t index_of_phone_context = number_to_parse.find(kRfc3966PhoneContext);
  absl::optional<absl::string_view> phone_context =
      ExtractPhoneContext(number_to_parse, index_of_phone_context);
  if (!IsPhoneContextValid(phone_context)) {
    VLOG(2) << "The phone-context value is invalid.";
    return NOT_A_NUMBER;
  }
  if (phone_context.has_value()) {
    if (phone_context.value().at(0) == kPlusSign[0]) {
      StrAppend(national_number, phone_context.value());
    }
    size_t index_of_rfc_prefix = number_to_parse.find(kRfc3966Prefix);
    int index_of_national_number = (index_of_rfc_prefix != std::string::npos) ?
        static_cast<int>(index_of_rfc_prefix + strlen(kRfc3966Prefix)) : 0;
    StrAppend(
        national_number,
        number_to_parse.substr(
            index_of_national_number,
            index_of_phone_context - index_of_national_number));
  } else {
    ExtractPossibleNumber(number_to_parse, national_number);
  }
  size_t index_of_isdn = national_number->find(kRfc3966IsdnSubaddress);
  if (index_of_isdn != std::string::npos) {
    national_number->erase(index_of_isdn);
  }
  return NO_PARSING_ERROR;
}
PhoneNumberUtil::ErrorType PhoneNumberUtil::ParseHelper(
    absl::string_view number_to_parse, const std::string& default_region,
    bool keep_raw_input, bool check_region, PhoneNumber* phone_number) const {
  DCHECK(phone_number);
  std::string national_number;
  PhoneNumberUtil::ErrorType build_national_number_for_parsing_return =
      BuildNationalNumberForParsing(number_to_parse, &national_number);
  if (build_national_number_for_parsing_return != NO_PARSING_ERROR) {
    return build_national_number_for_parsing_return;
  }
  if (!IsViablePhoneNumber(national_number)) {
    VLOG(2) << "The string supplied did not seem to be a phone number.";
    return NOT_A_NUMBER;
  }
  if (check_region &&
      !CheckRegionForParsing(national_number, default_region)) {
    VLOG(1) << "Missing or invalid default country.";
    return INVALID_COUNTRY_CODE_ERROR;
  }
  PhoneNumber temp_number;
  if (keep_raw_input) {
    temp_number.set_raw_input(number_to_parse.data(), number_to_parse.size());
  }
  std::string extension;
  MaybeStripExtension(&national_number, &extension);
  if (!extension.empty()) {
    temp_number.set_extension(extension);
  }
  const PhoneMetadata* country_metadata = GetMetadataForRegion(default_region);
  std::string normalized_national_number(national_number);
  ErrorType country_code_error =
      MaybeExtractCountryCode(country_metadata, keep_raw_input,
                              &normalized_national_number, &temp_number);
  if (country_code_error != NO_PARSING_ERROR) {
    const scoped_ptr<RegExpInput> number_string_piece(
        reg_exps_->regexp_factory_->CreateInput(national_number));
    if ((country_code_error == INVALID_COUNTRY_CODE_ERROR) &&
        (reg_exps_->plus_chars_pattern_->Consume(number_string_piece.get()))) {
      normalized_national_number.assign(number_string_piece->ToString());
      MaybeExtractCountryCode(country_metadata,
                              keep_raw_input,
                              &normalized_national_number,
                              &temp_number);
      if (temp_number.country_code() == 0) {
        return INVALID_COUNTRY_CODE_ERROR;
      }
    } else {
      return country_code_error;
    }
  }
  int country_code = temp_number.country_code();
  if (country_code != 0) {
    std::string phone_number_region;
    GetRegionCodeForCountryCode(country_code, &phone_number_region);
    if (phone_number_region != default_region) {
      country_metadata =
          GetMetadataForRegionOrCallingCode(country_code, phone_number_region);
    }
  } else if (country_metadata) {
    country_code = country_metadata->country_code();
  }
  if (normalized_national_number.length() < kMinLengthForNsn) {
    VLOG(2) << "The string supplied is too short to be a phone number.";
    return TOO_SHORT_NSN;
  }
  if (country_metadata) {
    std::string carrier_code;
    std::string potential_national_number(normalized_national_number);
    MaybeStripNationalPrefixAndCarrierCode(*country_metadata,
                                           &potential_national_number,
                                           &carrier_code);
    ValidationResult validation_result =
        TestNumberLength(potential_national_number, *country_metadata);
    if (validation_result != TOO_SHORT &&
        validation_result != IS_POSSIBLE_LOCAL_ONLY &&
        validation_result != INVALID_LENGTH) {
      normalized_national_number.assign(potential_national_number);
      if (keep_raw_input && !carrier_code.empty()) {
        temp_number.set_preferred_domestic_carrier_code(carrier_code);
      }
    }
  }
  size_t normalized_national_number_length =
      normalized_national_number.length();
  if (normalized_national_number_length < kMinLengthForNsn) {
    VLOG(2) << "The string supplied is too short to be a phone number.";
    return TOO_SHORT_NSN;
  }
  if (normalized_national_number_length > kMaxLengthForNsn) {
    VLOG(2) << "The string supplied is too long to be a phone number.";
    return TOO_LONG_NSN;
  }
  temp_number.set_country_code(country_code);
  SetItalianLeadingZerosForPhoneNumber(normalized_national_number,
      &temp_number);
  uint64 number_as_int;
  safe_strtou64(normalized_national_number, &number_as_int);
  temp_number.set_national_number(number_as_int);
  phone_number->Swap(&temp_number);
  return NO_PARSING_ERROR;
}
void PhoneNumberUtil::ExtractPossibleNumber(
    absl::string_view number, std::string* extracted_number) const {
  DCHECK(extracted_number);
  UnicodeText number_as_unicode;
  number_as_unicode.PointToUTF8(number.data(), static_cast<int>(number.size()));
  if (!number_as_unicode.UTF8WasValid()) {
    extracted_number->clear();
    return;
  }
  char current_char[5];
  int len;
  UnicodeText::const_iterator it;
  for (it = number_as_unicode.begin(); it != number_as_unicode.end(); ++it) {
    len = it.get_utf8(current_char);
    current_char[len] = '\0';
    if (reg_exps_->valid_start_char_pattern_->FullMatch(current_char)) {
      break;
    }
  }
  if (it == number_as_unicode.end()) {
    extracted_number->clear();
    return;
  }
  extracted_number->assign(
      UnicodeText::UTF8Substring(it, number_as_unicode.end()));
  TrimUnwantedEndChars(extracted_number);
  if (extracted_number->length() == 0) {
    return;
  }
  reg_exps_->capture_up_to_second_number_start_pattern_->
      PartialMatch(*extracted_number, extracted_number);
}
bool PhoneNumberUtil::IsPossibleNumber(const PhoneNumber& number) const {
  ValidationResult result = IsPossibleNumberWithReason(number);
  return result == IS_POSSIBLE || result == IS_POSSIBLE_LOCAL_ONLY;
}
bool PhoneNumberUtil::IsPossibleNumberForType(
    const PhoneNumber& number, const PhoneNumberType type) const {
  ValidationResult result = IsPossibleNumberForTypeWithReason(number, type);
  return result == IS_POSSIBLE || result == IS_POSSIBLE_LOCAL_ONLY;
}
bool PhoneNumberUtil::IsPossibleNumberForString(
    absl::string_view number, const std::string& region_dialing_from) const {
  PhoneNumber number_proto;
  if (Parse(number, region_dialing_from, &number_proto) == NO_PARSING_ERROR) {
    return IsPossibleNumber(number_proto);
  } else {
    return false;
  }
}
PhoneNumberUtil::ValidationResult PhoneNumberUtil::IsPossibleNumberWithReason(
    const PhoneNumber& number) const {
  return IsPossibleNumberForTypeWithReason(number, PhoneNumberUtil::UNKNOWN);
}
PhoneNumberUtil::ValidationResult
PhoneNumberUtil::IsPossibleNumberForTypeWithReason(const PhoneNumber& number,
                                                   PhoneNumberType type) const {
  std::string national_number;
  GetNationalSignificantNumber(number, &national_number);
  int country_code = number.country_code();
  if (!HasValidCountryCallingCode(country_code)) {
    return INVALID_COUNTRY_CODE;
  }
  std::string region_code;
  GetRegionCodeForCountryCode(country_code, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_code, region_code);
  return TestNumberLength(national_number, *metadata, type);
}
bool PhoneNumberUtil::TruncateTooLongNumber(PhoneNumber* number) const {
  if (IsValidNumber(*number)) {
    return true;
  }
  PhoneNumber number_copy(*number);
  uint64 national_number = number->national_number();
  do {
    national_number /= 10;
    number_copy.set_national_number(national_number);
    if (IsPossibleNumberWithReason(number_copy) == TOO_SHORT ||
        national_number == 0) {
      return false;
    }
  } while (!IsValidNumber(number_copy));
  number->set_national_number(national_number);
  return true;
}
PhoneNumberUtil::PhoneNumberType PhoneNumberUtil::GetNumberType(
    const PhoneNumber& number) const {
  std::string region_code;
  GetRegionCodeForNumber(number, &region_code);
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(number.country_code(), region_code);
  if (!metadata) {
    return UNKNOWN;
  }
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  return GetNumberTypeHelper(national_significant_number, *metadata);
}
bool PhoneNumberUtil::IsValidNumber(const PhoneNumber& number) const {
  std::string region_code;
  GetRegionCodeForNumber(number, &region_code);
  return IsValidNumberForRegion(number, region_code);
}
bool PhoneNumberUtil::IsValidNumberForRegion(const PhoneNumber& number,
                                             const std::string& region_code) const {
  int country_code = number.country_code();
  const PhoneMetadata* metadata =
      GetMetadataForRegionOrCallingCode(country_code, region_code);
  if (!metadata ||
      ((kRegionCodeForNonGeoEntity != region_code) &&
       country_code != GetCountryCodeForValidRegion(region_code))) {
    return false;
  }
  std::string national_number;
  GetNationalSignificantNumber(number, &national_number);
  return GetNumberTypeHelper(national_number, *metadata) != UNKNOWN;
}
bool PhoneNumberUtil::IsNumberGeographical(
    const PhoneNumber& phone_number) const {
  return IsNumberGeographical(GetNumberType(phone_number),
                              phone_number.country_code());
}
bool PhoneNumberUtil::IsNumberGeographical(
    PhoneNumberType phone_number_type, int country_calling_code) const {
  return phone_number_type == PhoneNumberUtil::FIXED_LINE ||
      phone_number_type == PhoneNumberUtil::FIXED_LINE_OR_MOBILE ||
      (reg_exps_->geo_mobile_countries_.find(country_calling_code)
           != reg_exps_->geo_mobile_countries_.end() &&
       phone_number_type == PhoneNumberUtil::MOBILE);
}
void PhoneNumberUtil::SetItalianLeadingZerosForPhoneNumber(
    const std::string& national_number, PhoneNumber* phone_number) const {
  if (national_number.length() > 1 && national_number[0] == '0') {
    phone_number->set_italian_leading_zero(true);
    size_t number_of_leading_zeros = 1;
    while (number_of_leading_zeros < national_number.length() - 1 &&
        national_number[number_of_leading_zeros] == '0') {
      number_of_leading_zeros++;
    }
    if (number_of_leading_zeros != 1) {
      phone_number->set_number_of_leading_zeros(static_cast<int32_t>(number_of_leading_zeros));
    }
  }
}
bool PhoneNumberUtil::IsNumberMatchingDesc(
    const std::string& national_number, const PhoneNumberDesc& number_desc) const {
  int actual_length = static_cast<int>(national_number.length());
  if (number_desc.possible_length_size() > 0 &&
      std::find(number_desc.possible_length().begin(),
                number_desc.possible_length().end(),
                actual_length) == number_desc.possible_length().end()) {
    return false;
  }
  return IsMatch(*matcher_api_, national_number, number_desc);
}
PhoneNumberUtil::PhoneNumberType PhoneNumberUtil::GetNumberTypeHelper(
    const std::string& national_number, const PhoneMetadata& metadata) const {
  if (!IsNumberMatchingDesc(national_number, metadata.general_desc())) {
    VLOG(4) << "Number type unknown - doesn't match general national number"
            << " pattern.";
    return PhoneNumberUtil::UNKNOWN;
  }
  if (IsNumberMatchingDesc(national_number, metadata.premium_rate())) {
    VLOG(4) << "Number is a premium number.";
    return PhoneNumberUtil::PREMIUM_RATE;
  }
  if (IsNumberMatchingDesc(national_number, metadata.toll_free())) {
    VLOG(4) << "Number is a toll-free number.";
    return PhoneNumberUtil::TOLL_FREE;
  }
  if (IsNumberMatchingDesc(national_number, metadata.shared_cost())) {
    VLOG(4) << "Number is a shared cost number.";
    return PhoneNumberUtil::SHARED_COST;
  }
  if (IsNumberMatchingDesc(national_number, metadata.voip())) {
    VLOG(4) << "Number is a VOIP (Voice over IP) number.";
    return PhoneNumberUtil::VOIP;
  }
  if (IsNumberMatchingDesc(national_number, metadata.personal_number())) {
    VLOG(4) << "Number is a personal number.";
    return PhoneNumberUtil::PERSONAL_NUMBER;
  }
  if (IsNumberMatchingDesc(national_number, metadata.pager())) {
    VLOG(4) << "Number is a pager number.";
    return PhoneNumberUtil::PAGER;
  }
  if (IsNumberMatchingDesc(national_number, metadata.uan())) {
    VLOG(4) << "Number is a UAN.";
    return PhoneNumberUtil::UAN;
  }
  if (IsNumberMatchingDesc(national_number, metadata.voicemail())) {
    VLOG(4) << "Number is a voicemail number.";
    return PhoneNumberUtil::VOICEMAIL;
  }
  bool is_fixed_line =
      IsNumberMatchingDesc(national_number, metadata.fixed_line());
  if (is_fixed_line) {
    if (metadata.same_mobile_and_fixed_line_pattern()) {
      VLOG(4) << "Fixed-line and mobile patterns equal, number is fixed-line"
              << " or mobile";
      return PhoneNumberUtil::FIXED_LINE_OR_MOBILE;
    } else if (IsNumberMatchingDesc(national_number, metadata.mobile())) {
      VLOG(4) << "Fixed-line and mobile patterns differ, but number is "
              << "still fixed-line or mobile";
      return PhoneNumberUtil::FIXED_LINE_OR_MOBILE;
    }
    VLOG(4) << "Number is a fixed line number.";
    return PhoneNumberUtil::FIXED_LINE;
  }
  if (!metadata.same_mobile_and_fixed_line_pattern() &&
      IsNumberMatchingDesc(national_number, metadata.mobile())) {
    VLOG(4) << "Number is a mobile number.";
    return PhoneNumberUtil::MOBILE;
  }
  VLOG(4) << "Number type unknown - doesn\'t match any specific number type"
          << " pattern.";
  return PhoneNumberUtil::UNKNOWN;
}
void PhoneNumberUtil::GetNationalSignificantNumber(
    const PhoneNumber& number, std::string* national_number) const {
  DCHECK(national_number);
  StrAppend(national_number, number.italian_leading_zero() ?
      std::string(std::max(number.number_of_leading_zeros(), 0), '0') : "");
  StrAppend(national_number, number.national_number());
}
int PhoneNumberUtil::GetLengthOfGeographicalAreaCode(
    const PhoneNumber& number) const {
  std::string region_code;
  GetRegionCodeForNumber(number, &region_code);
  const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
  if (!metadata) {
    return 0;
  }
  PhoneNumberType type = GetNumberType(number);
  int country_calling_code = number.country_code();
  if (!metadata->has_national_prefix() && !number.italian_leading_zero() &&
      reg_exps_->countries_without_national_prefix_with_area_codes_.find(
          country_calling_code) ==
          reg_exps_->countries_without_national_prefix_with_area_codes_.end()) {
    return 0;
  }
  if (type == PhoneNumberUtil::MOBILE &&
      reg_exps_->geo_mobile_countries_without_mobile_area_codes_.find(
          country_calling_code) !=
          reg_exps_->geo_mobile_countries_without_mobile_area_codes_.end()) {
    return 0;
  }
  if (!IsNumberGeographical(type, country_calling_code)) {
    return 0;
  }
  return GetLengthOfNationalDestinationCode(number);
}
int PhoneNumberUtil::GetLengthOfNationalDestinationCode(
    const PhoneNumber& number) const {
  PhoneNumber copied_proto(number);
  if (number.has_extension()) {
    copied_proto.clear_extension();
  }
  std::string formatted_number;
  Format(copied_proto, INTERNATIONAL, &formatted_number);
  const scoped_ptr<RegExpInput> i18n_number(
      reg_exps_->regexp_factory_->CreateInput(formatted_number));
  std::string digit_group;
  std::string ndc;
  std::string third_group;
  for (int i = 0; i < 3; ++i) {
    if (!reg_exps_->capturing_ascii_digits_pattern_->FindAndConsume(
            i18n_number.get(), &digit_group)) {
      return 0;
    }
    if (i == 1) {
      ndc = digit_group;
    } else if (i == 2) {
      third_group = digit_group;
    }
  }
  if (GetNumberType(number) == MOBILE) {
    std::string mobile_token;
    GetCountryMobileToken(number.country_code(), &mobile_token);
    if (!mobile_token.empty()) {
      return static_cast<int>(third_group.size() + mobile_token.size());
    }
  }
  return static_cast<int>(ndc.size());
}
void PhoneNumberUtil::GetCountryMobileToken(int country_calling_code,
                                            std::string* mobile_token) const {
  DCHECK(mobile_token);
  std::map<int, char>::iterator it = reg_exps_->mobile_token_mappings_.find(
      country_calling_code);
  if (it != reg_exps_->mobile_token_mappings_.end()) {
    *mobile_token = it->second;
  } else {
    mobile_token->assign("");
  }
}
void PhoneNumberUtil::NormalizeDigitsOnly(std::string* number) const {
  DCHECK(number);
  const RegExp& non_digits_pattern = reg_exps_->regexp_cache_->GetRegExp(
      StrCat("[^", kDigits, "]"));
  non_digits_pattern.GlobalReplace(number, "");
  number->assign(NormalizeUTF8::NormalizeDecimalDigits(*number));
}
void PhoneNumberUtil::NormalizeDiallableCharsOnly(std::string* number) const {
  DCHECK(number);
  NormalizeHelper(reg_exps_->diallable_char_mappings_,
                  true , number);
}
bool PhoneNumberUtil::IsAlphaNumber(const std::string& number) const {
  if (!IsViablePhoneNumber(number)) {
    return false;
  }
  std::string number_copy(number);
  std::string extension;
  MaybeStripExtension(&number_copy, &extension);
  return reg_exps_->valid_alpha_phone_pattern_->FullMatch(number_copy);
}
void PhoneNumberUtil::ConvertAlphaCharactersInNumber(
    std::string* number) const {
  DCHECK(number);
  NormalizeHelper(reg_exps_->alpha_phone_mappings_, false, number);
}
void PhoneNumberUtil::Normalize(std::string* number) const {
  DCHECK(number);
  if (reg_exps_->valid_alpha_phone_pattern_->PartialMatch(*number)) {
    NormalizeHelper(reg_exps_->alpha_phone_mappings_, true, number);
  }
  NormalizeDigitsOnly(number);
}
bool PhoneNumberUtil::IsViablePhoneNumber(const std::string& number) const {
  if (number.length() < kMinLengthForNsn) {
    return false;
  }
  return reg_exps_->valid_phone_number_pattern_->FullMatch(number);
}
bool PhoneNumberUtil::ParsePrefixAsIdd(const RegExp& idd_pattern,
                                       std::string* number) const {
  DCHECK(number);
  const scoped_ptr<RegExpInput> number_copy(
      reg_exps_->regexp_factory_->CreateInput(*number));
  if (idd_pattern.Consume(number_copy.get())) {
    std::string extracted_digit;
    if (reg_exps_->capturing_digit_pattern_->PartialMatch(
            number_copy->ToString(), &extracted_digit)) {
      NormalizeDigitsOnly(&extracted_digit);
      if (extracted_digit == "0") {
        return false;
      }
    }
    number->assign(number_copy->ToString());
    return true;
  }
  return false;
}
PhoneNumber::CountryCodeSource
PhoneNumberUtil::MaybeStripInternationalPrefixAndNormalize(
    const std::string& possible_idd_prefix, std::string* number) const {
  DCHECK(number);
  if (number->empty()) {
    return PhoneNumber::FROM_DEFAULT_COUNTRY;
  }
  const scoped_ptr<RegExpInput> number_string_piece(
      reg_exps_->regexp_factory_->CreateInput(*number));
  if (reg_exps_->plus_chars_pattern_->Consume(number_string_piece.get())) {
    number->assign(number_string_piece->ToString());
    Normalize(number);
    return PhoneNumber::FROM_NUMBER_WITH_PLUS_SIGN;
  }
  const RegExp& idd_pattern =
      reg_exps_->regexp_cache_->GetRegExp(possible_idd_prefix);
  Normalize(number);
  return ParsePrefixAsIdd(idd_pattern, number)
      ? PhoneNumber::FROM_NUMBER_WITH_IDD
      : PhoneNumber::FROM_DEFAULT_COUNTRY;
}
bool PhoneNumberUtil::MaybeStripNationalPrefixAndCarrierCode(
    const PhoneMetadata& metadata, std::string* number,
    std::string* carrier_code) const {
  DCHECK(number);
  std::string carrier_code_temp;
  const std::string& possible_national_prefix =
      metadata.national_prefix_for_parsing();
  if (number->empty() || possible_national_prefix.empty()) {
    return false;
  }
  const scoped_ptr<RegExpInput> number_copy(
      reg_exps_->regexp_factory_->CreateInput(*number));
  const scoped_ptr<RegExpInput> number_copy_without_transform(
      reg_exps_->regexp_factory_->CreateInput(*number));
  std::string number_string_copy(*number);
  std::string captured_part_of_prefix;
  const PhoneNumberDesc& general_desc = metadata.general_desc();
  bool is_viable_original_number =
      IsMatch(*matcher_api_, *number, general_desc);
  const std::string& transform_rule = metadata.national_prefix_transform_rule();
  const RegExp& possible_national_prefix_pattern =
      reg_exps_->regexp_cache_->GetRegExp(possible_national_prefix);
  if (!transform_rule.empty() &&
      (possible_national_prefix_pattern.Consume(
          number_copy.get(), &carrier_code_temp, &captured_part_of_prefix) ||
       possible_national_prefix_pattern.Consume(
           number_copy.get(), &captured_part_of_prefix)) &&
      !captured_part_of_prefix.empty()) {
    possible_national_prefix_pattern.Replace(&number_string_copy,
                                             transform_rule);
    if (is_viable_original_number &&
        !IsMatch(*matcher_api_, number_string_copy, general_desc)) {
      return false;
    }
    number->assign(number_string_copy);
    if (carrier_code) {
      carrier_code->assign(carrier_code_temp);
    }
  } else if (possible_national_prefix_pattern.Consume(
                 number_copy_without_transform.get(), &carrier_code_temp) ||
             possible_national_prefix_pattern.Consume(
                 number_copy_without_transform.get())) {
    VLOG(4) << "Parsed the first digits as a national prefix.";
    const std::string number_copy_as_string =
        number_copy_without_transform->ToString();
    if (is_viable_original_number &&
        !IsMatch(*matcher_api_, number_copy_as_string, general_desc)) {
      return false;
    }
    number->assign(number_copy_as_string);
    if (carrier_code) {
      carrier_code->assign(carrier_code_temp);
    }
  } else {
    return false;
    VLOG(4) << "The first digits did not match the national prefix.";
  }
  return true;
}
bool PhoneNumberUtil::MaybeStripExtension(std::string* number,
                                          std::string* extension) const {
  DCHECK(number);
  DCHECK(extension);
  std::string possible_extension_one;
  std::string possible_extension_two;
  std::string possible_extension_three;
  std::string possible_extension_four;
  std::string possible_extension_five;
  std::string possible_extension_six;
  std::string number_copy(*number);
  const scoped_ptr<RegExpInput> number_copy_as_regexp_input(
      reg_exps_->regexp_factory_->CreateInput(number_copy));
  if (reg_exps_->extn_pattern_->Consume(
          number_copy_as_regexp_input.get(), false, &possible_extension_one,
          &possible_extension_two, &possible_extension_three,
          &possible_extension_four, &possible_extension_five,
          &possible_extension_six)) {
    reg_exps_->extn_pattern_->Replace(&number_copy, "");
    if ((!possible_extension_one.empty() || !possible_extension_two.empty() ||
         !possible_extension_three.empty() ||
         !possible_extension_four.empty() || !possible_extension_five.empty() ||
         !possible_extension_six.empty()) &&
        IsViablePhoneNumber(number_copy)) {
      number->assign(number_copy);
      if (!possible_extension_one.empty()) {
        extension->assign(possible_extension_one);
      } else if (!possible_extension_two.empty()) {
        extension->assign(possible_extension_two);
      } else if (!possible_extension_three.empty()) {
        extension->assign(possible_extension_three);
      } else if (!possible_extension_four.empty()) {
        extension->assign(possible_extension_four);
      } else if (!possible_extension_five.empty()) {
        extension->assign(possible_extension_five);
      } else if (!possible_extension_six.empty()) {
        extension->assign(possible_extension_six);
      }
      return true;
    }
  }
  return false;
}
int PhoneNumberUtil::ExtractCountryCode(std::string* national_number) const {
  int potential_country_code;
  if (national_number->empty() || (national_number->at(0) == '0')) {
    return 0;
  }
  for (size_t i = 1; i <= kMaxLengthCountryCode; ++i) {
    safe_strto32(national_number->substr(0, i), &potential_country_code);
    std::string region_code;
    GetRegionCodeForCountryCode(potential_country_code, &region_code);
    if (region_code != RegionCode::GetUnknown()) {
      national_number->erase(0, i);
      return potential_country_code;
    }
  }
  return 0;
}
PhoneNumberUtil::ErrorType PhoneNumberUtil::MaybeExtractCountryCode(
    const PhoneMetadata* default_region_metadata,
    bool keep_raw_input,
    std::string* national_number,
    PhoneNumber* phone_number) const {
  DCHECK(national_number);
  DCHECK(phone_number);
  std::string possible_country_idd_prefix = default_region_metadata
      ?  default_region_metadata->international_prefix()
      : "NonMatch";
  PhoneNumber::CountryCodeSource country_code_source =
      MaybeStripInternationalPrefixAndNormalize(possible_country_idd_prefix,
                                                national_number);
  if (keep_raw_input) {
    phone_number->set_country_code_source(country_code_source);
  }
  if (country_code_source != PhoneNumber::FROM_DEFAULT_COUNTRY) {
    if (national_number->length() <= kMinLengthForNsn) {
      VLOG(2) << "Phone number had an IDD, but after this was not "
              << "long enough to be a viable phone number.";
      return TOO_SHORT_AFTER_IDD;
    }
    int potential_country_code = ExtractCountryCode(national_number);
    if (potential_country_code != 0) {
      phone_number->set_country_code(potential_country_code);
      return NO_PARSING_ERROR;
    }
    return INVALID_COUNTRY_CODE_ERROR;
  } else if (default_region_metadata) {
    int default_country_code = default_region_metadata->country_code();
    std::string default_country_code_string(SimpleItoa(default_country_code));
    VLOG(4) << "Possible country calling code: " << default_country_code_string;
    std::string potential_national_number;
    if (TryStripPrefixString(*national_number,
                             default_country_code_string,
                             &potential_national_number)) {
      const PhoneNumberDesc& general_num_desc =
          default_region_metadata->general_desc();
      MaybeStripNationalPrefixAndCarrierCode(*default_region_metadata,
                                             &potential_national_number,
                                             NULL);
      VLOG(4) << "Number without country calling code prefix";
      if ((!IsMatch(*matcher_api_, *national_number, general_num_desc) &&
          IsMatch(
              *matcher_api_, potential_national_number, general_num_desc)) ||
          TestNumberLength(*national_number, *default_region_metadata) ==
              TOO_LONG) {
        national_number->assign(potential_national_number);
        if (keep_raw_input) {
          phone_number->set_country_code_source(
              PhoneNumber::FROM_NUMBER_WITHOUT_PLUS_SIGN);
        }
        phone_number->set_country_code(default_country_code);
        return NO_PARSING_ERROR;
      }
    }
  }
  phone_number->set_country_code(0);
  return NO_PARSING_ERROR;
}
PhoneNumberUtil::MatchType PhoneNumberUtil::IsNumberMatch(
    const PhoneNumber& first_number_in,
    const PhoneNumber& second_number_in) const {
  PhoneNumber first_number;
  CopyCoreFieldsOnly(first_number_in, &first_number);
  PhoneNumber second_number;
  CopyCoreFieldsOnly(second_number_in, &second_number);
  if (first_number.has_extension() && second_number.has_extension() &&
      first_number.extension() != second_number.extension()) {
    return NO_MATCH;
  }
  int first_number_country_code = first_number.country_code();
  int second_number_country_code = second_number.country_code();
  if (first_number_country_code != 0 && second_number_country_code != 0) {
    if (ExactlySameAs(first_number, second_number)) {
      return EXACT_MATCH;
    } else if (first_number_country_code == second_number_country_code &&
               IsNationalNumberSuffixOfTheOther(first_number, second_number)) {
      return SHORT_NSN_MATCH;
    }
    return NO_MATCH;
  }
  first_number.set_country_code(second_number_country_code);
  if (ExactlySameAs(first_number, second_number)) {
    return NSN_MATCH;
  }
  if (IsNationalNumberSuffixOfTheOther(first_number, second_number)) {
    return SHORT_NSN_MATCH;
  }
  return NO_MATCH;
}
PhoneNumberUtil::MatchType PhoneNumberUtil::IsNumberMatchWithTwoStrings(
    absl::string_view first_number, absl::string_view second_number) const {
  PhoneNumber first_number_as_proto;
  ErrorType error_type =
      Parse(first_number, RegionCode::GetUnknown(), &first_number_as_proto);
  if (error_type == NO_PARSING_ERROR) {
    return IsNumberMatchWithOneString(first_number_as_proto, second_number);
  }
  if (error_type == INVALID_COUNTRY_CODE_ERROR) {
    PhoneNumber second_number_as_proto;
    ErrorType error_type = Parse(second_number, RegionCode::GetUnknown(),
                                 &second_number_as_proto);
    if (error_type == NO_PARSING_ERROR) {
      return IsNumberMatchWithOneString(second_number_as_proto, first_number);
    }
    if (error_type == INVALID_COUNTRY_CODE_ERROR) {
      error_type  = ParseHelper(first_number, RegionCode::GetUnknown(), false,
                                false, &first_number_as_proto);
      if (error_type == NO_PARSING_ERROR) {
        error_type = ParseHelper(second_number, RegionCode::GetUnknown(), false,
                                 false, &second_number_as_proto);
        if (error_type == NO_PARSING_ERROR) {
          return IsNumberMatch(first_number_as_proto, second_number_as_proto);
        }
      }
    }
  }
  return INVALID_NUMBER;
}
PhoneNumberUtil::MatchType PhoneNumberUtil::IsNumberMatchWithOneString(
    const PhoneNumber& first_number, absl::string_view second_number) const {
  PhoneNumber second_number_as_proto;
  ErrorType error_type =
      Parse(second_number, RegionCode::GetUnknown(), &second_number_as_proto);
  if (error_type == NO_PARSING_ERROR) {
    return IsNumberMatch(first_number, second_number_as_proto);
  }
  if (error_type == INVALID_COUNTRY_CODE_ERROR) {
    std::string first_number_region;
    GetRegionCodeForCountryCode(first_number.country_code(),
                                &first_number_region);
    if (first_number_region != RegionCode::GetUnknown()) {
      PhoneNumber second_number_with_first_number_region;
      Parse(second_number, first_number_region,
            &second_number_with_first_number_region);
      MatchType match = IsNumberMatch(first_number,
                                      second_number_with_first_number_region);
      if (match == EXACT_MATCH) {
        return NSN_MATCH;
      }
      return match;
    } else {
      error_type = ParseHelper(second_number, RegionCode::GetUnknown(), false,
                               false, &second_number_as_proto);
      if (error_type == NO_PARSING_ERROR) {
        return IsNumberMatch(first_number, second_number_as_proto);
      }
    }
  }
  return INVALID_NUMBER;
}
AsYouTypeFormatter* PhoneNumberUtil::GetAsYouTypeFormatter(
    const std::string& region_code) const {
  return new AsYouTypeFormatter(region_code);
}
bool PhoneNumberUtil::CanBeInternationallyDialled(
    const PhoneNumber& number) const {
  std::string region_code;
  GetRegionCodeForNumber(number, &region_code);
  const PhoneMetadata* metadata = GetMetadataForRegion(region_code);
  if (!metadata) {
    return true;
  }
  std::string national_significant_number;
  GetNationalSignificantNumber(number, &national_significant_number);
  return !IsNumberMatchingDesc(
      national_significant_number, metadata->no_international_dialling());
}
}  
}  