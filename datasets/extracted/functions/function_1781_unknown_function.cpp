#ifndef I18N_PHONENUMBERS_GEOCODING_DATA
#define I18N_PHONENUMBERS_GEOCODING_DATA
#include <cstdint>
namespace i18n {
namespace phonenumbers {
struct CountryLanguages {
  const char** available_languages;
  const int available_languages_size;
};
struct PrefixDescriptions {
  const int32_t* prefixes;
  const int prefixes_size;
  const char** descriptions;
  const int32_t* possible_lengths;
  const int possible_lengths_size;
};
const int* get_country_calling_codes();
int get_country_calling_codes_size();
const CountryLanguages* get_country_languages(int index);
const char** get_prefix_language_code_pairs();
int get_prefix_language_code_pairs_size();
const PrefixDescriptions* get_prefix_descriptions(int index);
}  
}  
#endif  