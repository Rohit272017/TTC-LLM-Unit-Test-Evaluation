#ifndef I18N_PHONENUMBERS_REGEXP_ADAPTER_H_
#define I18N_PHONENUMBERS_REGEXP_ADAPTER_H_
#include <cstddef>
#include <string>
namespace i18n {
namespace phonenumbers {
using std::string;
class RegExpInput {
 public:
  virtual ~RegExpInput() {}
  virtual string ToString() const = 0;
};
class RegExp {
 public:
  virtual ~RegExp() {}
  virtual bool Consume(RegExpInput* input_string,
                       bool anchor_at_start,
                       string* matched_string1,
                       string* matched_string2,
                       string* matched_string3,
                       string* matched_string4,
                       string* matched_string5,
                       string* matched_string6) const = 0;
  inline bool Consume(RegExpInput* input_string, string* matched_string1,
                      string* matched_string2,
                      string* matched_string3,
                      string* matched_string4,
                      string* matched_string5,
                      string* matched_string6) const {
    return Consume(input_string, true, matched_string1, matched_string2,
                   matched_string3, matched_string4, matched_string5,
                   matched_string6);
  }
  inline bool Consume(RegExpInput* input_string, string* matched_string1,
                      string* matched_string2,
                      string* matched_string3,
                      string* matched_string4,
                      string* matched_string5) const {
    return Consume(input_string, true, matched_string1, matched_string2,
                   matched_string3, matched_string4, matched_string5, NULL);
  }
  inline bool Consume(RegExpInput* input_string, string* matched_string1,
                      string* matched_string2,
                      string* matched_string3,
                      string* matched_string4) const {
    return Consume(input_string, true, matched_string1, matched_string2,
                   matched_string3, matched_string4, NULL, NULL);
  }
  inline bool Consume(RegExpInput* input_string,
                      string* matched_string1,
                      string* matched_string2,
                      string* matched_string3) const {
    return Consume(input_string, true, matched_string1, matched_string2,
                   matched_string3, NULL, NULL, NULL);
  }
  inline bool Consume(RegExpInput* input_string,
                      string* matched_string1,
                      string* matched_string2) const {
    return Consume(input_string, true, matched_string1, matched_string2, NULL,
    		   NULL, NULL, NULL);
  }
  inline bool Consume(RegExpInput* input_string, string* matched_string) const {
    return Consume(input_string, true, matched_string, NULL, NULL, NULL, NULL,
    	   	   NULL);
  }
  inline bool Consume(RegExpInput* input_string) const {
    return Consume(input_string, true, NULL, NULL, NULL, NULL, NULL, NULL);
  }
  inline bool FindAndConsume(RegExpInput* input_string,
                             string* matched_string) const {
    return Consume(input_string, false, matched_string, NULL, NULL, NULL, NULL,
    	           NULL);
  }
  virtual bool Match(const string& input_string,
                     bool full_match,
                     string* matched_string) const = 0;
  inline bool PartialMatch(const string& input_string,
                           string* matched_string) const {
    return Match(input_string, false, matched_string);
  }
  inline bool PartialMatch(const string& input_string) const {
    return Match(input_string, false, NULL);
  }
  inline bool FullMatch(const string& input_string,
                        string* matched_string) const {
    return Match(input_string, true, matched_string);
  }
  inline bool FullMatch(const string& input_string) const {
    return Match(input_string, true, NULL);
  }
  virtual bool Replace(string* string_to_process,
                       bool global,
                       const string& replacement_string) const = 0;
  inline bool Replace(string* string_to_process,
                      const string& replacement_string) const {
    return Replace(string_to_process, false, replacement_string);
  }
  inline bool GlobalReplace(string* string_to_process,
                            const string& replacement_string) const {
    return Replace(string_to_process, true, replacement_string);
  }
};
class AbstractRegExpFactory {
 public:
  virtual ~AbstractRegExpFactory() {}
  virtual RegExpInput* CreateInput(const string& utf8_input) const = 0;
  virtual RegExp* CreateRegExp(const string& utf8_regexp) const = 0;
};
}  
}  
#endif  