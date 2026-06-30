#include "demangle.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include "utilities.h"
#if defined(HAVE___CXA_DEMANGLE)
#  include <cxxabi.h>
#endif
#if defined(GLOG_OS_WINDOWS)
#  include <dbghelp.h>
#endif
namespace google {
inline namespace glog_internal_namespace_ {
#if !defined(GLOG_OS_WINDOWS) && !defined(HAVE___CXA_DEMANGLE)
namespace {
struct AbbrevPair {
  const char* const abbrev;
  const char* const real_name;
};
const AbbrevPair kOperatorList[] = {
    {"nw", "new"},    {"na", "new[]"},    {"dl", "delete"}, {"da", "delete[]"},
    {"ps", "+"},      {"ng", "-"},        {"ad", "&"},      {"de", "*"},
    {"co", "~"},      {"pl", "+"},        {"mi", "-"},      {"ml", "*"},
    {"dv", "/"},      {"rm", "%"},        {"an", "&"},      {"or", "|"},
    {"eo", "^"},      {"aS", "="},        {"pL", "+="},     {"mI", "-="},
    {"mL", "*="},     {"dV", "/="},       {"rM", "%="},     {"aN", "&="},
    {"oR", "|="},     {"eO", "^="},       {"ls", "<<"},     {"rs", ">>"},
    {"lS", "<<="},    {"rS", ">>="},      {"eq", "=="},     {"ne", "!="},
    {"lt", "<"},      {"gt", ">"},        {"le", "<="},     {"ge", ">="},
    {"nt", "!"},      {"aa", "&&"},       {"oo", "||"},     {"pp", "++"},
    {"mm", "--"},     {"cm", ","},        {"pm", "->*"},    {"pt", "->"},
    {"cl", "()"},     {"ix", "[]"},       {"qu", "?"},      {"st", "sizeof"},
    {"sz", "sizeof"}, {nullptr, nullptr},
};
const AbbrevPair kBuiltinTypeList[] = {
    {"v", "void"},        {"w", "wchar_t"},
    {"b", "bool"},        {"c", "char"},
    {"a", "signed char"}, {"h", "unsigned char"},
    {"s", "short"},       {"t", "unsigned short"},
    {"i", "int"},         {"j", "unsigned int"},
    {"l", "long"},        {"m", "unsigned long"},
    {"x", "long long"},   {"y", "unsigned long long"},
    {"n", "__int128"},    {"o", "unsigned __int128"},
    {"f", "float"},       {"d", "double"},
    {"e", "long double"}, {"g", "__float128"},
    {"z", "ellipsis"},    {"Dn", "decltype(nullptr)"},
    {nullptr, nullptr}};
const AbbrevPair kSubstitutionList[] = {
    {"St", ""},
    {"Sa", "allocator"},
    {"Sb", "basic_string"},
    {"Ss", "string"},
    {"Si", "istream"},
    {"So", "ostream"},
    {"Sd", "iostream"},
    {nullptr, nullptr}};
struct State {
  const char* mangled_cur;   
  char* out_cur;             
  const char* out_begin;     
  const char* out_end;       
  const char* prev_name;     
  ssize_t prev_name_length;  
  short nest_level;          
  bool append;               
  bool overflowed;           
  uint32 local_level;
  uint32 expr_level;
  uint32 arg_level;
};
size_t StrLen(const char* str) {
  size_t len = 0;
  while (*str != '\0') {
    ++str;
    ++len;
  }
  return len;
}
bool AtLeastNumCharsRemaining(const char* str, ssize_t n) {
  for (ssize_t i = 0; i < n; ++i) {
    if (str[i] == '\0') {
      return false;
    }
  }
  return true;
}
bool StrPrefix(const char* str, const char* prefix) {
  size_t i = 0;
  while (str[i] != '\0' && prefix[i] != '\0' && str[i] == prefix[i]) {
    ++i;
  }
  return prefix[i] == '\0';  
}
void InitState(State* state, const char* mangled, char* out, size_t out_size) {
  state->mangled_cur = mangled;
  state->out_cur = out;
  state->out_begin = out;
  state->out_end = out + out_size;
  state->prev_name = nullptr;
  state->prev_name_length = -1;
  state->nest_level = -1;
  state->append = true;
  state->overflowed = false;
  state->local_level = 0;
  state->expr_level = 0;
  state->arg_level = 0;
}
bool ParseOneCharToken(State* state, const char one_char_token) {
  if (state->mangled_cur[0] == one_char_token) {
    ++state->mangled_cur;
    return true;
  }
  return false;
}
bool ParseTwoCharToken(State* state, const char* two_char_token) {
  if (state->mangled_cur[0] == two_char_token[0] &&
      state->mangled_cur[1] == two_char_token[1]) {
    state->mangled_cur += 2;
    return true;
  }
  return false;
}
bool ParseCharClass(State* state, const char* char_class) {
  const char* p = char_class;
  for (; *p != '\0'; ++p) {
    if (state->mangled_cur[0] == *p) {
      ++state->mangled_cur;
      return true;
    }
  }
  return false;
}
bool Optional(bool) { return true; }
using ParseFunc = bool (*)(State*);
bool OneOrMore(ParseFunc parse_func, State* state) {
  if (parse_func(state)) {
    while (parse_func(state)) {
    }
    return true;
  }
  return false;
}
bool ZeroOrMore(ParseFunc parse_func, State* state) {
  while (parse_func(state)) {
  }
  return true;
}
void Append(State* state, const char* const str, ssize_t length) {
  if (state->out_cur == nullptr) {
    state->overflowed = true;
    return;
  }
  for (ssize_t i = 0; i < length; ++i) {
    if (state->out_cur + 1 < state->out_end) {  
      *state->out_cur = str[i];
      ++state->out_cur;
    } else {
      state->overflowed = true;
      break;
    }
  }
  if (!state->overflowed) {
    *state->out_cur = '\0';  
  }
}
bool IsLower(char c) { return c >= 'a' && c <= 'z'; }
bool IsAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool IsDigit(char c) { return c >= '0' && c <= '9'; }
bool IsFunctionCloneSuffix(const char* str) {
  size_t i = 0;
  while (str[i] != '\0') {
    if (str[i] != '.' || !IsAlpha(str[i + 1])) {
      return false;
    }
    i += 2;
    while (IsAlpha(str[i])) {
      ++i;
    }
    if (str[i] != '.' || !IsDigit(str[i + 1])) {
      return false;
    }
    i += 2;
    while (IsDigit(str[i])) {
      ++i;
    }
  }
  return true;  
}
void MaybeAppendWithLength(State* state, const char* const str,
                           ssize_t length) {
  if (state->append && length > 0) {
    if (str[0] == '<' && state->out_begin < state->out_cur &&
        state->out_cur[-1] == '<') {
      Append(state, " ", 1);
    }
    if (IsAlpha(str[0]) || str[0] == '_') {
      state->prev_name = state->out_cur;
      state->prev_name_length = length;
    }
    Append(state, str, length);
  }
}
bool MaybeAppend(State* state, const char* const str) {
  if (state->append) {
    size_t length = StrLen(str);
    MaybeAppendWithLength(state, str, static_cast<ssize_t>(length));
  }
  return true;
}
bool EnterNestedName(State* state) {
  state->nest_level = 0;
  return true;
}
bool LeaveNestedName(State* state, short prev_value) {
  state->nest_level = prev_value;
  return true;
}
bool DisableAppend(State* state) {
  state->append = false;
  return true;
}
bool RestoreAppend(State* state, bool prev_value) {
  state->append = prev_value;
  return true;
}
void MaybeIncreaseNestLevel(State* state) {
  if (state->nest_level > -1) {
    ++state->nest_level;
  }
}
void MaybeAppendSeparator(State* state) {
  if (state->nest_level >= 1) {
    MaybeAppend(state, "::");
  }
}
void MaybeCancelLastSeparator(State* state) {
  if (state->nest_level >= 1 && state->append &&
      state->out_begin <= state->out_cur - 2) {
    state->out_cur -= 2;
    *state->out_cur = '\0';
  }
}
bool IdentifierIsAnonymousNamespace(State* state, ssize_t length) {
  const char anon_prefix[] = "_GLOBAL__N_";
  return (length > static_cast<ssize_t>(sizeof(anon_prefix)) -
                       1 &&  
          StrPrefix(state->mangled_cur, anon_prefix));
}
bool ParseMangledName(State* state);
bool ParseEncoding(State* state);
bool ParseName(State* state);
bool ParseUnscopedName(State* state);
bool ParseUnscopedTemplateName(State* state);
bool ParseNestedName(State* state);
bool ParsePrefix(State* state);
bool ParseUnqualifiedName(State* state);
bool ParseSourceName(State* state);
bool ParseLocalSourceName(State* state);
bool ParseNumber(State* state, int* number_out);
bool ParseFloatNumber(State* state);
bool ParseSeqId(State* state);
bool ParseIdentifier(State* state, ssize_t length);
bool ParseAbiTags(State* state);
bool ParseAbiTag(State* state);
bool ParseOperatorName(State* state);
bool ParseSpecialName(State* state);
bool ParseCallOffset(State* state);
bool ParseNVOffset(State* state);
bool ParseVOffset(State* state);
bool ParseCtorDtorName(State* state);
bool ParseType(State* state);
bool ParseCVQualifiers(State* state);
bool ParseBuiltinType(State* state);
bool ParseFunctionType(State* state);
bool ParseBareFunctionType(State* state);
bool ParseClassEnumType(State* state);
bool ParseArrayType(State* state);
bool ParsePointerToMemberType(State* state);
bool ParseTemplateParam(State* state);
bool ParseTemplateTemplateParam(State* state);
bool ParseTemplateArgs(State* state);
bool ParseTemplateArg(State* state);
bool ParseExpression(State* state);
bool ParseExprPrimary(State* state);
bool ParseLocalName(State* state);
bool ParseDiscriminator(State* state);
bool ParseSubstitution(State* state);
bool ParseMangledName(State* state) {
  return ParseTwoCharToken(state, "_Z") && ParseEncoding(state);
}
bool ParseEncoding(State* state) {
  State copy = *state;
  if (ParseName(state) && ParseBareFunctionType(state)) {
    return true;
  }
  *state = copy;
  if (ParseName(state) || ParseSpecialName(state)) {
    return true;
  }
  return false;
}
bool ParseName(State* state) {
  if (ParseNestedName(state) || ParseLocalName(state)) {
    return true;
  }
  State copy = *state;
  if (ParseUnscopedTemplateName(state) && ParseTemplateArgs(state)) {
    return true;
  }
  *state = copy;
  if (ParseUnscopedName(state)) {
    return true;
  }
  return false;
}
bool ParseUnscopedName(State* state) {
  if (ParseUnqualifiedName(state)) {
    return true;
  }
  State copy = *state;
  if (ParseTwoCharToken(state, "St") && MaybeAppend(state, "std::") &&
      ParseUnqualifiedName(state)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseUnscopedTemplateName(State* state) {
  return ParseUnscopedName(state) || ParseSubstitution(state);
}
bool ParseNestedName(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'N') && EnterNestedName(state) &&
      Optional(ParseCVQualifiers(state)) && ParsePrefix(state) &&
      LeaveNestedName(state, copy.nest_level) &&
      ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParsePrefix(State* state) {
  bool has_something = false;
  while (true) {
    MaybeAppendSeparator(state);
    if (ParseTemplateParam(state) || ParseSubstitution(state) ||
        ParseUnscopedName(state)) {
      has_something = true;
      MaybeIncreaseNestLevel(state);
      continue;
    }
    MaybeCancelLastSeparator(state);
    if (has_something && ParseTemplateArgs(state)) {
      return ParsePrefix(state);
    } else {
      break;
    }
  }
  return true;
}
bool ParseUnqualifiedName(State* state) {
  return (ParseOperatorName(state) || ParseCtorDtorName(state) ||
          (ParseSourceName(state) && Optional(ParseAbiTags(state))) ||
          (ParseLocalSourceName(state) && Optional(ParseAbiTags(state))));
}
bool ParseSourceName(State* state) {
  State copy = *state;
  int length = -1;
  if (ParseNumber(state, &length) && ParseIdentifier(state, length)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseLocalSourceName(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'L') && ParseSourceName(state) &&
      Optional(ParseDiscriminator(state))) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseNumber(State* state, int* number_out) {
  int sign = 1;
  if (ParseOneCharToken(state, 'n')) {
    sign = -1;
  }
  const char* p = state->mangled_cur;
  int number = 0;
  constexpr int int_max_by_10 = std::numeric_limits<int>::max() / 10;
  for (; *p != '\0'; ++p) {
    if (IsDigit(*p)) {
      if (number > int_max_by_10) {
        return false;
      }
      const int digit = *p - '0';
      const int shifted = number * 10;
      if (digit > std::numeric_limits<int>::max() - shifted) {
        return false;
      }
      number = shifted + digit;
    } else {
      break;
    }
  }
  if (p != state->mangled_cur) {  
    state->mangled_cur = p;
    if (number_out != nullptr) {
      *number_out = number * sign;
    }
    return true;
  }
  return false;
}
bool ParseFloatNumber(State* state) {
  const char* p = state->mangled_cur;
  for (; *p != '\0'; ++p) {
    if (!IsDigit(*p) && !(*p >= 'a' && *p <= 'f')) {
      break;
    }
  }
  if (p != state->mangled_cur) {  
    state->mangled_cur = p;
    return true;
  }
  return false;
}
bool ParseSeqId(State* state) {
  const char* p = state->mangled_cur;
  for (; *p != '\0'; ++p) {
    if (!IsDigit(*p) && !(*p >= 'A' && *p <= 'Z')) {
      break;
    }
  }
  if (p != state->mangled_cur) {  
    state->mangled_cur = p;
    return true;
  }
  return false;
}
bool ParseIdentifier(State* state, ssize_t length) {
  if (length == -1 || !AtLeastNumCharsRemaining(state->mangled_cur, length)) {
    return false;
  }
  if (IdentifierIsAnonymousNamespace(state, length)) {
    MaybeAppend(state, "(anonymous namespace)");
  } else {
    MaybeAppendWithLength(state, state->mangled_cur, length);
  }
  if (length < 0 ||
      static_cast<std::size_t>(length) > StrLen(state->mangled_cur)) {
    return false;
  }
  state->mangled_cur += length;
  return true;
}
bool ParseAbiTags(State* state) {
  State copy = *state;
  DisableAppend(state);
  if (OneOrMore(ParseAbiTag, state)) {
    RestoreAppend(state, copy.append);
    return true;
  }
  *state = copy;
  return false;
}
bool ParseAbiTag(State* state) {
  return ParseOneCharToken(state, 'B') && ParseSourceName(state);
}
bool ParseOperatorName(State* state) {
  if (!AtLeastNumCharsRemaining(state->mangled_cur, 2)) {
    return false;
  }
  State copy = *state;
  if (ParseTwoCharToken(state, "cv") && MaybeAppend(state, "operator ") &&
      EnterNestedName(state) && ParseType(state) &&
      LeaveNestedName(state, copy.nest_level)) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'v') && ParseCharClass(state, "0123456789") &&
      ParseSourceName(state)) {
    return true;
  }
  *state = copy;
  if (!(IsLower(state->mangled_cur[0]) && IsAlpha(state->mangled_cur[1]))) {
    return false;
  }
  const AbbrevPair* p;
  for (p = kOperatorList; p->abbrev != nullptr; ++p) {
    if (state->mangled_cur[0] == p->abbrev[0] &&
        state->mangled_cur[1] == p->abbrev[1]) {
      MaybeAppend(state, "operator");
      if (IsLower(*p->real_name)) {  
        MaybeAppend(state, " ");
      }
      MaybeAppend(state, p->real_name);
      state->mangled_cur += 2;
      return true;
    }
  }
  return false;
}
bool ParseSpecialName(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'T') && ParseCharClass(state, "VTIS") &&
      ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "Tc") && ParseCallOffset(state) &&
      ParseCallOffset(state) && ParseEncoding(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "GV") && ParseName(state)) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'T') && ParseCallOffset(state) &&
      ParseEncoding(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "TC") && ParseType(state) &&
      ParseNumber(state, nullptr) && ParseOneCharToken(state, '_') &&
      DisableAppend(state) && ParseType(state)) {
    RestoreAppend(state, copy.append);
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'T') && ParseCharClass(state, "FJ") &&
      ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "GR") && ParseName(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "GA") && ParseEncoding(state)) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'T') && ParseCharClass(state, "hv") &&
      ParseCallOffset(state) && ParseEncoding(state)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseCallOffset(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'h') && ParseNVOffset(state) &&
      ParseOneCharToken(state, '_')) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'v') && ParseVOffset(state) &&
      ParseOneCharToken(state, '_')) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseNVOffset(State* state) { return ParseNumber(state, nullptr); }
bool ParseVOffset(State* state) {
  State copy = *state;
  if (ParseNumber(state, nullptr) && ParseOneCharToken(state, '_') &&
      ParseNumber(state, nullptr)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseCtorDtorName(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'C') && ParseCharClass(state, "123")) {
    const char* const prev_name = state->prev_name;
    const ssize_t prev_name_length = state->prev_name_length;
    MaybeAppendWithLength(state, prev_name, prev_name_length);
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'D') && ParseCharClass(state, "012")) {
    const char* const prev_name = state->prev_name;
    const ssize_t prev_name_length = state->prev_name_length;
    MaybeAppend(state, "~");
    MaybeAppendWithLength(state, prev_name, prev_name_length);
    return true;
  }
  *state = copy;
  return false;
}
bool ParseType(State* state) {
  State copy = *state;
  if (ParseCVQualifiers(state) && ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseCharClass(state, "OPRCG") && ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "Dp") && ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'D') && ParseCharClass(state, "tT") &&
      ParseExpression(state) && ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'U') && ParseSourceName(state) &&
      ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseBuiltinType(state) || ParseFunctionType(state) ||
      ParseClassEnumType(state) || ParseArrayType(state) ||
      ParsePointerToMemberType(state) || ParseSubstitution(state)) {
    return true;
  }
  if (ParseTemplateTemplateParam(state) && ParseTemplateArgs(state)) {
    return true;
  }
  *state = copy;
  if (ParseTemplateParam(state)) {
    return true;
  }
  return false;
}
bool ParseCVQualifiers(State* state) {
  int num_cv_qualifiers = 0;
  num_cv_qualifiers += ParseOneCharToken(state, 'r');
  num_cv_qualifiers += ParseOneCharToken(state, 'V');
  num_cv_qualifiers += ParseOneCharToken(state, 'K');
  return num_cv_qualifiers > 0;
}
bool ParseBuiltinType(State* state) {
  const AbbrevPair* p;
  for (p = kBuiltinTypeList; p->abbrev != nullptr; ++p) {
    if (state->mangled_cur[0] == p->abbrev[0]) {
      MaybeAppend(state, p->real_name);
      ++state->mangled_cur;
      return true;
    }
  }
  State copy = *state;
  if (ParseOneCharToken(state, 'u') && ParseSourceName(state)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseFunctionType(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'F') &&
      Optional(ParseOneCharToken(state, 'Y')) && ParseBareFunctionType(state) &&
      ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseBareFunctionType(State* state) {
  State copy = *state;
  DisableAppend(state);
  if (OneOrMore(ParseType, state)) {
    RestoreAppend(state, copy.append);
    MaybeAppend(state, "()");
    return true;
  }
  *state = copy;
  return false;
}
bool ParseClassEnumType(State* state) { return ParseName(state); }
bool ParseArrayType(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'A') && ParseNumber(state, nullptr) &&
      ParseOneCharToken(state, '_') && ParseType(state)) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'A') && Optional(ParseExpression(state)) &&
      ParseOneCharToken(state, '_') && ParseType(state)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParsePointerToMemberType(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'M') && ParseType(state) && ParseType(state)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseTemplateParam(State* state) {
  if (ParseTwoCharToken(state, "T_")) {
    MaybeAppend(state, "?");  
    return true;
  }
  State copy = *state;
  if (ParseOneCharToken(state, 'T') && ParseNumber(state, nullptr) &&
      ParseOneCharToken(state, '_')) {
    MaybeAppend(state, "?");  
    return true;
  }
  *state = copy;
  return false;
}
bool ParseTemplateTemplateParam(State* state) {
  return (ParseTemplateParam(state) || ParseSubstitution(state));
}
bool ParseTemplateArgs(State* state) {
  State copy = *state;
  DisableAppend(state);
  if (ParseOneCharToken(state, 'I') && OneOrMore(ParseTemplateArg, state) &&
      ParseOneCharToken(state, 'E')) {
    RestoreAppend(state, copy.append);
    MaybeAppend(state, "<>");
    return true;
  }
  *state = copy;
  return false;
}
bool ParseTemplateArg(State* state) {
  constexpr uint32 max_levels = 6;
  if (state->arg_level > max_levels) {
    return false;
  }
  ++state->arg_level;
  State copy = *state;
  if ((ParseOneCharToken(state, 'I') || ParseOneCharToken(state, 'J')) &&
      ZeroOrMore(ParseTemplateArg, state) && ParseOneCharToken(state, 'E')) {
    --state->arg_level;
    return true;
  }
  *state = copy;
  if (ParseType(state) || ParseExprPrimary(state)) {
    --state->arg_level;
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'X') && ParseExpression(state) &&
      ParseOneCharToken(state, 'E')) {
    --state->arg_level;
    return true;
  }
  *state = copy;
  return false;
}
bool ParseExpression(State* state) {
  if (ParseTemplateParam(state) || ParseExprPrimary(state)) {
    return true;
  }
  constexpr uint32 max_levels = 5;
  if (state->expr_level > max_levels) {
    return false;
  }
  ++state->expr_level;
  State copy = *state;
  if (ParseOperatorName(state) && ParseExpression(state) &&
      ParseExpression(state) && ParseExpression(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  if (ParseOperatorName(state) && ParseExpression(state) &&
      ParseExpression(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  if (ParseOperatorName(state) && ParseExpression(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "st") && ParseType(state)) {
    return true;
    --state->expr_level;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "sr") && ParseType(state) &&
      ParseUnqualifiedName(state) && ParseTemplateArgs(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "sr") && ParseType(state) &&
      ParseUnqualifiedName(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "sp") && ParseType(state)) {
    --state->expr_level;
    return true;
  }
  *state = copy;
  return false;
}
bool ParseExprPrimary(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, 'L') && ParseType(state) &&
      ParseNumber(state, nullptr) && ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'L') && ParseType(state) &&
      ParseFloatNumber(state) && ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'L') && ParseMangledName(state) &&
      ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  if (ParseTwoCharToken(state, "LZ") && ParseEncoding(state) &&
      ParseOneCharToken(state, 'E')) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseLocalName(State* state) {
  constexpr uint32 max_levels = 5;
  if (state->local_level > max_levels) {
    return false;
  }
  ++state->local_level;
  State copy = *state;
  if (ParseOneCharToken(state, 'Z') && ParseEncoding(state) &&
      ParseOneCharToken(state, 'E') && MaybeAppend(state, "::") &&
      ParseName(state) && Optional(ParseDiscriminator(state))) {
    --state->local_level;
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'Z') && ParseEncoding(state) &&
      ParseTwoCharToken(state, "Es") && Optional(ParseDiscriminator(state))) {
    --state->local_level;
    return true;
  }
  *state = copy;
  return false;
}
bool ParseDiscriminator(State* state) {
  State copy = *state;
  if (ParseOneCharToken(state, '_') && ParseNumber(state, nullptr)) {
    return true;
  }
  *state = copy;
  return false;
}
bool ParseSubstitution(State* state) {
  if (ParseTwoCharToken(state, "S_")) {
    MaybeAppend(state, "?");  
    return true;
  }
  State copy = *state;
  if (ParseOneCharToken(state, 'S') && ParseSeqId(state) &&
      ParseOneCharToken(state, '_')) {
    MaybeAppend(state, "?");  
    return true;
  }
  *state = copy;
  if (ParseOneCharToken(state, 'S')) {
    const AbbrevPair* p;
    for (p = kSubstitutionList; p->abbrev != nullptr; ++p) {
      if (state->mangled_cur[0] == p->abbrev[1]) {
        MaybeAppend(state, "std");
        if (p->real_name[0] != '\0') {
          MaybeAppend(state, "::");
          MaybeAppend(state, p->real_name);
        }
        ++state->mangled_cur;
        return true;
      }
    }
  }
  *state = copy;
  return false;
}
bool ParseTopLevelMangledName(State* state) {
  if (ParseMangledName(state)) {
    if (state->mangled_cur[0] != '\0') {
      if (IsFunctionCloneSuffix(state->mangled_cur)) {
        return true;
      }
      if (state->mangled_cur[0] == '@') {
        MaybeAppend(state, state->mangled_cur);
        return true;
      }
      return ParseName(state);
    }
    return true;
  }
  return false;
}
}  
#endif
bool Demangle(const char* mangled, char* out, size_t out_size) {
#if defined(GLOG_OS_WINDOWS)
#  if defined(HAVE_DBGHELP)
  char buffer[1024];  
  const char* lparen = strchr(mangled, '(');
  if (lparen) {
    const char* rparen = strchr(lparen, ')');
    size_t length = static_cast<size_t>(rparen - lparen) - 1;
    strncpy(buffer, lparen + 1, length);
    buffer[length] = '\0';
    mangled = buffer;
  }  
  return UnDecorateSymbolName(mangled, out, out_size, UNDNAME_COMPLETE);
#  else
  (void)mangled;
  (void)out;
  (void)out_size;
  return false;
#  endif
#elif defined(HAVE___CXA_DEMANGLE)
  int status = -1;
  std::size_t n = 0;
  std::unique_ptr<char, decltype(&std::free)> unmangled{
      abi::__cxa_demangle(mangled, nullptr, &n, &status), &std::free};
  if (!unmangled) {
    return false;
  }
  std::copy_n(unmangled.get(), std::min(n, out_size), out);
  return status == 0;
#else
  State state;
  InitState(&state, mangled, out, out_size);
  return ParseTopLevelMangledName(&state) && !state.overflowed;
#endif
}
}  
}  