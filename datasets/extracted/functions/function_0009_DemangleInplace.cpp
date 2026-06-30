#ifdef GLOG_BUILD_CONFIG_INCLUDE
#  include GLOG_BUILD_CONFIG_INCLUDE
#endif  
#include "symbolize.h"
#include "utilities.h"
#if defined(HAVE_SYMBOLIZE)
#  include <algorithm>
#  include <cstdlib>
#  include <cstring>
#  include <limits>
#  include "demangle.h"
#  define GLOG_SAFE_ASSERT(expr) ((expr) ? 0 : (std::abort(), 0))
namespace google {
inline namespace glog_internal_namespace_ {
namespace {
SymbolizeCallback g_symbolize_callback = nullptr;
SymbolizeOpenObjectFileCallback g_symbolize_open_object_file_callback = nullptr;
ATTRIBUTE_NOINLINE
void DemangleInplace(char* out, size_t out_size) {
  char demangled[256];  
  if (Demangle(out, demangled, sizeof(demangled))) {
    size_t len = strlen(demangled);
    if (len + 1 <= out_size) {  
      GLOG_SAFE_ASSERT(len < sizeof(demangled));
      memmove(out, demangled, len + 1);
    }
  }
}
}  
void InstallSymbolizeCallback(SymbolizeCallback callback) {
  g_symbolize_callback = callback;
}
void InstallSymbolizeOpenObjectFileCallback(
    SymbolizeOpenObjectFileCallback callback) {
  g_symbolize_open_object_file_callback = callback;
}
}  
}  
#  if defined(HAVE_LINK_H)
#    if defined(HAVE_DLFCN_H)
#      include <dlfcn.h>
#    endif
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#    include <cerrno>
#    include <climits>
#    include <cstddef>
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include "config.h"
#    include "glog/raw_logging.h"
#    include "symbolize.h"
namespace google {
inline namespace glog_internal_namespace_ {
namespace {
template <class Functor>
auto FailureRetry(Functor run, int error = EINTR) noexcept(noexcept(run())) {
  decltype(run()) result;
  while ((result = run()) == -1 && errno == error) {
  }
  return result;
}
}  
static ssize_t ReadFromOffset(const int fd, void* buf, const size_t count,
                              const size_t offset) {
  GLOG_SAFE_ASSERT(fd >= 0);
  GLOG_SAFE_ASSERT(count <=
                   static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
  char* buf0 = reinterpret_cast<char*>(buf);
  size_t num_bytes = 0;
  while (num_bytes < count) {
    ssize_t len = FailureRetry([fd, p = buf0 + num_bytes, n = count - num_bytes,
                                m = static_cast<off_t>(offset + num_bytes)] {
      return pread(fd, p, n, m);
    });
    if (len < 0) {  
      return -1;
    }
    if (len == 0) {  
      break;
    }
    num_bytes += static_cast<size_t>(len);
  }
  GLOG_SAFE_ASSERT(num_bytes <= count);
  return static_cast<ssize_t>(num_bytes);
}
static bool ReadFromOffsetExact(const int fd, void* buf, const size_t count,
                                const size_t offset) {
  ssize_t len = ReadFromOffset(fd, buf, count, offset);
  return static_cast<size_t>(len) == count;
}
static int FileGetElfType(const int fd) {
  ElfW(Ehdr) elf_header;
  if (!ReadFromOffsetExact(fd, &elf_header, sizeof(elf_header), 0)) {
    return -1;
  }
  if (memcmp(elf_header.e_ident, ELFMAG, SELFMAG) != 0) {
    return -1;
  }
  return elf_header.e_type;
}
static ATTRIBUTE_NOINLINE bool GetSectionHeaderByType(const int fd,
                                                      ElfW(Half) sh_num,
                                                      const size_t sh_offset,
                                                      ElfW(Word) type,
                                                      ElfW(Shdr) * out) {
  ElfW(Shdr) buf[16];
  for (size_t i = 0; i < sh_num;) {
    const size_t num_bytes_left = (sh_num - i) * sizeof(buf[0]);
    const size_t num_bytes_to_read =
        (sizeof(buf) > num_bytes_left) ? num_bytes_left : sizeof(buf);
    const ssize_t len = ReadFromOffset(fd, buf, num_bytes_to_read,
                                       sh_offset + i * sizeof(buf[0]));
    if (len == -1) {
      return false;
    }
    GLOG_SAFE_ASSERT(static_cast<size_t>(len) % sizeof(buf[0]) == 0);
    const size_t num_headers_in_buf = static_cast<size_t>(len) / sizeof(buf[0]);
    GLOG_SAFE_ASSERT(num_headers_in_buf <= sizeof(buf) / sizeof(buf[0]));
    for (size_t j = 0; j < num_headers_in_buf; ++j) {
      if (buf[j].sh_type == type) {
        *out = buf[j];
        return true;
      }
    }
    i += num_headers_in_buf;
  }
  return false;
}
const int kMaxSectionNameLen = 64;
bool GetSectionHeaderByName(int fd, const char* name, size_t name_len,
                            ElfW(Shdr) * out) {
  ElfW(Ehdr) elf_header;
  if (!ReadFromOffsetExact(fd, &elf_header, sizeof(elf_header), 0)) {
    return false;
  }
  ElfW(Shdr) shstrtab;
  size_t shstrtab_offset =
      (elf_header.e_shoff + static_cast<size_t>(elf_header.e_shentsize) *
                                static_cast<size_t>(elf_header.e_shstrndx));
  if (!ReadFromOffsetExact(fd, &shstrtab, sizeof(shstrtab), shstrtab_offset)) {
    return false;
  }
  for (size_t i = 0; i < elf_header.e_shnum; ++i) {
    size_t section_header_offset =
        (elf_header.e_shoff + elf_header.e_shentsize * i);
    if (!ReadFromOffsetExact(fd, out, sizeof(*out), section_header_offset)) {
      return false;
    }
    char header_name[kMaxSectionNameLen];
    if (sizeof(header_name) < name_len) {
      RAW_LOG(WARNING,
              "Section name '%s' is too long (%zu); "
              "section will not be found (even if present).",
              name, name_len);
      return false;
    }
    size_t name_offset = shstrtab.sh_offset + out->sh_name;
    ssize_t n_read = ReadFromOffset(fd, &header_name, name_len, name_offset);
    if (n_read == -1) {
      return false;
    } else if (static_cast<size_t>(n_read) != name_len) {
      continue;
    }
    if (memcmp(header_name, name, name_len) == 0) {
      return true;
    }
  }
  return false;
}
static ATTRIBUTE_NOINLINE bool FindSymbol(uint64_t pc, const int fd, char* out,
                                          size_t out_size,
                                          uint64_t symbol_offset,
                                          const ElfW(Shdr) * strtab,
                                          const ElfW(Shdr) * symtab) {
  if (symtab == nullptr) {
    return false;
  }
  const size_t num_symbols = symtab->sh_size / symtab->sh_entsize;
  for (unsigned i = 0; i < num_symbols;) {
    size_t offset = symtab->sh_offset + i * symtab->sh_entsize;
#    if defined(__WORDSIZE) && __WORDSIZE == 64
    const size_t NUM_SYMBOLS = 32U;
#    else
    const size_t NUM_SYMBOLS = 64U;
#    endif
    ElfW(Sym) buf[NUM_SYMBOLS];
    size_t num_symbols_to_read = std::min(NUM_SYMBOLS, num_symbols - i);
    const ssize_t len =
        ReadFromOffset(fd, &buf, sizeof(buf[0]) * num_symbols_to_read, offset);
    GLOG_SAFE_ASSERT(static_cast<size_t>(len) % sizeof(buf[0]) == 0);
    const size_t num_symbols_in_buf = static_cast<size_t>(len) / sizeof(buf[0]);
    GLOG_SAFE_ASSERT(num_symbols_in_buf <= num_symbols_to_read);
    for (unsigned j = 0; j < num_symbols_in_buf; ++j) {
      const ElfW(Sym)& symbol = buf[j];
      uint64_t start_address = symbol.st_value;
      start_address += symbol_offset;
      uint64_t end_address = start_address + symbol.st_size;
      if (symbol.st_value != 0 &&  
          symbol.st_shndx != 0 &&  
          start_address <= pc && pc < end_address) {
        ssize_t len1 = ReadFromOffset(fd, out, out_size,
                                      strtab->sh_offset + symbol.st_name);
        if (len1 <= 0 || memchr(out, '\0', out_size) == nullptr) {
          memset(out, 0, out_size);
          return false;
        }
        return true;  
      }
    }
    i += num_symbols_in_buf;
  }
  return false;
}
static bool GetSymbolFromObjectFile(const int fd, uint64_t pc, char* out,
                                    size_t out_size, uint64_t base_address) {
  ElfW(Ehdr) elf_header;
  if (!ReadFromOffsetExact(fd, &elf_header, sizeof(elf_header), 0)) {
    return false;
  }
  ElfW(Shdr) symtab, strtab;
  if (GetSectionHeaderByType(fd, elf_header.e_shnum, elf_header.e_shoff,
                             SHT_SYMTAB, &symtab)) {
    if (!ReadFromOffsetExact(
            fd, &strtab, sizeof(strtab),
            elf_header.e_shoff + symtab.sh_link * sizeof(symtab))) {
      return false;
    }
    if (FindSymbol(pc, fd, out, out_size, base_address, &strtab, &symtab)) {
      return true;  
    }
  }
  if (GetSectionHeaderByType(fd, elf_header.e_shnum, elf_header.e_shoff,
                             SHT_DYNSYM, &symtab)) {
    if (!ReadFromOffsetExact(
            fd, &strtab, sizeof(strtab),
            elf_header.e_shoff + symtab.sh_link * sizeof(symtab))) {
      return false;
    }
    if (FindSymbol(pc, fd, out, out_size, base_address, &strtab, &symtab)) {
      return true;  
    }
  }
  return false;
}
namespace {
class LineReader {
 public:
  explicit LineReader(int fd, char* buf, size_t buf_len, size_t offset)
      : fd_(fd),
        buf_(buf),
        buf_len_(buf_len),
        offset_(offset),
        bol_(buf),
        eol_(buf),
        eod_(buf) {}
  bool ReadLine(const char** bol, const char** eol) {
    if (BufferIsEmpty()) {  
      const ssize_t num_bytes = ReadFromOffset(fd_, buf_, buf_len_, offset_);
      if (num_bytes <= 0) {  
        return false;
      }
      offset_ += static_cast<size_t>(num_bytes);
      eod_ = buf_ + num_bytes;
      bol_ = buf_;
    } else {
      bol_ = eol_ + 1;  
      GLOG_SAFE_ASSERT(bol_ <= eod_);  
      if (!HasCompleteLine()) {
        const auto incomplete_line_length = static_cast<size_t>(eod_ - bol_);
        memmove(buf_, bol_, incomplete_line_length);
        char* const append_pos = buf_ + incomplete_line_length;
        const size_t capacity_left = buf_len_ - incomplete_line_length;
        const ssize_t num_bytes =
            ReadFromOffset(fd_, append_pos, capacity_left, offset_);
        if (num_bytes <= 0) {  
          return false;
        }
        offset_ += static_cast<size_t>(num_bytes);
        eod_ = append_pos + num_bytes;
        bol_ = buf_;
      }
    }
    eol_ = FindLineFeed();
    if (eol_ == nullptr) {  
      return false;
    }
    *eol_ = '\0';  
    *bol = bol_;
    *eol = eol_;
    return true;
  }
  const char* bol() { return bol_; }
  const char* eol() { return eol_; }
 private:
  LineReader(const LineReader&) = delete;
  void operator=(const LineReader&) = delete;
  char* FindLineFeed() {
    return reinterpret_cast<char*>(
        memchr(bol_, '\n', static_cast<size_t>(eod_ - bol_)));
  }
  bool BufferIsEmpty() { return buf_ == eod_; }
  bool HasCompleteLine() {
    return !BufferIsEmpty() && FindLineFeed() != nullptr;
  }
  const int fd_;
  char* const buf_;
  const size_t buf_len_;
  size_t offset_;
  char* bol_;
  char* eol_;
  const char* eod_;  
};
}  
static char* GetHex(const char* start, const char* end, uint64_t* hex) {
  *hex = 0;
  const char* p;
  for (p = start; p < end; ++p) {
    int ch = *p;
    if ((ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') ||
        (ch >= 'a' && ch <= 'f')) {
      *hex = (*hex << 4U) |
             (ch < 'A' ? static_cast<uint64_t>(ch - '0') : (ch & 0xF) + 9U);
    } else {  
      break;
    }
  }
  GLOG_SAFE_ASSERT(p <= end);
  return const_cast<char*>(p);
}
static ATTRIBUTE_NOINLINE FileDescriptor
OpenObjectFileContainingPcAndGetStartAddress(uint64_t pc,
                                             uint64_t& start_address,
                                             uint64_t& base_address,
                                             char* out_file_name,
                                             size_t out_file_name_size) {
  FileDescriptor maps_fd{
      FailureRetry([] { return open("/proc/self/maps", O_RDONLY); })};
  if (!maps_fd) {
    return nullptr;
  }
  FileDescriptor mem_fd{
      FailureRetry([] { return open("/proc/self/mem", O_RDONLY); })};
  if (!mem_fd) {
    return nullptr;
  }
  char buf[1024];  
  LineReader reader(maps_fd.get(), buf, sizeof(buf), 0);
  while (true) {
    const char* cursor;
    const char* eol;
    if (!reader.ReadLine(&cursor, &eol)) {  
      return nullptr;
    }
    cursor = GetHex(cursor, eol, &start_address);
    if (cursor == eol || *cursor != '-') {
      return nullptr;  
    }
    ++cursor;  
    uint64_t end_address;
    cursor = GetHex(cursor, eol, &end_address);
    if (cursor == eol || *cursor != ' ') {
      return nullptr;  
    }
    ++cursor;  
    const char* const flags_start = cursor;
    while (cursor < eol && *cursor != ' ') {
      ++cursor;
    }
    if (cursor == eol || cursor < flags_start + 4) {
      return nullptr;  
    }
    ElfW(Ehdr) ehdr;
    if (flags_start[0] == 'r' &&
        ReadFromOffsetExact(mem_fd.get(), &ehdr, sizeof(ElfW(Ehdr)),
                            start_address) &&
        memcmp(ehdr.e_ident, ELFMAG, SELFMAG) == 0) {
      switch (ehdr.e_type) {
        case ET_EXEC:
          base_address = 0;
          break;
        case ET_DYN:
          base_address = start_address;
          for (unsigned i = 0; i != ehdr.e_phnum; ++i) {
            ElfW(Phdr) phdr;
            if (ReadFromOffsetExact(
                    mem_fd.get(), &phdr, sizeof(phdr),
                    start_address + ehdr.e_phoff + i * sizeof(phdr)) &&
                phdr.p_type == PT_LOAD && phdr.p_offset == 0) {
              base_address = start_address - phdr.p_vaddr;
              break;
            }
          }
          break;
        default:
          break;
      }
    }
    if (start_address > pc || pc >= end_address) {
      continue;  
    }
    if (flags_start[0] != 'r' || flags_start[2] != 'x') {
      continue;  
    }
    ++cursor;  
    uint64_t file_offset;
    cursor = GetHex(cursor, eol, &file_offset);
    if (cursor == eol || *cursor != ' ') {
      return nullptr;  
    }
    ++cursor;  
    int num_spaces = 0;
    while (cursor < eol) {
      if (*cursor == ' ') {
        ++num_spaces;
      } else if (num_spaces >= 2) {
        break;
      }
      ++cursor;
    }
    if (cursor == eol) {
      return nullptr;  
    }
    strncpy(out_file_name, cursor, out_file_name_size);
    out_file_name[out_file_name_size - 1] = '\0';
    return FileDescriptor{
        FailureRetry([cursor] { return open(cursor, O_RDONLY); })};
  }
}
static char* itoa_r(uintptr_t i, char* buf, size_t sz, unsigned base,
                    size_t padding) {
  size_t n = 1;
  if (n > sz) {
    return nullptr;
  }
  if (base < 2 || base > 16) {
    buf[0] = '\000';
    return nullptr;
  }
  char* start = buf;
  char* ptr = start;
  do {
    if (++n > sz) {
      buf[0] = '\000';
      return nullptr;
    }
    *ptr++ = "0123456789abcdef"[i % base];
    i /= base;
    if (padding > 0) {
      padding--;
    }
  } while (i > 0 || padding > 0);
  *ptr = '\000';
  while (--ptr > start) {
    char ch = *ptr;
    *ptr = *start;
    *start++ = ch;
  }
  return buf;
}
static void SafeAppendString(const char* source, char* dest, size_t dest_size) {
  size_t dest_string_length = strlen(dest);
  GLOG_SAFE_ASSERT(dest_string_length < dest_size);
  dest += dest_string_length;
  dest_size -= dest_string_length;
  strncpy(dest, source, dest_size);
  dest[dest_size - 1] = '\0';
}
static void SafeAppendHexNumber(uint64_t value, char* dest, size_t dest_size) {
  char buf[17] = {'\0'};
  SafeAppendString(itoa_r(value, buf, sizeof(buf), 16, 0), dest, dest_size);
}
static ATTRIBUTE_NOINLINE bool SymbolizeAndDemangle(
    void* pc, char* out, size_t out_size, SymbolizeOptions ) {
  auto pc0 = reinterpret_cast<uintptr_t>(pc);
  uint64_t start_address = 0;
  uint64_t base_address = 0;
  FileDescriptor object_fd;
  if (out_size < 1) {
    return false;
  }
  out[0] = '\0';
  SafeAppendString("(", out, out_size);
  if (g_symbolize_open_object_file_callback) {
    object_fd.reset(g_symbolize_open_object_file_callback(
        pc0, start_address, base_address, out + 1, out_size - 1));
  } else {
    object_fd = OpenObjectFileContainingPcAndGetStartAddress(
        pc0, start_address, base_address, out + 1, out_size - 1);
  }
#    if defined(PRINT_UNSYMBOLIZED_STACK_TRACES)
  {
#    else
  if (!object_fd) {
#    endif
    if (out[1]) {
      out[out_size - 1] = '\0';  
      SafeAppendString("+0x", out, out_size);
      SafeAppendHexNumber(pc0 - base_address, out, out_size);
      SafeAppendString(")", out, out_size);
      return true;
    }
    return false;
  }
  int elf_type = FileGetElfType(object_fd.get());
  if (elf_type == -1) {
    return false;
  }
  if (g_symbolize_callback) {
    uint64_t relocation = (elf_type == ET_DYN) ? start_address : 0;
    int num_bytes_written =
        g_symbolize_callback(object_fd.get(), pc, out, out_size, relocation);
    if (num_bytes_written > 0) {
      out += static_cast<size_t>(num_bytes_written);
      out_size -= static_cast<size_t>(num_bytes_written);
    }
  }
  if (!GetSymbolFromObjectFile(object_fd.get(), pc0, out, out_size,
                               base_address)) {
    if (out[1] && !g_symbolize_callback) {
      out[out_size - 1] = '\0';  
      SafeAppendString("+0x", out, out_size);
      SafeAppendHexNumber(pc0 - base_address, out, out_size);
      SafeAppendString(")", out, out_size);
      return true;
    }
    return false;
  }
  DemangleInplace(out, out_size);
  return true;
}
}  
}  
#  elif defined(GLOG_OS_MACOSX) && defined(HAVE_DLADDR)
#    include <dlfcn.h>
#    include <cstring>
namespace google {
inline namespace glog_internal_namespace_ {
static ATTRIBUTE_NOINLINE bool SymbolizeAndDemangle(
    void* pc, char* out, size_t out_size, SymbolizeOptions ) {
  Dl_info info;
  if (dladdr(pc, &info)) {
    if (info.dli_sname) {
      if (strlen(info.dli_sname) < out_size) {
        strcpy(out, info.dli_sname);
        DemangleInplace(out, out_size);
        return true;
      }
    }
  }
  return false;
}
}  
}  
#  elif defined(GLOG_OS_WINDOWS) || defined(GLOG_OS_CYGWIN)
#    include <dbghelp.h>
#    include <windows.h>
namespace google {
inline namespace glog_internal_namespace_ {
namespace {
class SymInitializer final {
 public:
  HANDLE process;
  bool ready;
  SymInitializer() : process(GetCurrentProcess()), ready(false) {
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (SymInitialize(process, nullptr, true)) {
      ready = true;
    }
  }
  ~SymInitializer() {
    SymCleanup(process);
  }
  SymInitializer(const SymInitializer&) = delete;
  SymInitializer& operator=(const SymInitializer&) = delete;
  SymInitializer(SymInitializer&&) = delete;
  SymInitializer& operator=(SymInitializer&&) = delete;
};
}  
static ATTRIBUTE_NOINLINE bool SymbolizeAndDemangle(void* pc, char* out,
                                                    size_t out_size,
                                                    SymbolizeOptions options) {
  const static SymInitializer symInitializer;
  if (!symInitializer.ready) {
    return false;
  }
  char buf[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
  SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(buf);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen = MAX_SYM_NAME;
  BOOL ret = SymFromAddr(symInitializer.process, reinterpret_cast<DWORD64>(pc),
                         0, symbol);
  std::size_t namelen = static_cast<size_t>(symbol->NameLen);
  if (ret && namelen < out_size) {
    std::strncpy(out, symbol->Name, namelen);
    out[namelen] = '\0';
    DWORD displacement;
    IMAGEHLP_LINE64 line{sizeof(IMAGEHLP_LINE64)};
    BOOL found = FALSE;
    if ((options & SymbolizeOptions::kNoLineNumbers) !=
        SymbolizeOptions::kNoLineNumbers) {
      found = SymGetLineFromAddr64(symInitializer.process,
                                   reinterpret_cast<DWORD64>(pc), &displacement,
                                   &line);
    }
    DemangleInplace(out, out_size);
    out_size -= std::strlen(out);
    if (found) {
      std::size_t fnlen = std::strlen(line.FileName);
      std::size_t digits = 1;  
      for (DWORD value = line.LineNumber; (value /= 10) != 0; ++digits) {
      }
      constexpr std::size_t extralen = 4;  
      const std::size_t suffixlen = fnlen + extralen + fnlen + digits;
      if (suffixlen < out_size) {
        out_size -= std::snprintf(out + namelen, out_size, " (%s:%lu)",
                                  line.FileName, line.LineNumber);
      }
    }
    return true;
  }
  return false;
}
}  
}  
#  else
#    error BUG: HAVE_SYMBOLIZE was wrongly set
#  endif
namespace google {
inline namespace glog_internal_namespace_ {
bool Symbolize(void* pc, char* out, size_t out_size, SymbolizeOptions options) {
  return SymbolizeAndDemangle(pc, out, out_size, options);
}
}  
}  
#endif