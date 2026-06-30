#ifndef LANGSVR_LSP_COMPARATORS_H_
#define LANGSVR_LSP_COMPARATORS_H_
#include "langsvr/lsp/lsp.h"
namespace langsvr::lsp {
inline int Compare(Position a, Position b) {
    if (a.line < b.line) {
        return -1;
    }
    if (a.line > b.line) {
        return 1;
    }
    if (a.character < b.character) {
        return -1;
    }
    if (a.character > b.character) {
        return 1;
    }
    return 0;
}
inline bool operator<(Position a, Position b) {
    return Compare(a, b) < 0;
}
inline bool operator<=(Position a, Position b) {
    return Compare(a, b) <= 0;
}
inline bool operator>(Position a, Position b) {
    return Compare(a, b) > 0;
}
inline bool operator>=(Position a, Position b) {
    return Compare(a, b) >= 0;
}
inline bool ContainsExclusive(Range r, Position p) {
    return p >= r.start && p < r.end;
}
inline bool ContainsInclusive(Range r, Position p) {
    return p >= r.start && p <= r.end;
}
}  
#endif  