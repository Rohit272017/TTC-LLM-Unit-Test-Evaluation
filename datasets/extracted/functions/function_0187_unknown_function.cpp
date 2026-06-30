#ifndef TENSORFLOW_TSL_PLATFORM_CTSTRING_H_
#define TENSORFLOW_TSL_PLATFORM_CTSTRING_H_
#include <stdint.h>
#include <stdlib.h>
#include "tsl/platform/ctstring_internal.h"
inline void TF_TString_Init(TF_TString *str);
inline void TF_TString_Dealloc(TF_TString *str);
inline char *TF_TString_Resize(TF_TString *str, size_t new_size, char c);
inline char *TF_TString_ResizeUninitialized(TF_TString *str, size_t new_size);
inline void TF_TString_Reserve(TF_TString *str, size_t new_cap);
inline void TF_TString_ReserveAmortized(TF_TString *str, size_t new_cap);
inline size_t TF_TString_GetSize(const TF_TString *str);
inline size_t TF_TString_GetCapacity(const TF_TString *str);
inline TF_TString_Type TF_TString_GetType(const TF_TString *str);
inline const char *TF_TString_GetDataPointer(const TF_TString *str);
inline char *TF_TString_GetMutableDataPointer(TF_TString *str);
inline void TF_TString_AssignView(TF_TString *dst, const char *src,
                                  size_t size);
inline void TF_TString_Append(TF_TString *dst, const TF_TString *src);
inline void TF_TString_AppendN(TF_TString *dst, const char *src, size_t size);
inline void TF_TString_Copy(TF_TString *dst, const char *src, size_t size);
inline void TF_TString_Assign(TF_TString *dst, const TF_TString *src);
inline void TF_TString_Move(TF_TString *dst, TF_TString *src);
#endif  