#ifndef TENSORSTORE_INTERNAL_GLOBAL_INITIALIZER_H_
#define TENSORSTORE_INTERNAL_GLOBAL_INITIALIZER_H_
#include "tensorstore/internal/preprocessor/cat.h"
#define TENSORSTORE_GLOBAL_INITIALIZER                            \
  namespace {                                                     \
  const struct TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__) {       \
    TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__)                    \
    ();                                                           \
  } TENSORSTORE_PP_CAT(tensorstore_global_init, __LINE__);        \
  }                                                               \
  TENSORSTORE_PP_CAT(TsGlobalInit, __LINE__)::TENSORSTORE_PP_CAT( \
      TsGlobalInit, __LINE__)() 
#endif  