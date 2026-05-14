#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(PGO_C_BUILDING_LIBRARY)
#        define PGO_C_API __declspec(dllexport)
#    else
#        define PGO_C_API __declspec(dllimport)
#    endif
#else
#    define PGO_C_API __attribute__((visibility("default")))
#endif
