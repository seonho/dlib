#ifndef DLIB_CONCURRENCY_Hh_
#define DLIB_CONCURRENCY_H

#ifndef _TBB

#ifdef _MSC_VER
#include <ppl.h>

namespace concurrency_compat = concurrency;
#endif // _MSC_VER

#else

//#include <tbb/compat/ppl.h>
#include <tbb/tbb.h>

namespace concurrency_compat = tbb;

#endif // _TBB

#endif // DLIB_CONCURRENCY_H