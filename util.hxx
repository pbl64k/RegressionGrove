
#ifndef INCLUDE__UTIL_HXX

#define INCLUDE__UTIL_HXX

#include <iostream>
#include <istream>
#include <ostream>

#include <cassert>

#include <exception>
#include <string>

#ifdef FATAL_ASSERTION_FAILURES
#define FLEX_ASSERT(x) assert(x);
#else
#define FLEX_ASSERT(x) throw std::exception();
#endif

template<typename T>
inline T IX(T i, T j, T w)
{
	return (i * w) + j;
}

inline void expect(std::istream &in, const std::string &str)
{
	std::string tag;

	in >> tag;

#ifndef NDEBUG
	FLEX_ASSERT(tag == str);
#endif
}

#endif

