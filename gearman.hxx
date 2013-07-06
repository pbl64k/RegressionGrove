
#ifndef INCLUDE__GEARMAN_HXX

#define INCLUDE__GEARMAN_HXX

#include <libgearman/gearman.h>

#define GEARMAN(x) { \
	gearman_return_t status = x; \
	FLEX_ASSERT(status == GEARMAN_SUCCESS); \
}

#endif

