
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>

#include <cstdlib>
#include <ctime>

#include <string>
#include <vector>

#include "rg.hxx"

#include "util.hxx"

#include "regression_grove.hxx"

int main(int argc, char **argv)
{
	std::srand(std::time(NULL));

#ifndef NDEBUG
	FLEX_ASSERT(argc == 2);
#endif

	std::string tree_num_str(argv[1]);

	std::stringstream tree_num_stream(tree_num_str);

	size_t tree_num;

	tree_num_stream >> tree_num;

#ifndef NDEBUG
	FLEX_ASSERT(tree_num > 1);
#endif

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Reading training data..." << std::endl;
#endif

	size_t n, v;

	std::cin >> v;
	std::cin >> n;

	std::vector<fp_type> y(n);
	std::vector<fp_type> x(n * (v - 1));

	for (size_t i = 0; i != n; ++i)
	{
		std::cin >> y[i];

		for (size_t j = 0; j != (v - 1); ++j)
		{
			std::cin >> x[IX(i, j, v - 1)];
		}
	}

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Done." << std::endl;
#endif

	// TODO: levels should be generated by the data provider
	std::vector<size_t> l(v - 1);

	for (size_t i = 0; i != (v - 1); ++i)
	{
		// Fixed variable layout for my dataset. Four first continuous, the rest are binary.
		l[i] = (i < 4) ? 0 : 2;
	}

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Training regression grove..." << std::endl;
#endif

	regression_grove<fp_type, std::vector<fp_type>, std::vector<fp_type>, std::vector<size_t> >
			rg(y, x, l, tree_num, std::cerr);

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Done." << std::endl;
#endif

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Serializing model..." << std::endl;
#endif

	rg.serialize(std::cout);

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Done." << std::endl;
#endif

	return 0;
}

