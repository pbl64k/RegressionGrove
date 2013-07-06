
#ifndef INCLUDE__REGRESSION_GROVE_HXX

#define INCLUDE__REGRESSION_GROVE_HXX

#include <iostream>
#include <istream>
#include <ostream>

#include <cstdlib>

#include <iterator>
#include <vector>

#include "util.hxx"

#include "regression_tree.hxx"

template<typename T>
inline T SAMPLE_VARIABLES(T v)
{
	return v / 3;
}

template<typename T, typename Y, typename X, typename L>
class regression_grove
{
	private:
	bool trained_;
	size_t dim_n_, dim_v_;
	std::vector<regression_tree<T, Y, X, std::vector<size_t>, L> > grove_;

	public:
	regression_grove(const Y &y, const X &x, const L &l, size_t n, std::ostream &log):
			trained_(false), dim_n_(y.size()), dim_v_(x.size() / dim_n_), grove_()
	{
		train(y, x, l, n, log);
	}

	regression_grove(std::istream &in):
			trained_(true), grove_()
	{
		expect(in, "regression_grove");
		expect(in, "{");

		in >> dim_n_;
		in >> dim_v_;

		size_t grove_size;

		in >> grove_size;

		expect(in, "{");

		for (size_t i = 0; i != grove_size; ++i)
		{
			grove_.push_back(regression_tree<T, Y, X, std::vector<size_t>, L>(in));

#ifdef LOG_OUTPUT_RG_TICK
			std::cerr << "#";
#endif
		}

#ifdef LOG_OUTPUT_RG_TICK
		std::cerr << std::endl;
#endif

		expect(in, "}");

		expect(in, "}");
	}

	void train(const Y &y, const X &x, const L &l, size_t n, std::ostream &log)
	{
#ifndef NDEBUG
		FLEX_ASSERT(! trained_);
#endif

#ifdef LOG_OUTPUT
		log << "Training regression grove of " << n << " trees..." << std::endl;
#endif

		std::vector<size_t> ix(dim_n_);

		for (size_t i = 0; i != n; ++i)
		{
#ifdef LOG_OUTPUT
			log << "Training tree #" << (i + 1) << "..." << std::endl;
#endif

			for (size_t j = 0; j != dim_n_; ++j)
			{
				ix[j] = floor((static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX)) * dim_n_);
			}

			grove_.push_back(regression_tree<T, Y, X, std::vector<size_t>, L>(
					y, x, ix, l, SAMPLE_VARIABLES(dim_v_), log));

#ifdef LOG_OUTPUT
			log << "Done training tree #" << (i + 1) << "." << std::endl;
#endif

#ifdef LOG_OUTPUT_RG_TICK
			std::cerr << "Done training tree #" << (i + 1) << "." << std::endl;
#endif
		}

#ifdef LOG_OUTPUT
		log << "Training complete." << std::endl;
#endif

		trained_ = true;
	}

	std::vector<T> predict(const X &x, size_t i = 0) const
	{
#ifndef NDEBUG
		FLEX_ASSERT(trained_);
#endif

		std::vector<T> predictions;

		for (typename std::vector<regression_tree<T, Y, X, std::vector<size_t>, L> >::const_iterator
				grove_iter = grove_.begin(); grove_iter != grove_.end(); ++grove_iter)
		{
			predictions.push_back(grove_iter->predict(x, i));
		}

		return predictions;
	}

	void serialize(std::ostream &out) const
	{
#ifndef NDEBUG
		FLEX_ASSERT(trained_);
#endif

		out << "regression_grove" << std::endl;
		out << "{" << std::endl;

		out << dim_n_ << " " << dim_v_ << std::endl;

		out << grove_.size() << std::endl;
		out << "{" << std::endl;

		for (typename std::vector<regression_tree<T, Y, X, std::vector<size_t>, L> >::const_iterator
				grove_iter = grove_.begin(); grove_iter != grove_.end(); ++grove_iter)
		{
			grove_iter->serialize(out);

#ifdef LOG_OUTPUT_RG_TICK
			std::cerr << "#";
#endif
		}

#ifdef LOG_OUTPUT_RG_TICK
		std::cerr << std::endl;
#endif

		out << "}" << std::endl;

		out << "}" << std::endl;
	}

	const size_t &v() const
	{
		return dim_v_;
	}
};

#endif

