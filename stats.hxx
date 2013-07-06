
#ifndef INCLUDE__STATS_HXX

#define INCLUDE__STATS_HXX

#include <cmath>

#include <algorithm>
#include <iterator>
#include <utility>

#include "util.hxx"

static const fp_type BINARY_THRESHOLD = 0.5;

template<typename T, typename Y, typename I>
inline T mu(const Y &y, const I &i)
{
	T acc = T();

	for (typename I::const_iterator i_iter = i.begin(); i_iter != i.end(); ++i_iter)
	{
		acc += y[*i_iter];
	}

	return acc / i.size();
}

template<typename T, typename Y, typename I>
inline T sigma(const Y &y, const I &i)
{
	T acc = T();

	if (i.size() == 0)
	{
		return acc;
	}

	T mean = mu<T, Y, I>(y, i);

	for (typename I::const_iterator i_iter = i.begin(); i_iter != i.end(); ++i_iter)
	{
		acc += std::pow((y[*i_iter] - mean), 2);
	}

	return acc / i.size();
}

template<typename T, typename Y, typename X, typename I>
inline std::pair<T, std::pair<I, I> > ig_binary(
		T sigma_y, const Y &y, const X &x, size_t v_num, size_t v, const I &i)
{
	std::pair<I, I> ixs;

	for (typename I::const_iterator i_iter = i.begin(); i_iter != i.end(); ++i_iter)
	{
		if (x[IX(*i_iter, v, v_num)] < BINARY_THRESHOLD)
		{
			ixs.first.push_back(*i_iter);
		}
		else
		{
			ixs.second.push_back(*i_iter);
		}
	}

	T ig = T();

	if ((ixs.first.size() > 0) && (ixs.second.size() > 0))
	{
		size_t total_len = i.size();

		ig = sigma_y
				- (sigma<T, Y, I>(y, ixs.first) * ixs.first.size() / total_len)
				- (sigma<T, Y, I>(y, ixs.second) * ixs.second.size() / total_len);
	}

	return make_pair(ig, ixs);
}

template<typename X>
class xv_comparator
{
	private:
	size_t v_num_, v_;
	const X &x_;

	public:
	xv_comparator(const X &x, size_t v_num, size_t v):
			v_num_(v_num), v_(v), x_(x)
	{
	}

	inline bool operator()(size_t a, size_t b) const
	{
		return x_[IX(a, v_, v_num_)] > x_[IX(b, v_, v_num_)];
	}
};

template<typename T, typename Y, typename X, typename I>
inline std::pair<T, std::pair<T, std::pair<I, I> > > ig_continuous(
		T sigma_y, const Y &y, const X &x, size_t v_num, size_t v, const I &i)
{
	size_t total_len = i.size();

	I sorted_i(i);
	
	std::sort(sorted_i.begin(), sorted_i.end(), xv_comparator<X>(x, v_num, v));

	T best_ig = T();
	std::pair<I, I> best_ixs;
	T best_split = T();

	T ig = T();
	std::pair<I, I> ixs;
	ixs.second = sorted_i;
	T split = x[IX(ixs.second.back(), v, v_num)];

	while (true)
	{
		while ((ixs.second.size() > 0) && (x[IX(ixs.second.back(), v, v_num)] == split))
		{
			ixs.first.push_back(ixs.second.back());
			ixs.second.pop_back();
		}

		if (ixs.second.size() == 0)
		{
			break;
		}

		split = x[IX(ixs.second.back(), v, v_num)];

		ig = sigma_y
				- (sigma<T, Y, I>(y, ixs.first) * ixs.first.size() / total_len)
				- (sigma<T, Y, I>(y, ixs.second) * ixs.second.size() / total_len);

		if (ig > best_ig)
		{
			best_ig = ig;
			best_ixs = ixs;
			best_split = split;
		}
	}

	return make_pair(best_ig, make_pair(best_split, best_ixs));
}

template<typename T, typename Y, typename X, typename I>
inline std::pair<T, std::pair<T, std::pair<I, I> > > ig_continuous_opt(
		T sigma_y, const Y &y, const X &x, size_t v_num, size_t v, const I &i)
{
	size_t total_len = i.size();

	I sorted_i(i);
	
	std::sort(sorted_i.begin(), sorted_i.end(), xv_comparator<X>(x, v_num, v));

	T best_ig = T();
	std::pair<I, I> best_ixs;
	T best_split = T();

	T ig = T();
	std::pair<I, I> ixs;
	ixs.second = sorted_i;
	T split = x[IX(ixs.second.back(), v, v_num)];

	T mu_a = T();
	T sigma_a = T();
	T mu_b = mu<T, Y, I>(y, i);
	T sigma_b = sigma<T, Y, I>(y, i) * i.size();
	size_t size_a = 0;
	size_t size_b = i.size();

	bool frist = true;

	while (true)
	{
		while ((ixs.second.size() > 0) && (x[IX(ixs.second.back(), v, v_num)] == split))
		{
			ixs.first.push_back(ixs.second.back());
			ixs.second.pop_back();

			++size_a;
			--size_b;

			if (frist)
			{
				frist = false;

				mu_a = y[ixs.first.back()];
				sigma_a = 0;
			}
			else
			{
				T mu_a_old = mu_a;
				mu_a = mu_a_old + ((y[ixs.first.back()] - mu_a_old) / size_a);
				sigma_a = sigma_a + ((y[ixs.first.back()] - mu_a_old) * (y[ixs.first.back()] - mu_a));
			}

			T mu_b_old = mu_b;
			mu_b = (((size_b + 1) * mu_b_old) - y[ixs.first.back()]) / (size_b);
			sigma_b = sigma_b - ((y[ixs.first.back()] - mu_b_old) * (y[ixs.first.back()] - mu_b));
		}

		if (ixs.second.size() == 0)
		{
			break;
		}

		split = x[IX(ixs.second.back(), v, v_num)];

		ig = sigma_y - (sigma_a / total_len) - (sigma_b / total_len);

		if (ig > best_ig)
		{
			best_ig = ig;
			best_ixs = ixs;
			best_split = split;
		}
	}

	return make_pair(best_ig, make_pair(best_split, best_ixs));
}

#endif

