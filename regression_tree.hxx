
#ifndef INCLUDE__REGRESSION_TREE_HXX

#define INCLUDE__REGRESSION_TREE_HXX

#include <iostream>
#include <istream>
#include <ostream>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "util.hxx"

#include "stats.hxx"

static const fp_type TOLERANCE_THRESHOLD = 1e-6;

template<typename T, typename Y, typename X, typename I, typename L>
class regression_tree
{
	private:
	enum node_type
	{
		RT_NODE_TYPE_INVALID,
		RT_NODE_TYPE_LEAF,
		RT_NODE_TYPE_BINARY,
		RT_NODE_TYPE_CONTINUOUS
	};

	size_t dim_n_, dim_v_;
	node_type node_type_;
	T prediction_;
	size_t v_;
	T split_;
	std::vector<regression_tree<T, Y, X, I, L> > children_;

	public:
	regression_tree(const Y &y, const X &x, const I &i, const L &l, size_t n, std::ostream &log):
			dim_n_(y.size()), dim_v_(x.size() / dim_n_), node_type_(RT_NODE_TYPE_INVALID),
			prediction_(), v_(), split_(), children_()
	{
		train(y, x, i, l, n, log);
	}

	regression_tree(std::istream &in):
			children_()
	{
		expect(in, "regression_tree");
		expect(in, "{");

		in >> dim_n_;
		in >> dim_v_;

		std::string node_type_str;

		in >> node_type_str;

		if (node_type_str == "RT_NODE_TYPE_LEAF")
		{
			node_type_ = RT_NODE_TYPE_LEAF;
		}
		else if (node_type_str == "RT_NODE_TYPE_BINARY")
		{
			node_type_ = RT_NODE_TYPE_BINARY;
		}
		else if (node_type_str == "RT_NODE_TYPE_CONTINUOUS")
		{
			node_type_ = RT_NODE_TYPE_CONTINUOUS;
		}
		else
		{
#ifndef NDEBUG
			FLEX_ASSERT(false);
#endif

			node_type_ = RT_NODE_TYPE_INVALID;
		}

		in >> prediction_;
		in >> v_;
		in >> split_;

		size_t children_size;

		in >> children_size;

		expect(in, "{");

		for (size_t i = 0; i != children_size; ++i)
		{
			children_.push_back(regression_tree<T, Y, X, I, L>(in));
		}

		expect(in, "}");

		expect(in, "}");
	}

	void train(const Y &y, const X &x, const I &i, const L &l, size_t n, std::ostream &log)
	{
#ifndef NDEBUG
		FLEX_ASSERT(node_type_ == RT_NODE_TYPE_INVALID);
#endif

#ifdef LOG_OUTPUT
		log << i.size() << " observation(s) of " << dim_v_ << " variable(s)" << std::endl;
#endif

		T sigma_y = sigma<T, Y, I>(y, i);

#ifdef LOG_OUTPUT
		log << "H(Y): " << sigma_y << std::endl;
#endif

		std::vector<size_t> vs(dim_v_);

		size_t j = 0;

		for (std::vector<size_t>::iterator vs_iter = vs.begin(); vs_iter != vs.end(); ++vs_iter)
		{
			(*vs_iter) = j++;
		}

		std::random_shuffle(vs.begin(), vs.end());

		T best_ig = T();
		size_t best_v = 0;
		node_type best_node_type = RT_NODE_TYPE_INVALID;
		T best_split = T();
		std::pair<I, I> best_ixs;

		if (sigma_y > TOLERANCE_THRESHOLD)
		{
			for (size_t j = 0; j != n; ++j)
			{
				size_t v = vs[j];

				if (l[v] == 2)
				{
					std::pair<T, std::pair<I, I> > ig = ig_binary(sigma_y, y, x, dim_v_, v, i);

#ifdef LOG_OUTPUT
#ifdef LOG_OUTPUT_VERBOSE_TRAINING
					log << "IG(Y | X" << v << ") = " << ig.first << std::endl;
#endif
#endif

					if (ig.first > best_ig)
					{
						best_ig = ig.first;
						best_v = v;
						best_node_type = RT_NODE_TYPE_BINARY;
						best_split = T();
						best_ixs = ig.second;
					}
				}
				else if (l[v] == 0)
				{
					/*
					std::pair<T, std::pair<T, std::pair<I, I> > > ig_0 =
							ig_continuous(sigma_y, y, x, dim_v_, v, i);
					*/

					std::pair<T, std::pair<T, std::pair<I, I> > > ig =
							ig_continuous_opt(sigma_y, y, x, dim_v_, v, i);

					// assert(std::abs(ig_0.first - ig.first) < 0.001);

#ifdef LOG_OUTPUT
#ifdef LOG_OUTPUT_VERBOSE_TRAINING
					log << "IG(Y | X" << v << " / " << ig.second.first << ") = " << ig.first << std::endl;
#endif
#endif

					if (ig.first > best_ig)
					{
						best_ig = ig.first;
						best_v = v;
						best_node_type = RT_NODE_TYPE_CONTINUOUS;
						best_split = ig.second.first;
						best_ixs = ig.second.second;
					}
				}
				else
				{
#ifdef LOG_OUTPUT
					log << "Unsupported variable type for X" << v << " (" << l[v] << ") - skipping" << std::endl;
#endif

#ifndef NDEBUG
					FLEX_ASSERT(false);
#endif

					continue;
				}
			}
		}

		if (best_ig <= TOLERANCE_THRESHOLD)
		{
#ifdef LOG_OUTPUT
			log << "No information gain possible." << std::endl;
#endif

			node_type_ = RT_NODE_TYPE_LEAF;
			prediction_ = mu<T, Y, I>(y, i);

#ifdef LOG_OUTPUT
			log << "H(Y): " << sigma_y << std::endl;
			log << "Leaf node reached, prediction: " << prediction_ << std::endl;
#endif
		}
		else
		{
#ifdef LOG_OUTPUT
			log << "Splitting on X" << best_v;
#endif

			if (best_node_type == RT_NODE_TYPE_CONTINUOUS)
			{
#ifdef LOG_OUTPUT
				log << " (split on " << best_split << ")";
#endif
			}

#ifdef LOG_OUTPUT
			log << std::endl;

			log << "Information gain: " << best_ig << std::endl;
#endif

			prediction_ = T();
			v_ = best_v;
			node_type_ = best_node_type;
			split_ = best_split;

			children_.push_back(regression_tree(y, x, best_ixs.first, l, n, log));
			children_.push_back(regression_tree(y, x, best_ixs.second, l, n, log));
		}
	}

	T predict(const X &x, size_t i = 0) const
	{
#ifndef NDEBUG
		FLEX_ASSERT(node_type_ != RT_NODE_TYPE_INVALID);
#endif

		if (node_type_ == RT_NODE_TYPE_LEAF)
		{
			return prediction_;
		}
		else if (node_type_ == RT_NODE_TYPE_BINARY)
		{
			return children_[(x[(i * dim_v_) + v_] < BINARY_THRESHOLD) ? 0 : 1].predict(x, i);
		}
		else if (node_type_ == RT_NODE_TYPE_CONTINUOUS)
		{
			return children_[(x[(i * dim_v_) + v_] < split_) ? 0 : 1].predict(x, i);
		}
		else
		{
#ifndef NDEBUG
			FLEX_ASSERT(false);
#endif

			return T();
		}
	}

	void serialize(std::ostream &out) const
	{
#ifndef NDEBUG
		FLEX_ASSERT(node_type_ != RT_NODE_TYPE_INVALID);
#endif

		out << "regression_tree" << std::endl;
		out << "{" << std::endl;

		out << dim_n_ << " " << dim_v_ << std::endl;

		if (node_type_ == RT_NODE_TYPE_LEAF)
		{
			out << "RT_NODE_TYPE_LEAF" << std::endl;
		}
		else if (node_type_ == RT_NODE_TYPE_BINARY)
		{
			out << "RT_NODE_TYPE_BINARY" << std::endl;
		}
		else if (node_type_ == RT_NODE_TYPE_CONTINUOUS)
		{
			out << "RT_NODE_TYPE_CONTINUOUS" << std::endl;
		}
		else
		{
			out << "RT_NODE_TYPE_INVALID" << std::endl;
		}

		out << prediction_ << std::endl;
		out << v_ << std::endl;
		out << split_ << std::endl;

		out << children_.size() << std::endl;
		out << "{" << std::endl;

		for (typename std::vector<regression_tree<T, Y, X, I, L> >::const_iterator
				child_iter = children_.begin(); child_iter != children_.end(); ++child_iter)
		{
			child_iter->serialize(out);
		}

		out << "}" << std::endl;

		out << "}" << std::endl;
	}
};

#endif

