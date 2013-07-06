
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>

#include <cstdlib>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <string>
#include <vector>

#include <sys/time.h>
#include <sys/resource.h>

#include "rg.hxx"

#include "util.hxx"

#include "gearman.hxx"
#include "regression_grove.hxx"

static const int worker_timeout = 10;

static long long jobs_processed = 0;

void *predict(gearman_job_st *job, void *context, size_t *result_size, gearman_return_t *ret_ptr)
{
#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Job taken..." << std::endl;
#endif

	regression_grove<fp_type, std::vector<fp_type>, std::vector<fp_type>, std::vector<size_t> > *model =
			static_cast<regression_grove<fp_type, std::vector<fp_type>, std::vector<fp_type>, std::vector<size_t> > *>(context);

	char *inp_char = static_cast<char *>(std::malloc(gearman_job_workload_size(job) + 1));

	std::memcpy(inp_char, gearman_job_workload(job), gearman_job_workload_size(job));

	inp_char[gearman_job_workload_size(job)] = '\0';

	std::string input(inp_char);

#ifdef LOG_OUTPUT
	std::cerr << "Predictors: " << input << std::endl;
#endif

	std::stringstream input_stream(input);

	std::vector<fp_type> predictors(model->v());

	for (size_t i = 0; i != model->v(); ++i)
	{
		input_stream >> predictors[i];
	}

	free(inp_char);

	std::vector<fp_type> predictions = model->predict(predictors, 0);

	std::sort(predictions.begin(), predictions.end());

	std::stringstream result_stream;

	result_stream << "[\"";

	bool frist = true;

	for (std::vector<fp_type>::const_iterator pred_iter = predictions.begin(); pred_iter != predictions.end(); ++pred_iter)
	{
		if (frist)
		{
			frist = false;
		}
		else
		{
			result_stream << " ";
		}

		result_stream << (*pred_iter);
	}

	result_stream << "\"]";

	std::string result(result_stream.str());
	
#ifdef LOG_OUTPUT
	std::cerr << "Predictions: " << result << std::endl;
#endif

	char *res_char = static_cast<char *>(std::malloc(result.size()));

	std::memcpy(res_char, result.c_str(), result.size());

	*result_size = result.size();

	*ret_ptr = GEARMAN_SUCCESS;

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Complete." << std::endl;
#endif

	++jobs_processed;

	return res_char;
}

int main(int argc, char **argv)
{
#ifndef NDEBUG
	FLEX_ASSERT(argc == 3);
#endif

	std::srand(std::time(NULL));

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Loading model..." << std::endl;
#endif

	std::ifstream model(argv[1]);

	regression_grove<fp_type, std::vector<fp_type>, std::vector<fp_type>, std::vector<size_t> >
			rg(model);

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Done." << std::endl;
#endif

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Initializing gearman worker..." << std::endl;
#endif

	gearman_worker_st *worker = gearman_worker_create(NULL);

	GEARMAN(gearman_worker_add_servers(worker, argv[2]));

	GEARMAN(gearman_worker_add_function(worker, "predict", worker_timeout,
			&predict, static_cast<void *>(&rg)));

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Done." << std::endl;
#endif

#ifdef LOG_OUTPUT_BASIC
	std::cerr << "Ready." << std::endl;
#endif

	while (true)
	{
		rusage res_usage;

		FLEX_ASSERT(getrusage(RUSAGE_SELF, &res_usage) == 0);

#ifdef LOG_OUTPUT_BASIC
		std::cerr << " maxrss: " << res_usage.ru_maxrss;
		std::cerr << " (jobs processed: " << jobs_processed << ")";
		std::cerr << std::endl;
#endif

		GEARMAN(gearman_worker_work(worker));
	}

	return 0;
}

