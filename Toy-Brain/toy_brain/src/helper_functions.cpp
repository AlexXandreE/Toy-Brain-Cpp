
#include "helper_functions.h"

double getOutputError(std::vector<double> output, std::vector<double> target) {

	std::vector<double>::const_iterator target_iterator = target.begin(), prediction_iterator = result.begin();

	while (prediction_iterator != result.end()) {
		error += *target_iterator - *prediction_iterator;

		target_iterator++,
			prediction_iterator++
	}

	return error;
}