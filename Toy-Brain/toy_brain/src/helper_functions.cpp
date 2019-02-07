
#include "helper_functions.h"

double computeError(std::vector<double> output, std::vector<double> target) {

	std::vector<double>::const_iterator target_iterator = target.begin(), prediction_iterator = output.begin();
	double error = 0;

	while (prediction_iterator != output.end()) {
		error += *target_iterator - *prediction_iterator;

		target_iterator++;
		prediction_iterator++;
	}

	return error;
}