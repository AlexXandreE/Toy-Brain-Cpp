
#include "helper_functions.h"

double computeError(std::span<double> output, std::span<double> target) {

	std::span<double>::const_iterator target_iterator = target.begin();
	std::span<double>::const_iterator prediction_iterator = output.begin();
	double error = 0;

	while (prediction_iterator != output.end()) {
		error += *target_iterator - *prediction_iterator;

		target_iterator++;
		prediction_iterator++;
	}

	return error;
}