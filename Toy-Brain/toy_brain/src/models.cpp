
#include "models.h"
#include "helper_functions.h"
#include <cfloat>

namespace ToyBrain {
	
	/* ErrorFunction implementation */
	ErrorFunction::ErrorFunction() {
		this->function = Function::sigmoid;
	}

	ErrorFunction::ErrorFunction(Function type) {
		this->function = type;
	}

	double ErrorFunction::compute(double value) {
		switch (this->function)
		{
		case sigmoid:
			return 1.0 / (1.0 + exp(-value));

		case step:
			return value > 0 ? 1 : 0;

		case rectifier:
			return value < 0 ? 0 : value;

		case least_mean_square:
			return value * value;
		default:
			return 0;
		}
	}
}