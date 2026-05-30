#include "activation_function.h"
#include <cmath>

namespace ToyBrain {

	ActivationFunction::ActivationFunction() {
		this->function = Function::sigmoid;
	}

	ActivationFunction::ActivationFunction(Function type) {
		this->function = type;
	}

	double ActivationFunction::compute(double value) {
		switch (this->function)
		{
			case sigmoid:
				return 1.0 / (1.0 + exp(-value));

			case step:
				return value > 0 ? 1 : 0;

			case rectifier:
				return value < 0 ? 0 : value;
			default:
				return 0;
		}
	}

}
