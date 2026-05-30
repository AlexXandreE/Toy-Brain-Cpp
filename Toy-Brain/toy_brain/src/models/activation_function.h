
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

namespace ToyBrain {

	enum Function {
		sigmoid,
		step,
		rectifier,
		least_mean_square
	};

	class ActivationFunction {
	private:
		Function function;
	public:
		ActivationFunction();
		ActivationFunction(Function type);
		double compute(double value);
	};

}

#endif

