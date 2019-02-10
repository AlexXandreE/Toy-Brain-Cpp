
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

enum Function {
	sigmoid,
	step,
	rectifier
};

class ActivationFunction {
private:
	Function function;
public:
	ActivationFunction();
	ActivationFunction(Function type);
	double compute(double value);
};

#endif

