
#include "models.h"

/*
RandomGenerator::RandomGenerator(int seed) {

}

double RandomGenerator::generateRandomDoubleNumber(double lowerBound, double upperBound) {
	std::default_random_engine generator;
	std::uniform_int_distribution<double> distribution(lowerBound, upperBound);
	return distribution(generator);
}
*/

/*
	Activation function definition:
		- Expects the input of type Function 
		- calculate() function retrieves the function result
*/

ActivationFunction::ActivationFunction() {
	this->function = Function::sigmoid;
}

ActivationFunction::ActivationFunction(Function type) {
	this->function = type;
}

double ActivationFunction::calculate(double value) {
	switch (this->function)
	{
	case sigmoid:
		return 1 / (1 + exp(value));

	case step:
		return -1;
	
	case rectifier:
		return value < 0 ? 0 : value;
	default:
		return 0;
	}
}

Neuron::Neuron(int number_of_inputs, Function activation_function) {
	if (number_of_inputs <= 0) {
		// TODO: Throw error
	}

	this->weights = std::vector<double>();
	
	// Generating random values between -1 and 1
	// TODO: get seed instead of complete random
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_real_distribution<double> uniform_dist(-1, 1);

	for (size_t i = 0; i < number_of_inputs; i++) {
		this->weights.push_back(uniform_dist(re));
	}

	this->function = ActivationFunction(activation_function);
}

Neuron::~Neuron() {
	//delete(this->weights);
}

double Neuron::feed_forward(std::vector<double> inputs) {
	if (inputs.size() != this->weights.size()) {
		std::cout << "Throw error" << std::endl;
	}
	
	double total = 0;
	for (size_t i = 0; i < inputs.size(); i++) {
		total += inputs[i] * this->weights[i] + this->bias;
	}

	return this->function.calculate(total);
}

// Operator '<<' override to print 
std::ostream &operator<<(std::ostream &os, Neuron const &m) {
	return os << "[Neuron] {\n\tweights -> " << m.weights[0] << ",\n\tBias -> " << m.bias << "\n}";
}
