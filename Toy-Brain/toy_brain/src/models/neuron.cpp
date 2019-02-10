
#include "neuron.h"

/* Neuron */
// Class implementation
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

double Neuron::feed_forward(std::vector<double> inputs) {
	if (inputs.size() != this->weights.size()) {
		std::cout << "Throw error -> Inputs are different than weights length" << std::endl;
	}

	double total = 0;
	for (size_t i = 0; i < inputs.size(); i++) {
		total += inputs[i] * this->weights[i];// + this->bias;
	}

	return this->function.compute(total);
}

void Neuron::updateWeights(double delta_error, double learning_rate, std::vector<double> inputs) {
	for (size_t i = 0; i < this->weights.size(); i++) {
		this->weights[i] = this->weights[i] + (learning_rate * delta_error) * inputs[i];
	}
}

// Operator '<<' override to print 
std::ostream &operator<<(std::ostream &os, Neuron const &m) {
	return os << "[Neuron] {\n\tweights -> " << m.weights[0] << ",\n\tBias -> " << m.bias << "\n}";
}
