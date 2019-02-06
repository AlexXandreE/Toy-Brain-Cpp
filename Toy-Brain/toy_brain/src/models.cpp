
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
		std::cout << "Throw error -> Inputs are different than weights length" << std::endl;
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


Layer::Layer(int num_neurons, int number_of_inputs, Function activation_function) {
	for (int i = 0; i < num_neurons; i++) {
		this->neurons.push_back(Neuron(number_of_inputs, activation_function));
	}
}

std::vector<double> Layer::feed_forward(std::vector<double> inputs) {
	std::vector<double> outputs;
	for (Neuron neuron : this->neurons) {
		outputs.push_back(neuron.feed_forward(inputs));
	}

	return outputs;
}

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) {
	this->layers = layers;
}

std::vector<double> NeuralNetwork::train(int epochs, std::vector<std::vector<double>> inputs, std::vector<double> expected_outputs, double learning_rate, double minimum_error) {
	int total_epochs_done = 0;
	double current_error = DBL_MAX;
	std::vector<double> last_result;
	std::cout << "Started training" << std::endl;
	
	while(total_epochs_done < epochs || current_error < minimum_error) {
		
		for (std::vector<double> input : inputs) {

			std::vector<Layer>::iterator layer_iterator = this->layers.begin();

			std::vector<double> result = input;

			do {
				result = layer_iterator->feed_forward(result);
				layer_iterator++;

			} while (layer_iterator < this->layers.end());

			double error = 0;

			for (
				std::vector<double>::const_iterator target_iterator = expected_outputs.begin(),
				prediction_iterator = result.begin();
				prediction_iterator != result.end();
				target_iterator++,
				prediction_iterator++
				) {

				error += *target_iterator - *prediction_iterator;

			}

			current_error += error;
			last_result = result;
		}
		total_epochs_done++;
		std::cout << "In epoch: " << total_epochs_done << std::endl;
	}

	return last_result;
}


