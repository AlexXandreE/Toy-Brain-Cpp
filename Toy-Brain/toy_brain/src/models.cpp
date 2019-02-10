
#include "models.h"
#include "helper_functions.h"

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

/**
 * Activation function class implementation
 */
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

		for (size_t i = 0; i < number_of_inputs + 1; i++) {
			this->weights.push_back(uniform_dist(re));
		}

		this->function = ActivationFunction(activation_function);
	}

	double Neuron::feed_forward(std::vector<double> inputs) {
		if (inputs.size() != this->weights.size() - 1) {
			std::cout << "Throw error -> Inputs are different than weights length" << std::endl;
		}

		double total = 0;
		for (size_t i = 0; i < inputs.size(); i++) {
			total += inputs[i] * this->weights[i];// + this->bias;
		}

		total += this->weights[this->weights.size() - 1] * this->bias;

		return this->function.compute(total);
	}

	void Neuron::updateWeights(double delta_error, double learning_rate, std::vector<double> inputs) {
		for (size_t i = 0; i < this->weights.size() - 1; i++) {
			this->weights[i] += (learning_rate * delta_error) * inputs[i];
		}

		this->weights[this->weights.size() - 1] += (learning_rate * delta_error) * this->bias;
	}

	// Operator '<<' override to print 
	std::ostream &operator<<(std::ostream &os, Neuron const &m) {
		return os << "[Neuron] {\n\tweights -> " << m.weights[0] << ",\n\tBias -> " << m.bias << "\n}";
	}

	/* Layer */
	// Class implementation
	Layer::Layer(int num_neurons, int number_of_inputs, Function activation_function) {
		for (int i = 0; i < num_neurons; i++) {
			this->neurons.push_back(Neuron(number_of_inputs, activation_function));
		}
		this->activation_function = activation_function;
	}

	std::vector<double> Layer::feed_forward(std::vector<double> inputs) {
		std::vector<double> outputs;
		for (Neuron neuron : this->neurons) {
			outputs.push_back(neuron.feed_forward(inputs));
		}

		return outputs;
	}

	/* NEURAL NETWORK */
	// Class implementation
	NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) {
		this->layers = layers;
	}

	void NeuralNetwork::backpropagation(std::vector<double> inputs, std::vector<double> expected_outputs, std::vector<std::vector<double>> results_per_layer, std::vector<std::vector<double>> errors_per_layer, double learning_rate) {
		std::vector<double> output_layer_error;
		std::vector<double> result = inputs;
		// last layer
		std::vector<Neuron> neurons = this->layers[this->layers.size() - 1].getMembers();
		for (size_t index = 0; index < neurons.size(); index++) {
			double delta_error = (expected_outputs[index] - result[index]) *  result[index] * (1 - result[index]);
			output_layer_error.push_back(delta_error);

			if (this->layers.size() > 1) {
				neurons[index].updateWeights(delta_error, learning_rate, results_per_layer[this->layers.size() - 2]);
			}
			else {
				neurons[index].updateWeights(delta_error, learning_rate, inputs);
			}
		}
		errors_per_layer.push_back(output_layer_error);
		//layer_error.clear();

		// Hidden layers
		for (size_t layer_index = this->layers.size() - 2, errors_index = 0; layer_index >= 1; layer_index--, errors_index++) {
			std::vector<Neuron> neurons = this->layers[layer_index + 1].getMembers();
			std::vector<double> hidden_layer_error;
			double error_weight_sum = 0;
			// Sum of values from last layer
			for (size_t index = 0; index < neurons.size(); index++) {
				for (double weight : neurons[index].getWeights()) {
					error_weight_sum += weight * errors_per_layer[errors_index][index];
				}
			}
			neurons = this->layers[layer_index].getMembers();

			for (size_t index = 0; index < neurons.size(); index++) {
				double delta_error = results_per_layer[layer_index][index] * (1 - results_per_layer[layer_index][index]) * error_weight_sum;
				hidden_layer_error.push_back(delta_error);
				neurons[index].updateWeights(delta_error, learning_rate, results_per_layer[layer_index]);
			}
			errors_per_layer.push_back(hidden_layer_error);
		}

		// updating first layer weights
		std::vector<Neuron> first_layer_neurons = this->layers[0].getMembers();

		double error_weight_sum = 0;
		for (size_t index = 0; index < first_layer_neurons.size(); index++) {
			for (double weight : first_layer_neurons[index].getWeights()) {
				error_weight_sum += weight * errors_per_layer[errors_per_layer.size() - 1][index];
			}
		}

		for (size_t index = 0; index < first_layer_neurons.size(); index++) {
			double delta_error = results_per_layer[0][index] * (1 - results_per_layer[0][index]) * error_weight_sum;

			first_layer_neurons[index].updateWeights(delta_error, learning_rate, inputs);
		}
	}

	void NeuralNetwork::train(int epochs, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected_outputs, double learning_rate, double minimum_error) {
		int total_epochs_done = 0;
		double current_error = DBL_MAX;

		std::cout << "Started training" << std::endl;

		while (total_epochs_done < epochs || current_error < minimum_error) {
			int current_result_index = 0;

			for (std::vector<double> input : inputs) {

				std::vector<Layer>::iterator layer_iterator = this->layers.begin();

				std::vector<double> result = input;
				std::vector<std::vector<double>> results_per_layer;
				std::vector<std::vector<double>> errors_per_layer;

				do {
					result = layer_iterator->feed_forward(result);
					results_per_layer.push_back(result);
					layer_iterator++;

				} while (layer_iterator < this->layers.end());

				bool changes = false;

				for (size_t index = 0; index < result.size(); index++) {
					if (std::round(result[index]) != expected_outputs[current_result_index][index]) {
						changes = true;
						break;
					}
				}

				if (changes) {
					/**
					* Back propagation
					*/
					this->backpropagation(input, expected_outputs[current_result_index], results_per_layer, errors_per_layer, learning_rate);
				}
				current_result_index++;
			}
			total_epochs_done++;
		}
	}

	std::vector<std::vector<double>> NeuralNetwork::compute(std::vector<std::vector<double>> inputs) {
		std::vector<std::vector<double>> results;
		for (std::vector<double> input : inputs) {
			std::vector<Layer>::iterator layer_iterator = this->layers.begin();

			std::vector<double> result = input;
			do {
				result = layer_iterator->feed_forward(result);
				layer_iterator++;
			} while (layer_iterator < this->layers.end());

			results.push_back(result);
		}

		return results;
	}
}