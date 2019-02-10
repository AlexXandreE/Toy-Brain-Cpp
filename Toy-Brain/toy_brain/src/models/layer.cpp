
#include "layer.h"

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