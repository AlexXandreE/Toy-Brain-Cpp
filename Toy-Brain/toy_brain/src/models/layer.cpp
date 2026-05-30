
#include "layer.h"

namespace ToyBrain {

	/* Layer */
	// Class implementation
	Layer::Layer(int num_neurons, int number_of_inputs, Function activation_function) {
		for (int i = 0; i < num_neurons; i++) {
			this->neurons.push_back(Neuron(number_of_inputs, activation_function));
		}
		this->activation_function = activation_function;
	}

	void Layer::feed_forward(std::span<double> inputs, std::vector<double>& outputs) {
		outputs.resize(this->neurons.size());
		for (size_t i = 0; i < this->neurons.size(); ++i) {
			outputs[i] = this->neurons[i].feed_forward(inputs);
		}
	}

	std::vector<double> Layer::feed_forward(std::span<double> inputs) {
		std::vector<double> outputs;
		feed_forward(inputs, outputs);
		return outputs;
	}

}