
#include "neuron.h"
#include <random>
#include <iostream>
#include <cassert>

namespace ToyBrain {

	/* Neuron */
	// Class implementation
	Neuron::Neuron(int number_of_inputs, Function activation_function) {
		if (number_of_inputs <= 0) {
			throw std::invalid_argument("Number of inputs must be greater than 0");
		}

		// Generating random values between -1 and 1
		// TODO: get seed instead of complete random
		std::random_device rd;
		std::default_random_engine re(rd());
		std::uniform_real_distribution<double> uniform_dist(-1, 1);

		this->weights = std::vector<double>(number_of_inputs);

		for (size_t i = 0; i < number_of_inputs; i++) {
			this->weights[i] = uniform_dist(re);
		}

		this->bias = uniform_dist(re);
		this->function = ActivationFunction(activation_function);
	}

	double Neuron::feed_forward(std::span<double> inputs) {
		assert(inputs.size() == this->weights.size());

		double total = this->bias;
		for (size_t i = 0; i < inputs.size(); i++) {
			total += inputs[i] * this->weights[i];
		}

		return this->function.compute(total);
	}

	void Neuron::updateWeights(double delta_error, double learning_rate, std::span<double> inputs) {
		assert(inputs.size() == this->weights.size());

		for (size_t i = 0; i < this->weights.size(); i++) {
			this->weights[i] = this->weights[i] + (learning_rate * delta_error) * inputs[i];
		}
		this->bias += learning_rate * delta_error;
	}
}
