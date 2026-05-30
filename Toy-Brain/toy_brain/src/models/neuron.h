#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <span>
#include <iostream>
#include "activation_function.h"

namespace ToyBrain {

	class Neuron {
	private:
		std::vector<double> weights;
		double bias = 0;
		ActivationFunction function;

	public:
		Neuron(int number_of_inputs, Function activation_function);

		double feed_forward(std::span<double> inputs);

		void updateWeights(double delta_error, double learning_rate, std::span<double> inputs);

		const std::vector<double>& getWeights() const { return this->weights; }
		double getBias() { return this->bias; }
		ActivationFunction getActivationFunction() { return this->function; }

		friend std::ostream &operator<<(std::ostream &os, const Neuron &m);
	};

}

#endif // NEURON_H


