#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include "neuron.h"

namespace ToyBrain {

	class Layer {
	protected:
		std::vector<Neuron> neurons;
		Function activation_function;

	public:
		Layer(int num_neurons, int number_of_inputs, Function activation_function);
		void feed_forward(std::span<double> inputs, std::vector<double>& outputs);
		std::vector<double> feed_forward(std::span<double> inputs);
		Function getErrorFunction() { return this->activation_function; };
		std::vector<Neuron>& getMembers() { return this->neurons; };
	};

	class InputLayer : public Layer {

	};

	class OutputLayer : public Layer {

	};

}

#endif // LAYER_H
