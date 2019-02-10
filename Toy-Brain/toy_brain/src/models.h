#pragma once

// TODO: Implement these

#ifndef MODELS_H
#define MODELS_H

#include <stdlib.h> 
#include <random>
#include <optional>
#include <iostream>
#include <math.h>
#include <limits>
#include <cmath>

// TODO: Check if I need to use this
/*
class RandomGenerator {
private:
	std::optional<int> seed;
	std::default_random_engine generator;

public:
	RandomGenerator(int seed);
	double generateRandomDoubleNumber();
	int getCurrentSeed();
};
*/

namespace ToyBrain {

	enum Function {
		sigmoid,
		step,
		rectifier,
		least_mean_square
	};

	class ErrorFunction {
	private:
		Function function;
	public:
		ErrorFunction();
		ErrorFunction(Function type);
		double compute(double value);
	};

	class ActivationFunction {
	private:
		Function function;
	public:
		ActivationFunction();
		ActivationFunction(Function type);
		double compute(double value);
	};

	class Neuron {
	private:
		std::vector<double> weights;
		double bias = -1;
		ActivationFunction function;

	public:
		Neuron(int number_of_inputs, Function activation_function); //, RandomGenerator randomHandler);

		double feed_forward(std::vector<double> inputs);

		void updateWeights(double delta_error, double learning_rate, std::vector<double> inputs);

		std::vector<double> getWeights() { return this->weights; }

		friend std::ostream &operator<<(std::ostream &os, const Neuron &m);
	};

	/**
	 * Layer class implementation
	 */
	class Layer {
	private:
		std::vector<Neuron> neurons;
		Function activation_function;

	public:
		Layer(int num_neurons, int number_of_inputs, Function activation_function);
		std::vector<double> feed_forward(std::vector<double> inputs);
		Function getErrorFunction() { return this->activation_function; };
		std::vector<Neuron> getMembers() { return this->neurons; };
	};

	/**
	 * Neural Network class implementation
	 */
	class NeuralNetwork {
	private:
		std::vector<Layer> layers;
		void NeuralNetwork::backpropagation(std::vector<double> inputs, std::vector<double> expected_outputs, std::vector<std::vector<double>> results_per_layer, std::vector<std::vector<double>> errors_per_layer, double learning_rate);

	public:
		NeuralNetwork(std::vector<Layer> layers);
		void NeuralNetwork::train(int epochs, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expected_outputs, double learning_rate, double minimum_error);
		std::vector<std::vector<double>> compute(std::vector<std::vector<double>> inputs);
	};
}

#endif //  MODELS_H