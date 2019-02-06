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

enum Function {
	sigmoid,
	step,
	rectifier
};

class ActivationFunction {
private:
	Function function;
public: 
	ActivationFunction();
	ActivationFunction(Function type);
	double calculate(double value);
};

class Neuron {
private:
	std::vector<double> weights;
	double bias = -1;
	ActivationFunction function;

public:
	Neuron(int number_of_inputs, Function activation_function); //, RandomGenerator randomHandler);
	~Neuron();

	double feed_forward(std::vector<double> inputs);

	friend std::ostream &operator<<(std::ostream &os, const Neuron &m);
};

class Layer {
private:
	std::vector<Neuron> neurons;

public:
	Layer(int num_neurons, int number_of_inputs, Function activation_function);
	std::vector<double> feed_forward(std::vector<double> inputs);
};

class NeuralNetwork {
private:
	std::vector<Layer> layers;

public:
	NeuralNetwork(std::vector<Layer> layers);
	std::vector<double> NeuralNetwork::train(int epochs, std::vector<std::vector<double>> inputs, std::vector<double> expected_outputs, double learning_rate, double minimum_error);
};

#endif //  MODELS_H