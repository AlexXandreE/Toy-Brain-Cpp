#pragma once

// TODO: Implement these

#ifndef MODELS_H
#define MODELS_H

#include <stdlib.h> 
#include <random>
#include <optional>
#include <iostream>
#include <math.h>


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

#endif //  MODELS_H