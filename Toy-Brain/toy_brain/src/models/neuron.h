#pragma once

#ifndef NEURON_H
#define NEURON_H

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
	double getBias() { return this->bias; }
	ActivationFunction getActivationFunction() { return this->function; }

	friend std::ostream &operator<<(std::ostream &os, const Neuron &m);
};

#endif // NEURON_H


