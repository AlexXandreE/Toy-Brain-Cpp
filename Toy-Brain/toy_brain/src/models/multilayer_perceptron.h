#pragma once

#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include <cstdint>
#include <vector>

#include "layer.h"

namespace ToyBrain {

class MultiLayerPerceptron {
private:
	std::vector<Layer> layers;
	std::vector<int> layer_sizes;
	std::vector<std::vector<double>> activations_buffer;
	std::vector<std::vector<double>> deltas_buffer;
	std::vector<std::vector<std::vector<double>>> weights_cache;

	void initialize_buffers();
	void forward_with_activations(const std::vector<double>& input);

public:
	MultiLayerPerceptron(const std::vector<int>& layer_sizes, Function activation_function = Function::sigmoid);

	std::vector<double> forward(const std::vector<double>& input);
	int predict(const std::vector<double>& input);
	double train_sample(std::vector<double>& input, uint8_t label, double learning_rate);
	double evaluate(
		std::vector<std::vector<double>>& inputs,
		std::vector<uint8_t>& labels,
		std::vector<std::vector<int>>* confusion_matrix = nullptr);
};

} // namespace ToyBrain

#endif // MULTILAYER_PERCEPTRON_H
