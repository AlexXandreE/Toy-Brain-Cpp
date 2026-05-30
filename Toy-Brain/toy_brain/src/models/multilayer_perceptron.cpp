#include "multilayer_perceptron.h"

#include <algorithm>
#include <stdexcept>

namespace ToyBrain {

MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int>& layer_sizes, Function activation_function) {
	if (layer_sizes.size() < 2) {
		throw std::invalid_argument("layer_sizes must include at least input and output");
	}

	this->layer_sizes = layer_sizes;

	for (size_t i = 1; i < layer_sizes.size(); ++i) {
		if (layer_sizes[i - 1] <= 0 || layer_sizes[i] <= 0) {
			throw std::invalid_argument("layer sizes must be greater than 0");
		}
		layers.emplace_back(layer_sizes[i], layer_sizes[i - 1], activation_function);
	}

	initialize_buffers();
}

void MultiLayerPerceptron::initialize_buffers() {
	activations_buffer.resize(layer_sizes.size());
	for (size_t i = 0; i < layer_sizes.size(); ++i) {
		activations_buffer[i].assign(static_cast<size_t>(layer_sizes[i]), 0.0);
	}

	deltas_buffer.resize(layers.size());
	for (size_t i = 0; i < layers.size(); ++i) {
		deltas_buffer[i].assign(static_cast<size_t>(layer_sizes[i + 1]), 0.0);
	}

	weights_cache.resize(layers.size());
	for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
		weights_cache[layer_index].resize(static_cast<size_t>(layer_sizes[layer_index + 1]));
		for (size_t neuron_index = 0; neuron_index < weights_cache[layer_index].size(); ++neuron_index) {
			weights_cache[layer_index][neuron_index].assign(static_cast<size_t>(layer_sizes[layer_index]), 0.0);
		}
	}
}

void MultiLayerPerceptron::forward_with_activations(const std::vector<double>& input) {
	if (input.size() != static_cast<size_t>(layer_sizes[0])) {
		throw std::invalid_argument("input size mismatch");
	}

	activations_buffer[0] = input;
	for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
		layers[layer_index].feed_forward(activations_buffer[layer_index], activations_buffer[layer_index + 1]);
	}
}

std::vector<double> MultiLayerPerceptron::forward(const std::vector<double>& input) {
	forward_with_activations(input);
	return activations_buffer.back();
}

int MultiLayerPerceptron::predict(const std::vector<double>& input) {
	forward_with_activations(input);
	const std::vector<double>& scores = activations_buffer.back();
	return static_cast<int>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
}

double MultiLayerPerceptron::train_sample(std::vector<double>& input, uint8_t label, double learning_rate) {
	forward_with_activations(input);
	std::vector<double>& output = activations_buffer.back();

	if (label >= output.size()) {
		throw std::invalid_argument("label out of output range");
	}

	std::fill(deltas_buffer.back().begin(), deltas_buffer.back().end(), 0.0);

	double sample_loss = 0.0;
	for (size_t j = 0; j < output.size(); ++j) {
		const double target = (j == label) ? 1.0 : 0.0;
		const double diff = target - output[j];
		sample_loss += 0.5 * diff * diff;
		deltas_buffer.back()[j] = diff * output[j] * (1.0 - output[j]);
	}

	for (int layer_index = static_cast<int>(layers.size()) - 2; layer_index >= 0; --layer_index) {
		std::fill(
			deltas_buffer[static_cast<size_t>(layer_index)].begin(),
			deltas_buffer[static_cast<size_t>(layer_index)].end(),
			0.0);

		std::vector<Neuron>& next_members = layers[static_cast<size_t>(layer_index + 1)].getMembers();
		std::vector<std::vector<double>>& next_weights_cache = weights_cache[static_cast<size_t>(layer_index + 1)];
		for (size_t j = 0; j < next_members.size(); ++j) {
			next_weights_cache[j] = next_members[j].getWeights();
		}

		const size_t width = deltas_buffer[static_cast<size_t>(layer_index)].size();

		for (size_t i = 0; i < width; ++i) {
			double weighted_error = 0.0;
			for (size_t j = 0; j < next_members.size(); ++j) {
				weighted_error +=
					next_weights_cache[j][i] * deltas_buffer[static_cast<size_t>(layer_index + 1)][j];
			}

			const double a = activations_buffer[static_cast<size_t>(layer_index + 1)][i];
			deltas_buffer[static_cast<size_t>(layer_index)][i] = a * (1.0 - a) * weighted_error;
		}
	}

	for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
		std::vector<Neuron>& members = layers[layer_index].getMembers();
		std::vector<double>& layer_input = activations_buffer[layer_index];
		for (size_t neuron_index = 0; neuron_index < members.size(); ++neuron_index) {
			members[neuron_index].updateWeights(
				deltas_buffer[layer_index][neuron_index],
				learning_rate,
				layer_input);
		}
	}

	return sample_loss;
}

double MultiLayerPerceptron::evaluate(
	std::vector<std::vector<double>>& inputs,
	std::vector<uint8_t>& labels,
	std::vector<std::vector<int>>* confusion_matrix) {
	if (inputs.size() != labels.size()) {
		throw std::invalid_argument("inputs/labels count mismatch");
	}

	if (confusion_matrix != nullptr) {
		const size_t classes = layers.back().getMembers().size();
		confusion_matrix->assign(classes, std::vector<int>(classes, 0));
	}

	size_t correct = 0;
	for (size_t i = 0; i < inputs.size(); ++i) {
		const int pred = predict(inputs[i]);
		const int truth = static_cast<int>(labels[i]);
		if (pred == truth) {
			++correct;
		}
		if (confusion_matrix != nullptr && truth >= 0 && pred >= 0 &&
			truth < static_cast<int>(confusion_matrix->size()) &&
			pred < static_cast<int>((*confusion_matrix)[truth].size())) {
			(*confusion_matrix)[truth][pred]++;
		}
	}

	return inputs.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(inputs.size());
}

} // namespace ToyBrain
