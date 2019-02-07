// toy_brain.cpp : Defines the entry point for the application.
//

#include "toy_brain_main.h"
#include "src/models.h"

using namespace std;

int main()
{
	//ActivationFunction function1(Function::sigmoid);

	//Neuron first(2, Function::sigmoid);
	std::vector<Layer> layers;
	Layer firstLayer(2, 2, Function::sigmoid);
	layers.push_back(firstLayer);
	Layer secondLayer(1, 2, Function::sigmoid);
	layers.push_back(secondLayer);
	//Layer thirdLayer(1, 2, Function::sigmoid);
	//layers.push_back(thirdLayer);
	// TODO: Or operation test
	
	std::vector<std::vector<double>> inputs;

	std::vector<double> firstInput = { 1, 1 };
	inputs.push_back(firstInput);
	std::vector<double> secondInput = { 0, 1 };
	inputs.push_back(secondInput);
	std::vector<double> thirdInput = { 1, 0 };
	inputs.push_back(thirdInput);
	std::vector<double> fourthInput = { 0, 0 };
	inputs.push_back(fourthInput);
	//cout << first << endl;


	//std::vector<double> output = singleLayer.feed_forward(inputs);
	
	std::vector<std::vector<double>> outputs;
	std::vector<double> result1 = { 1 };
	std::vector<double> result2 = { 1 };
	std::vector<double> result3 = { 1 };
	std::vector<double> result4 = { 0 };

	outputs.push_back(result1);
	outputs.push_back(result2);
	outputs.push_back(result3);
	outputs.push_back(result4);

	NeuralNetwork network(layers);

	
	network.train(50, inputs, outputs, 0.1, 0.1);
	
	std::vector<std::vector<double>> results = network.compute(inputs);

	std::cout << "OR Function:" << std::endl;
	for (std::vector<double> value : results) {
		std::cout << "=> [" << value[0] << "]" << std::endl;
	}
	system("pause");

	return 0;
}
