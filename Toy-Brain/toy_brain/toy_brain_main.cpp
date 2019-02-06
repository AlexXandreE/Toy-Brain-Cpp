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
	Layer secondLayer(4, 2, Function::sigmoid);
	layers.push_back(secondLayer);
	Layer thirdLayer(1, 4, Function::sigmoid);
	layers.push_back(thirdLayer);
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
	
	std::vector<double> outputs = { 1, 1, 1, 0 };

	NeuralNetwork network(layers);

	
	std::vector<double> output = network.train(10, inputs, outputs, 0.1, 0.1);
	
	for (double value : output) {
		cout << "Feed foward result: " << value << endl;
	}
	system("pause");

	return 0;
}
