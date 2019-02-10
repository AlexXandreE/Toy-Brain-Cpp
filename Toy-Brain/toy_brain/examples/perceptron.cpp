
#include <iostream>
#include "..\src\models.h"

using namespace std;
using namespace ToyBrain;

int main()
{
	Neuron percetron(2, Function::sigmoid);

	// Training data for OR function
	std::vector<std::vector<double>> inputs;
	std::vector<double> firstInput = { 1, 1 };
	inputs.push_back(firstInput);
	std::vector<double> secondInput = { 0, 1 };
	inputs.push_back(secondInput);
	std::vector<double> thirdInput = { 1, 0 };
	inputs.push_back(thirdInput);
	std::vector<double> fourthInput = { 0, 0 };
	inputs.push_back(fourthInput);

	std::vector<std::vector<double>> targets;
	std::vector<double> result1 = { 1 };
	std::vector<double> result2 = { 1 };
	std::vector<double> result3 = { 1 };
	std::vector<double> result4 = { 0 };
	targets.push_back(result1);
	targets.push_back(result2);
	targets.push_back(result3);
	targets.push_back(result4);
	//

	int num_epochs = 10;
	double target_error = 0.1;
	double learning_rate = 1.0;
	bool done = false;

	for (int i = 0; i < num_epochs && !done; i++) {
		done = true;
		for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
			double output = percetron.feed_forward(inputs[input_index]);

			if (round(output) != targets[input_index][0]) {
				double delta_error = targets[input_index][0] - output;
				percetron.updateWeights(delta_error, learning_rate, inputs[input_index]);

				done = false;
			}
		}
		if (done) {
			std::cout << "Trained in " << i << " epochs" << std::endl;
			break;
		}
	}

	std::cout << "Success" << std::endl;
	system("pause");

	return 0;
}
