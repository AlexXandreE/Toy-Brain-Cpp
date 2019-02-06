// toy_brain.cpp : Defines the entry point for the application.
//

#include "toy_brain_main.h"
#include "src/models.h"

using namespace std;

int main()
{
	ActivationFunction function1(Function::sigmoid);

	Neuron first(2, Function::sigmoid);


	// TODO: Or operation test
	
	std::vector<double> inputs = { 0, 1 };

	cout << first << endl;
	cout << "Feed foward result: " << first.feed_forward(inputs) << endl;
	system("pause");

	return 0;
}
