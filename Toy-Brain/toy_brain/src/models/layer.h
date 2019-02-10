class Layer {
protected:
	std::vector<Neuron> neurons;
	Function activation_function;

public:
	Layer(int num_neurons, int number_of_inputs, Function activation_function);
	std::vector<double> feed_forward(std::vector<double> inputs);
	Function getErrorFunction() { return this->activation_function; };
	std::vector<Neuron> getMembers() { return this->neurons; };
};


class InputLayer : public Layer {

};

class OutputLayer : public Layer {

};