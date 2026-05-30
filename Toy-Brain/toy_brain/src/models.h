#pragma once

#ifndef MODELS_H
#define MODELS_H

#include <stdlib.h> 
#include <random>
#include <optional>
#include <iostream>
#include <math.h>
#include <limits>
#include <cmath>
#include <vector>

// Include all model components from models/ folder
#include "models/activation_function.h"
#include "models/neuron.h"
#include "models/layer.h"
#include "models/multilayer_perceptron.h"

using namespace ToyBrain;

namespace ToyBrain {

	class ErrorFunction {
	private:
		Function function;
	public:
		ErrorFunction();
		ErrorFunction(Function type);
		double compute(double value);
	};
}

#endif //  MODELS_H
