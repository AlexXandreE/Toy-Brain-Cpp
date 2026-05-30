#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <span>

double computeError(std::span<double> output, std::span<double> target);

#endif // !HELPER_FUNCTIONS_H
