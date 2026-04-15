#include <math.h>
#include "activations.h"

//sigmod defined as 1/(1+e^-x)
float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

//sigmoid derivative for backprop purposes
// (1 + e ^-x) -1
// -1(1 + e^-x)-2 * (-e^-x)
//so it just ends up becoming sigmoid(x) * (1 - sigmoid(x))
float sigmoid_prime(float x){
    return sigmoid(x) * (1 - sigmoid(x));
}

//ReLu (rectified linear unit) 
//much simpler, just a piecewise func

float relu(float x){
    return x > 0.0f ? x : 0.0f;
}

//derivative is again, just 1 for consztants
float relu_prime(float x){
    return x > 0.0f ? 1.0f : 0.0f;
}

//even simpler, return as is (control)
float linear(float x){
    return x;
}

float linear_prime(float x){
    return 1.0f;
}