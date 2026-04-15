#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// --- Activation Functions ---
float sigmoid(float x);
float relu(float x);
// You can easily add more here later (e.g., float tanh_act(float x);)

// --- Derivatives (for Backprop) ---
float sigmoid_prime(float x);
float relu_prime(float x);

#endif // ACTIVATIONS_H