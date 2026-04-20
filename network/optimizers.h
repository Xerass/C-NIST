#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

// Decoupled SGD with Momentum step
// v = momentum * v + learning_rate * gradients
// weights -= v
// If velocity is NULL, performs standard SGD without momentum: weights -= learning_rate * gradients
void sgd_step(float* weights, const float* gradients, float* velocity, int size, float learning_rate, float momentum);

// Decoupled Adam step
// m = beta1 * m + (1-beta1) * gradients
// v = beta2 * v + (1-beta2) * (gradients * gradients)
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// weights -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
void adam_step(float* weights, const float* gradients, float* m, float* v, int size, int t, float learning_rate, float beta1, float beta2, float epsilon);

#endif // OPTIMIZERS_H
