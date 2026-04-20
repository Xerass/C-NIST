#include "optimizers.h"
#include <math.h>
#include <stdlib.h>

//all funcs here modify the parameter matrices in place.


//SGD takes step by modifying the weight opposite the direction of the gradinet
//this variant however has momentum
void sgd_step(float* weights, const float* gradients, float* velocity, int size, float learning_rate, float momentum) {
    
    //we only update 2 things, the weights and the velocity

    //if momentum is less thatn 0 then it has no velocity and cannot keep going
    if (velocity == NULL || momentum <= 0.0f) {
        //in this case, just subtract lr * gradients to weights
        for (int i = 0; i < size; i++) {
            weights[i] -= learning_rate * gradients[i];
        }
    } else {
        //if we do have velocity, we need to update it first THEN subtract velocity to weigghts
        //same formula as before we just add momentum * velocity and we get the new velocity
        for (int i = 0; i < size; i++) {
            velocity[i] = momentum * velocity[i] + learning_rate * gradients[i];
            weights[i] -= velocity[i];
        }
    }
}


//Adaptive Movement Estimation (ADAM) is a lil more confusing
//instead of just applying one LR to all values, it has an adaptive LR for all parameters
//it keeps track of 2 things, the first is the momentum (m) and the second is the uncorrected velocity (v)
//m is first moment, v is for second moment, t is KEY, it keeps track of the number of iterations done. Beta1 and beta2 are decay rates for their respective moments, epsilon is just there to stop div by 0
 void adam_step(float* weights, const float* gradients, float* m, float* v, int size, int t, float learning_rate, float beta1, float beta2, float epsilon) {
    
    if (t <= 0) t = 1;
    

    //the first moment is directional matrix
    //we calc it by beta1 * m[i] + (1 - beta1) * gradients[i]
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    
    // Bias correction
    // correction_m = 1 - beta1^t
    // correction_v = 1 - beta2^t
    float correction_m = 1.0f - powf(beta1, (float)t);
    float correction_v = 1.0f - powf(beta2, (float)t);
    
    // equivalent optimized scaling: lr_t = learning_rate * sqrt(correction_v) / (correction_m)
    // weights -= lr_t * m / (sqrt(v) + epsilon_scaled)
    // but to match exactly: m_hat = m / correction_m, v_hat = v / correction_v
    for (int i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + one_minus_beta1 * gradients[i];
        v[i] = beta2 * v[i] + one_minus_beta2 * gradients[i] * gradients[i];
        
        float m_hat = m[i] / correction_m;
        float v_hat = v[i] / correction_v;
        
        weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}
