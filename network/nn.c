#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h" 
#include "activations.h"

//we'd like to seprate the work if possible to isolate issues
//so we'll create a per layer basis pass and a full network pass


//outputs the post-activation matrix
Matrix* layer_forward(Layer* layer, Matrix* input){
    //cache the input and remove prev caches (i love saving memory)
    //checks if prev layer exists, free's that prev and copies the new input
    if (layer->A_prev) mat_free(layer->A_prev);
    layer->A_prev = mat_copy(input);

    //actual work
    //computes Z = W * A_prev(prev layer) + b
    Matrix* WA = mat_dot(layer->W, layer->A_prev);
    //free if Z already exists
    if (layer->Z) mat_free(layer->Z);
    layer->Z = mat_add(WA, layer->b);

    //lastly pass through activation
    if (layer->A) mat_free(layer->A);
    switch (layer->act_type) {
        case ACT_SIGMOID: layer->A = mat_map(layer->Z, sigmoid); break;
        case ACT_RELU:    layer->A = mat_map(layer->Z, relu); break;
        default:          layer->A = mat_copy(layer->Z); break; // ACT_LINEAR
    }
    return layer->A;
}


//for network forward, we simply call forward sequentially from first to last
Matrix* network_forward(Network* net, Matrix* input) {
    Matrix* current_activation = input;
    
    for (int i = 0; i < net->num_layers; i++) {
        current_activation = layer_forward(net->layers[i], current_activation);
    }
    
    return current_activation; 
}

//layer level backward pass
Matrix* layer_backward(Layer* layer, Matrix* dA){
    //input from previous activation derivative

    //Calculate derivative of the activation function: g'(Z)
    //this matrix is the dl/dA
    Matrix* dZ_activation = NULL;
        switch (layer->act_type) {
        case ACT_SIGMOID: dZ_activation = mat_map(layer->Z, sigmoid_prime); break;
        case ACT_RELU:    dZ_activation = mat_map(layer->Z, relu_prime); break;
        default:
            //linear derivative is just 1          
            dZ_activation = mat_create(layer->Z->rows, layer->Z->cols);
            for(int i = 0; i < dZ_activation->rows * dZ_activation->cols; i++) 
                dZ_activation->nodes[i] = 1.0f;
            break;
    }

    //dZ = dA * g'(Z) this is the local error (change of Z with repect to activaion and its current activation)
    //technically a dot product but simplifies to hadamard since our matrices are the same size and external activations do not affect our Z (so diagonal of jacobian are the only non 0)
    Matrix* dZ = mat_hadamard(dA, dZ_activation);
    mat_free(dZ_activation);

    //compute weight gradients dW = dZ * A_prev^T (transposition to make dot work)
    if (layer->dW) mat_free(layer->dW);
    layer->dW = mat_dot_transposeB(dZ, layer->A_prev);

    //compute bias gradients db = sum(dZ) along rows (axis 1) (bias grad is simple its literally just summation)
    if (layer->db) mat_free(layer->db);
    layer->db = mat_sum_rows(dZ);

    //compute dA_prev = W^T * dZ (pass error back to previous layer)
    Matrix* dA_prev = mat_dot_transposeA(layer->W, dZ);
    mat_free(dZ);

    return dA_prev;
}


//same network logic for network_backward
void network_backward(Network* net, Matrix* loss_gradient) {
    Matrix* current_dA = mat_copy(loss_gradient);
    
    for (int i = net->num_layers - 1; i >= 0; i--) {
        Matrix* next_dA = layer_backward(net->layers[i], current_dA);
        mat_free(current_dA);
        current_dA = next_dA;
    }
    
    mat_free(current_dA); 
}