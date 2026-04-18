#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h" 
#include "activations.h"
#include "optimizers.h"

// ========================
// Network & Layer Creation
// ========================

Layer* create_layer(int in_features, int out_features, Activation activation) {
    Layer* l = (Layer*)malloc(sizeof(Layer));

    //init feats
    l->in_features = in_features;
    l->out_features = out_features;
    l->act_type = activation;
    
    //init weights and biases
    l->W = mat_create(out_features, in_features);
    l->b = mat_create(out_features, 1);

    //init caches to null for now they will be populated
    l->A_prev = NULL;
    l->Z = NULL;
    l->A = NULL;
    l->dW = NULL;
    l->db = NULL;

    //caches for momentum per layer
    l->m_W = NULL;
    l->v_W = NULL;
    l->m_b = NULL;
    l->v_b = NULL;

    return l;
}

Network* create_network(int capacity){
    Network* net = (Network*)malloc(sizeof(Network));
    net->capacity = capacity;
    net->num_layers = 0;
    net->layers = (Layer**)malloc(capacity * sizeof(Layer*));
    return net;
}

void network_add_layer(Network* net, Layer* l){
    if (net->num_layers < net->capacity){
        net->layers[net->num_layers++] = l;
    } else {
        printf("Error! Layer exceeded capacity\n");
    }
}

// ========================
// Forward & Backward Passes
// ========================

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

// optimization functions

//uses SGD with momentum
//given v = velocity , dW = weight gradient, lr = learning rate, m = momentum
//the formuala operates as:
//v = m * v + dW
//W = w - lr * v
//
void layer_update_sgd(Layer* layer, float learning_rate, float momentum) {
    //simple safety check, if no gradients, no updates
    if (layer->dW == NULL || layer->db == NULL) return;
    
    //if momentum is above zero, we meed to create the velocity matrices (so we dont just alloc when momentum is not yet notable)
    if (momentum > 0.0f) {
        //also check if velocity caches do exist
        if (layer->v_W == NULL) layer->v_W = mat_create(layer->W->rows, layer->W->cols);
        if (layer->v_b == NULL) layer->v_b = mat_create(layer->b->rows, layer->b->cols);
    }

    //just calc total number of nodes to apply the update to
    int size_W = layer->W->rows * layer->W->cols;
    //pass this to the actual function
    sgd_step(layer->W->nodes, layer->dW->nodes, momentum > 0.0f ? layer->v_W->nodes : NULL, size_W, learning_rate, momentum);

    //same logic but for bias
    int size_b = layer->b->rows * layer->b->cols;
    sgd_step(layer->b->nodes, layer->db->nodes, momentum > 0.0f ? layer->v_b->nodes : NULL, size_b, learning_rate, momentum);
}

//simply call per layer the SGD optimizer
void network_update_sgd(Network* net, float learning_rate, float momentum) {
    for (int i = 0; i < net->num_layers; i++) {
        layer_update_sgd(net->layers[i], learning_rate, momentum);
    }
}


//adam optimizer
void layer_update_adam(Layer* layer, int t, float learning_rate, float beta1, float beta2, float epsilon) {
    //same guardrails
    if (layer->dW == NULL || layer->db == NULL) return;

    //adam requires per layer momentum / velocity (m)
    //and it also needs RMSProp (v)
    //so we check if they exist, if not we create them
    if (layer->m_W == NULL) layer->m_W = mat_create(layer->W->rows, layer->W->cols);
    if (layer->v_W == NULL) layer->v_W = mat_create(layer->W->rows, layer->W->cols);
    if (layer->m_b == NULL) layer->m_b = mat_create(layer->b->rows, layer->b->cols);
    if (layer->v_b == NULL) layer->v_b = mat_create(layer->b->rows, layer->b->cols);

    //same logic as before, calc total nodes
    int size_W = layer->W->rows * layer->W->cols;
    //pass to the actual function
    adam_step(layer->W->nodes, layer->dW->nodes, layer->m_W->nodes, layer->v_W->nodes, size_W, t, learning_rate, beta1, beta2, epsilon);

    int size_b = layer->b->rows * layer->b->cols;
    adam_step(layer->b->nodes, layer->db->nodes, layer->m_b->nodes, layer->v_b->nodes, size_b, t, learning_rate, beta1, beta2, epsilon);
}

//apply adam update to every layer
void network_update_adam(Network* net, int t, float learning_rate, float beta1, float beta2, float epsilon) {
    for (int i = 0; i < net->num_layers; i++) {
        layer_update_adam(net->layers[i], t, learning_rate, beta1, beta2, epsilon);
    }
}
