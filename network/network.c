#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "nn.h" 
#include "activations.h"

//first define HOW to create a layer
//layer requirements, should have an input feat, output feat, some means to store wrights and biases and their gradients
//and also forward pass caches

Layer* create_layer(int in_features,int out_features,Activation activation) {
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

    return l;
}

//all we need to define are the max amount of layers
Network* create_network(int capacity){
    Network* net = (Network*)malloc(sizeof(Network));
    net->capacity = capacity;
    net->num_layers = 0;
    net->layers = (Layer**)malloc(capacity * sizeof(Layer*));
    return net;
}

//add a method to input layers into the empty network
void network_add_layer(Network* net, Layer* l){
    if (net->num_layers < net->capacity){
        net->layers[net->num_layers++] = layer;
    } else {
        printf("Error! Layer exceeded capacity")
    }
}

