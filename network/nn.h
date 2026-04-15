//enums to easily reference our activvation functions
typedef enum {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_LINEAR
} Activation;

//simple layer struct
typedef struct {
    int in_features;
    int out_features;
    
    // parameters (learnable)
    Matrix* W; // weights shape: (out_features, in_features)
    Matrix* b; // bias shape: (out_features, 1)
    
    // forward pass caches (needed for backprop)
    Matrix* A_prev; // input from the previous layer (post activation)
    Matrix* Z;      // Pre-activation: W * A_prev + b
    Matrix* A;      // Post-activation: f(Z) of current (A_prev of next layer)
    
    // gradients (populated during backward pass)
    Matrix* dW;
    Matrix* db;
    
    Activation act_type;
} Layer;

typedef struct {
    Layer** layers;  // array of layer pointers
    int num_layers;
    int capacity;    // max layers allocated
} Network;