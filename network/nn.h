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
    
    // momentum / adam state caching
    Matrix* m_W;
    Matrix* v_W;
    Matrix* m_b;
    Matrix* v_b;
    
    Activation act_type;
} Layer;

typedef struct {
    Layer** layers;  // array of layer pointers
    int num_layers;
    int capacity;    // max layers allocated
} Network;

// ========================
// Network & Layer Creation
// ========================
Layer* create_layer(int in_features, int out_features, Activation activation);
Network* create_network(int capacity);
void network_add_layer(Network* net, Layer* l);

// Optimizers Wrapper Functions
void layer_update_sgd(Layer* layer, float learning_rate, float momentum);
void network_update_sgd(Network* net, float learning_rate, float momentum);

void layer_update_adam(Layer* layer, int t, float learning_rate, float beta1, float beta2, float epsilon);
void network_update_adam(Network* net, int t, float learning_rate, float beta1, float beta2, float epsilon);
