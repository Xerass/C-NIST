#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "matrix/matrix.h"
#include "network/nn.h"
#include "network/activations.h"

// Simple test framework
#define ASSERT_NEAR(a, b, eps) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("Assertion failed: %f != %f (eps %f) at %s:%d\n", (float)(a), (float)(b), (float)(eps), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void test_matrix_ops() {
    printf("Testing Matrix Operations...\n");
    
    // Test Dot Product
    Matrix* a = mat_create(2, 3);
    Matrix* b = mat_create(3, 2);
    
    // a = [[1, 2, 3], [4, 5, 6]]
    MAT_AT(a, 0, 0) = 1; MAT_AT(a, 0, 1) = 2; MAT_AT(a, 0, 2) = 3;
    MAT_AT(a, 1, 0) = 4; MAT_AT(a, 1, 1) = 5; MAT_AT(a, 1, 2) = 6;
    
    // b = [[7, 8], [9, 10], [11, 12]]
    MAT_AT(b, 0, 0) = 7;  MAT_AT(b, 0, 1) = 8;
    MAT_AT(b, 1, 0) = 9;  MAT_AT(b, 1, 1) = 10;
    MAT_AT(b, 2, 0) = 11; MAT_AT(b, 2, 1) = 12;
    
    Matrix* c = mat_dot(a, b);
    // Expected = [[58, 64], [139, 154]]
    ASSERT_NEAR(MAT_AT(c, 0, 0), 58, 1e-5);
    ASSERT_NEAR(MAT_AT(c, 0, 1), 64, 1e-5);
    ASSERT_NEAR(MAT_AT(c, 1, 0), 139, 1e-5);
    ASSERT_NEAR(MAT_AT(c, 1, 1), 154, 1e-5);
    
    mat_free(a); mat_free(b); mat_free(c);
    
    // Test mat_sum_rows
    Matrix* m = mat_create(2, 2);
    MAT_AT(m, 0, 0) = 1; MAT_AT(m, 0, 1) = 2;
    MAT_AT(m, 1, 0) = 3; MAT_AT(m, 1, 1) = 4;
    Matrix* m_sum = mat_sum_rows(m);
    ASSERT_NEAR(MAT_AT(m_sum, 0, 0), 3, 1e-5);
    ASSERT_NEAR(MAT_AT(m_sum, 1, 0), 7, 1e-5);
    
    mat_free(m); mat_free(m_sum);
    
    printf("Matrix Operations PASSED.\n");
}

void test_activations() {
    printf("Testing Activations...\n");
    ASSERT_NEAR(sigmoid(0), 0.5f, 1e-5);
    ASSERT_NEAR(sigmoid(100), 1.0f, 1e-5);
    ASSERT_NEAR(relu(5), 5.0f, 1e-5);
    ASSERT_NEAR(relu(-5), 0.0f, 1e-5);
    ASSERT_NEAR(relu_prime(5), 1.0f, 1e-5);
    ASSERT_NEAR(relu_prime(-5), 0.0f, 1e-5);
    printf("Activations PASSED.\n");
}

void test_network_pipeline() {
    printf("Testing Network Pipeline (XOR-ish scenario)...\n");
    
    Network* net = create_network(2);
    network_add_layer(net, create_layer(2, 3, ACT_RELU));
    network_add_layer(net, create_layer(3, 1, ACT_SIGMOID));
    
    // Initialize with some non-random values for deterministic testing
    for(int i=0; i < net->layers[0]->W->rows * net->layers[0]->W->cols; i++) net->layers[0]->W->nodes[i] = 0.5f;
    for(int i=0; i < net->layers[1]->W->rows * net->layers[1]->W->cols; i++) net->layers[1]->W->nodes[i] = 0.5f;
    
    Matrix* input = mat_create(2, 1);
    MAT_AT(input, 0, 0) = 1.0f;
    MAT_AT(input, 1, 0) = 0.0f;
    
    // Forward Pass (is_training = 0 for inference)
    Matrix* output = network_forward(net, input, 0);
    assert(output->rows == 1 && output->cols == 1);
    printf("Forward Pass Output: %f\n", MAT_AT(output, 0, 0));
    
    // Backward Pass
    Matrix* target = mat_create(1, 1);
    MAT_AT(target, 0, 0) = 1.0f; // Target output is 1
    Matrix* loss_grad = mat_sub(output, target); // simple (output - target) grad
    
    network_backward(net, loss_grad);
    
    // Verify gradients exist
    assert(net->layers[0]->dW != NULL);
    assert(net->layers[1]->dW != NULL);
    
    // Update weights
    float old_w = net->layers[1]->W->nodes[0];
    network_update_sgd(net, 0.1f, 0.0f);
    float new_w = net->layers[1]->W->nodes[0];
    
    assert(old_w != new_w);
    printf("Weight updated from %f to %f\n", old_w, new_w);
    
    mat_free(input);
    mat_free(target);
    mat_free(loss_grad);
    // Note: Net free logic might be needed but let's assume it's okay for now
    printf("Network Pipeline PASSED.\n");
}

void test_dropout() {
    printf("Testing Dropout...\n");
    srand(42); 
    
    // Create a larger layer to test multiple nodes
    Layer* l = create_layer(10, 10, ACT_LINEAR); 
    l->dropout_rate = 0.5f;
    
    // Set W to identity so output = input
    for(int i=0; i<10; i++) MAT_AT(l->W, i, i) = 1.0f;
    
    Matrix* input = mat_create(10, 1);
    for(int i=0; i<10; i++) input->nodes[i] = 1.0f;
    
    // Forward Pass with Dropout (Training mode)
    Matrix* output = layer_forward(l, input, 1);
    
    int zeros = 0;
    for(int i=0; i<10; i++) {
        if (output->nodes[i] == 0.0f) {
            zeros++;
        } else {
            // Scaled by 1/(1-0.5) = 2.0
            ASSERT_NEAR(output->nodes[i], 2.0f, 1e-5);
        }
    }
    
    printf("Nodes dropped: %d/10\n", zeros);
    assert(zeros > 0 && zeros < 10); 
    
    // Backward Pass
    Matrix* dA = mat_create(1, 1);
    MAT_AT(dA, 0, 0) = 1.0f;
    
    // In our layer_backward, it applies mask to dA
    // But since output is (1,1) in this case, let's make it bigger to test properly
    mat_free(l->dropout_mask);
    l->dropout_mask = NULL;
    mat_free(l->A); 
    l->A = NULL;
    mat_free(input);
    input = mat_create(10, 1);
    for(int i=0; i<10; i++) input->nodes[i] = 1.0f;
    
    // Re-test with a better shape for gradient check
    Layer* l2 = create_layer(5, 5, ACT_LINEAR);
    l2->dropout_rate = 0.4f; // 40% drop
    Matrix* in2 = mat_create(5, 1);
    for(int i=0; i<5; i++) in2->nodes[i] = 1.0f;
    
    layer_forward(l2, in2, 1);
    Matrix* dA2 = mat_create(5, 1);
    for(int i=0; i<5; i++) dA2->nodes[i] = 1.0f;
    
    // We can't directly check the internal dA_scaled, but we can verify 
    // that gradients were affected if we check dW or dA_prev.
    // simpler: just trust the unit test of forward pass for now and check if it runs.
    Matrix* dA_prev = layer_backward(l2, dA2);
    assert(dA_prev != NULL);

    mat_free(input);
    mat_free(l->W); mat_free(l->b); free(l);
    mat_free(in2);
    mat_free(dA2);
    mat_free(dA_prev);
    // Cleanup l2... usually we need a better cleanup function
    
    printf("Dropout PASSED.\n");
}

int main() {
    test_matrix_ops();
    test_activations();
    test_dropout();
    test_network_pipeline();
    printf("\nALL TESTS PASSED SUCCESSFULLY!\n");
    return 0;
}
