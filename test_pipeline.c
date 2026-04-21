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
    
    // Forward Pass
    Matrix* output = network_forward(net, input);
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

int main() {
    test_matrix_ops();
    test_activations();
    test_network_pipeline();
    printf("\nALL TESTS PASSED SUCCESSFULLY!\n");
    return 0;
}
