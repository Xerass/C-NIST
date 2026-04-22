#include <stdio.h>
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
//solely here for testing
#include <assert.h>

//MACRO for ease of writing later down the line
//purpose is to access 1D array as a 2D matrix safely and cleanly
#define MAT_AT(m, r, c) (m)->nodes[(r) * (m)->cols + (c)]

// what we need:
// we need a matrix struct to streamline all operations

/*
    As for the Neural Network itself, we would need a couple of operations
    
    Z = output
    w = weight
    a = activation
    b = bias
    A = error

    Forward Pass would require matrix mult
    :: based on the formula Z = w * a + b 

    func sig: Matrix* matmul(Matrix* a, Matrix*b)
    
    ::Bias addition would require an add
    func sig: Matrix* mataddbias(Matrix* a, Matrix *bias)

    ::activation func is simply applying a function to each element, effectively a map
    ::we can make this function agnostic
    func sig: void matrix_activation(Matrix* a, double (*func)(double))

    Backward pass will require us to use jacobians to compute the gradients

    ::matsub used for error calc, specifically for direction, actual loss is calculated separately
    ::interestingly some functions fully canel out, meaning the derivateive ends up just being A - Y or a mat_sub
    ::we can take advantage of this!

    ::Hadamard product as a proxy for shorter dot product jacobians
    ::This will handle the chain rule and transpositions for us

    #skip transposition, 1D arrays can easily be handled
    func sig: Matrix* mat_dot_transposeB(Matrix* a, Matrix* b)
    -> represents: dW = dZ * Aprev^T (weight gradient)

    func sig: Matrix* mat_dot_transposedA(Matrix* a, Matrix* b)
    -> represents: dAprev = W^T * dZ (error gradient), literal punishment on nodes based on activation value change

    ::Scalar multipy to handle learning rate
    func sig: Matrix* mat_scalar_mult(Matrix* a, float scalar)
*/

// Matrix struct is defined in matrix.h

//=================Memory stuff============
//create mat
Matrix* mat_create(int rows, int cols){
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    //calloc for safe 0 defaults
    mat->nodes = (float*)calloc(rows * cols, sizeof(float));
    return mat;
}

void mat_free(Matrix *mat){
    if (mat != NULL){
        if (mat->nodes != NULL){
            free(mat->nodes);
        }
        free(mat);
    }
}

//we will need this to save the activation states for backprop
Matrix* mat_copy(Matrix* m) {
    Matrix* out = mat_create(m->rows, m->cols);
    // memcpy for literal memeory chunk copy
    memcpy(out->nodes, m->nodes, m->rows * m->cols * sizeof(float));
    return out;
}

//=========initializations===================
void mat_randomize(Matrix* m, float min, float max){
    //randomizes initial weights
    int total_nodes = m->rows * m->cols;
    for (int i = 0; i < total_nodes; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        m->nodes[i] = min + r * (max - min);
    }
}

void mat_dropout(Matrix* m, Matrix* mask, float p) {
    if (p <= 0.0f) {
        for (int i = 0; i < mask->rows * mask->cols; i++) mask->nodes[i] = 1.0f;
        return;
    }
    if (p >= 1.0f) {
        for (int i = 0; i < m->rows * m->cols; i++) {
            m->nodes[i] = 0.0f;
            mask->nodes[i] = 0.0f;
        }
        return;
    }

    float scale = 1.0f / (1.0f - p);
    int total = m->rows * m->cols;
    for (int i = 0; i < total; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < p) {
            m->nodes[i] = 0.0f;
            mask->nodes[i] = 0.0f;
        } else {
            m->nodes[i] *= scale;
            mask->nodes[i] = 1.0f;
        }
    }
}


//================Operations==================

//Matrix Addition (usually for bias)
Matrix* mat_add(Matrix* a, Matrix* b){
    //future uses of assert is for safety
    assert(a->rows == b->rows && a->cols == b->cols);

    Matrix* out = mat_create(a->rows, a->cols);

    int total_nodes = a->rows * a->cols;
    for (int i = 0; i < total_nodes; i++){
        out->nodes[i] = a->nodes[i] + b->nodes[i];
    }

    return out;
}

// sometimes for error / Gradient Subtraction
Matrix* mat_sub(Matrix* a, Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    
    Matrix* out = mat_create(a->rows, a->cols);
    
    int total_nodes = a->rows * a->cols;
    for (int i = 0; i < total_nodes; i++) {
        out->nodes[i] = a->nodes[i] - b->nodes[i];
    }
    return out;
}

// used for local Gradients / activation derivatives
Matrix* mat_hadamard(Matrix* a, Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    
    Matrix* out = mat_create(a->rows, a->cols);
    
    int total_nodes = a->rows * a->cols;
    for (int i = 0; i < total_nodes; i++) {
        out->nodes[i] = a->nodes[i] * b->nodes[i];
    }
    return out;
}

//used for scaling with learning rate
Matrix* mat_scalar_mult(Matrix* m, float scalar){
    Matrix* out = mat_create(m->rows, m->cols);
    int total_nodes = m->rows * m->cols;
    for (int i = 0; i < total_nodes; i++) {
        out->nodes[i] = m->nodes[i] * scalar;
    }
    return out;
}


//==========cool stuff================

//dot product for forward pass / linear transform
Matrix* mat_dot(Matrix* a, Matrix* b) {
    //the fundamental rule of matrix multiplication
    assert(a->cols == b->rows); 
    
    //output matrix shape is (Rows of A) x (Cols of B)
    Matrix* out = mat_create(a->rows, b->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
    return out;
}

//we now have some useful gradients at hand (the final matrices)

/*
    To define: do note since we do not have the nabla symbol we need to use d to denote gradient 
    though they are completely differnt things (gradients being vectors of partial derivatives of a scalar value function (the final loss (an altitude within the gradient) is the scalar value))
    -The full formula looks like this Loss = 
    dA is final activation gradient
    dZ is the per layer node gradient
    -these two are simply here to make the math work out
    -these are transient errors that get passed but wont result in the final output
    dW is the weight gradient from the weight layers
    db is the bias gradient from the bias layers
    -these two are the ones that actually get used to update the weights and biases
*/


//backprop with activation (error gradient) dAprev = W^T * dZ
Matrix* mat_dot_transposeA(Matrix* a, Matrix* b) {
    assert(a->rows == b->rows); 
    Matrix* out = mat_create(a->cols, b->cols);
    
    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->rows; k++) {
                sum += MAT_AT(a, k, i) * MAT_AT(b, k, j); // Note the inverted k, i
            }
            MAT_AT(out, i, j) = sum;
        }
    }
    return out;
}

//backprop with weights (weight gradient) dW = dZ * Aprev^T
Matrix* mat_dot_transposeB(Matrix* a, Matrix* b) {
    assert(a->cols == b->cols); 
    Matrix* out = mat_create(a->rows, b->rows);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, j, k); // Note the inverted j, k
            }
            MAT_AT(out, i, j) = sum;
        }
    }
    return out;
}

// used for applying any function (usually activations) onto the elements in a mat
Matrix* mat_map(Matrix* m, float (*func)(float)) {
    Matrix* out = mat_create(m->rows, m->cols);
    int total_nodes = m->rows * m->cols;
    for (int i = 0; i < total_nodes; i++) {
        out->nodes[i] = func(m->nodes[i]);
    }
    return out;
}

Matrix* mat_sum_rows(Matrix* m) {
    Matrix* out = mat_create(m->rows, 1);
    for (int i = 0; i < m->rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            sum += MAT_AT(m, i, j);
        }
        MAT_AT(out, i, 0) = sum;
    }
    return out;
}

