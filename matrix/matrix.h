#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Macro to access 1D array as 2D: matrix[row][col]
#define MAT_AT(m, r, c) (m)->nodes[(r) * (m)->cols + (c)]

// Core Matrix Structure
typedef struct {
    int rows;
    int cols;
    float *nodes; 
} Matrix;

// --- Memory Management ---
Matrix* mat_create(int rows, int cols);
void    mat_free(Matrix *mat);
Matrix* mat_copy(Matrix* m);

// --- Initialization ---
void    mat_randomize(Matrix* m, float min, float max);
void    mat_dropout(Matrix* m, Matrix* mask, float p);

// --- Basic Arithmetic ---
Matrix* mat_add(Matrix* a, Matrix* b);
Matrix* mat_sub(Matrix* a, Matrix* b);
Matrix* mat_hadamard(Matrix* a, Matrix* b);
Matrix* mat_scalar_mult(Matrix* m, float scalar);

// --- Neural Network Specific Math ---

// Standard Dot Product: Z = W * A
Matrix* mat_dot(Matrix* a, Matrix* b);

// Error Gradient: dA_prev = W^T * dZ
// Effectively multiplies the transpose of A by B
Matrix* mat_dot_transposeA(Matrix* a, Matrix* b);

// Weight Gradient: dW = dZ * A_prev^T
// Effectively multiplies A by the transpose of B
Matrix* mat_dot_transposeB(Matrix* a, Matrix* b);

// Apply activation functions or their derivatives
Matrix* mat_map(Matrix* m, float (*func)(float));

// Sum along rows (axis 1) resulting in a column vector: (rows, 1)
Matrix* mat_sum_rows(Matrix* m);

#endif // MATRIX_H