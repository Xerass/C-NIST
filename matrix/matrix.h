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

// Lazy-resize a Matrix slot. If *m is NULL or has the wrong shape, frees it
// and reallocates. Buffer contents are NOT preserved across a resize.
// Used by layers to size their per-batch caches without allocating per call.
void    mat_ensure_shape(Matrix** m, int rows, int cols);

// --- Initialization ---
void    mat_randomize(Matrix* m, float min, float max);
void    mat_dropout(Matrix* m, Matrix* mask, float p);

// Fills every element with a constant value.
void    mat_fill(Matrix* m, float val);

// --- Basic Arithmetic (allocating, returns new matrix) ---
Matrix* mat_add(Matrix* a, Matrix* b);
Matrix* mat_sub(Matrix* a, Matrix* b);
Matrix* mat_hadamard(Matrix* a, Matrix* b);
Matrix* mat_scalar_mult(Matrix* m, float scalar);

// Add a (rows, 1) column-vector bias to every column of a (rows, cols) matrix.
// Broadcasts the bias across the batch dimension. Returns a new matrix.
Matrix* mat_add_bias(Matrix* m, Matrix* bias);

// Numerically stable column-wise softmax. Each column of the result is a
// probability distribution: subtracts the per-column max before exp.
// Used for the output layer of multi-class classifiers.
Matrix* mat_softmax_cols(Matrix* m);

// --- In-place / _into variants (zero allocation) ---
// These write into a pre-sized output (or modify the operand in place).
// Use these in hot paths to avoid malloc/free churn.

// out := m  (out must be the same shape as m)
void mat_copy_into(Matrix* out, Matrix* m);

// out := a * b   (out must be (a->rows, b->cols))
void mat_dot_into(Matrix* out, Matrix* a, Matrix* b);

// out := a^T * b (out must be (a->cols, b->cols))
void mat_dot_transposeA_into(Matrix* out, Matrix* a, Matrix* b);

// out := a * b^T (out must be (a->rows, b->rows))
void mat_dot_transposeB_into(Matrix* out, Matrix* a, Matrix* b);

// a := a * b  (element-wise, b unchanged, shapes must match)
void mat_hadamard_inplace(Matrix* a, Matrix* b);

// m := m * scalar
void mat_scalar_mult_inplace(Matrix* m, float scalar);

// m[i,j] += bias[i,0] for every column j (bias broadcast across batch)
void mat_add_bias_inplace(Matrix* m, Matrix* bias);

// out := sum of m along axis=1 (collapsed to a column). out shape: (m->rows, 1)
void mat_sum_rows_into(Matrix* out, Matrix* m);

// out[i] := func(m[i])  (shapes must match)
void mat_map_into(Matrix* out, Matrix* m, float (*func)(float));

// out := softmax_cols(m). Same numerical-stability trick as mat_softmax_cols.
void mat_softmax_cols_into(Matrix* out, Matrix* m);

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
