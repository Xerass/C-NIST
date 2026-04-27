#include <stdio.h>
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// MACRO for ease of writing later down the line
// purpose is to access 1D array as a 2D matrix safely and cleanly
#define MAT_AT(m, r, c) (m)->nodes[(r) * (m)->cols + (c)]

// =================Memory stuff============
// create mat
Matrix* mat_create(int rows, int cols){
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    // calloc for safe 0 defaults
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

// we will need this to save the activation states for backprop
Matrix* mat_copy(Matrix* m) {
    Matrix* out = mat_create(m->rows, m->cols);
    memcpy(out->nodes, m->nodes, m->rows * m->cols * sizeof(float));
    return out;
}

// =========initializations===================
void mat_randomize(Matrix* m, float min, float max){
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

// ================Operations==================

// Matrix Addition (usually for bias when shapes match exactly)
Matrix* mat_add(Matrix* a, Matrix* b){
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

// used for scaling with learning rate
Matrix* mat_scalar_mult(Matrix* m, float scalar){
    Matrix* out = mat_create(m->rows, m->cols);
    int total_nodes = m->rows * m->cols;
    for (int i = 0; i < total_nodes; i++) {
        out->nodes[i] = m->nodes[i] * scalar;
    }
    return out;
}

// adds a (rows, 1) bias to every column of a (rows, cols) matrix.
// this is the broadcasted version of mat_add specifically for biases:
// with batching, Z = W*A + b needs b broadcasted across the batch.
Matrix* mat_add_bias(Matrix* m, Matrix* bias) {
    assert(bias->rows == m->rows && bias->cols == 1);

    Matrix* out = mat_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        float bi = MAT_AT(bias, i, 0);
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = MAT_AT(m, i, j) + bi;
        }
    }
    return out;
}

// numerically stable column-wise softmax.
// each column is one sample, softmax each column independently.
// trick: subtract the column max before exp() to avoid overflow,
// since softmax(x) == softmax(x - c) for any constant c.
Matrix* mat_softmax_cols(Matrix* m) {
    Matrix* out = mat_create(m->rows, m->cols);

    for (int j = 0; j < m->cols; j++) {
        // find max of this column for numerical stability
        float max_val = MAT_AT(m, 0, j);
        for (int i = 1; i < m->rows; i++) {
            float v = MAT_AT(m, i, j);
            if (v > max_val) max_val = v;
        }

        // exp(x - max) and accumulate sum
        float sum = 0.0f;
        for (int i = 0; i < m->rows; i++) {
            float e = expf(MAT_AT(m, i, j) - max_val);
            MAT_AT(out, i, j) = e;
            sum += e;
        }

        // normalize, with a guard against pathological sum=0
        if (sum <= 0.0f) sum = 1e-12f;
        float inv = 1.0f / sum;
        for (int i = 0; i < m->rows; i++) {
            MAT_AT(out, i, j) *= inv;
        }
    }
    return out;
}

// ==========cool stuff================

// dot product for forward pass / linear transform
Matrix* mat_dot(Matrix* a, Matrix* b) {
    assert(a->cols == b->rows);

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

// backprop with activation (error gradient) dAprev = W^T * dZ
Matrix* mat_dot_transposeA(Matrix* a, Matrix* b) {
    assert(a->rows == b->rows);
    Matrix* out = mat_create(a->cols, b->cols);

    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->rows; k++) {
                sum += MAT_AT(a, k, i) * MAT_AT(b, k, j);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
    return out;
}

// backprop with weights (weight gradient) dW = dZ * Aprev^T
Matrix* mat_dot_transposeB(Matrix* a, Matrix* b) {
    assert(a->cols == b->cols);
    Matrix* out = mat_create(a->rows, b->rows);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, j, k);
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


// =============================================================
// Lazy-resize utility for layer caches
// =============================================================

void mat_ensure_shape(Matrix** m, int rows, int cols) {
    if (*m != NULL && (*m)->rows == rows && (*m)->cols == cols) return;
    if (*m != NULL) mat_free(*m);
    *m = mat_create(rows, cols);
}

void mat_fill(Matrix* m, float val) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) m->nodes[i] = val;
}

// =============================================================
// In-place / _into variants
// =============================================================

void mat_copy_into(Matrix* out, Matrix* m) {
    assert(out->rows == m->rows && out->cols == m->cols);
    memcpy(out->nodes, m->nodes, (size_t)out->rows * (size_t)out->cols * sizeof(float));
}

void mat_dot_into(Matrix* out, Matrix* a, Matrix* b) {
    assert(a->cols == b->rows);
    assert(out->rows == a->rows && out->cols == b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
}

void mat_dot_transposeA_into(Matrix* out, Matrix* a, Matrix* b) {
    assert(a->rows == b->rows);
    assert(out->rows == a->cols && out->cols == b->cols);

    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->rows; k++) {
                sum += MAT_AT(a, k, i) * MAT_AT(b, k, j);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
}

void mat_dot_transposeB_into(Matrix* out, Matrix* a, Matrix* b) {
    assert(a->cols == b->cols);
    assert(out->rows == a->rows && out->cols == b->rows);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += MAT_AT(a, i, k) * MAT_AT(b, j, k);
            }
            MAT_AT(out, i, j) = sum;
        }
    }
}

void mat_hadamard_inplace(Matrix* a, Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    int n = a->rows * a->cols;
    for (int i = 0; i < n; i++) a->nodes[i] *= b->nodes[i];
}

void mat_scalar_mult_inplace(Matrix* m, float scalar) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) m->nodes[i] *= scalar;
}

void mat_add_bias_inplace(Matrix* m, Matrix* bias) {
    assert(bias->rows == m->rows && bias->cols == 1);
    for (int i = 0; i < m->rows; i++) {
        float bi = MAT_AT(bias, i, 0);
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) += bi;
        }
    }
}

void mat_sum_rows_into(Matrix* out, Matrix* m) {
    assert(out->rows == m->rows && out->cols == 1);
    for (int i = 0; i < m->rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            sum += MAT_AT(m, i, j);
        }
        MAT_AT(out, i, 0) = sum;
    }
}

void mat_map_into(Matrix* out, Matrix* m, float (*func)(float)) {
    assert(out->rows == m->rows && out->cols == m->cols);
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) out->nodes[i] = func(m->nodes[i]);
}

void mat_softmax_cols_into(Matrix* out, Matrix* m) {
    assert(out->rows == m->rows && out->cols == m->cols);

    for (int j = 0; j < m->cols; j++) {
        // per-column max for numerical stability
        float max_val = MAT_AT(m, 0, j);
        for (int i = 1; i < m->rows; i++) {
            float v = MAT_AT(m, i, j);
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int i = 0; i < m->rows; i++) {
            float e = expf(MAT_AT(m, i, j) - max_val);
            MAT_AT(out, i, j) = e;
            sum += e;
        }

        if (sum <= 0.0f) sum = 1e-12f;
        float inv = 1.0f / sum;
        for (int i = 0; i < m->rows; i++) {
            MAT_AT(out, i, j) *= inv;
        }
    }
}
