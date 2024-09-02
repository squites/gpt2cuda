#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float *data;
    int *dims;
    int ndim;
    int size;
} Tensor;

Tensor create_tensor(int ndim, int *dims);
void init_zeros(Tensor *t);
void init_ones(Tensor *t);
void init_random(Tensor *t);
void add_tensor(Tensor *t1, Tensor *t2, Tensor *out, int size);
void mul_tensor(Tensor *t1, Tensor *t2, Tensor *out, int size);


#endif