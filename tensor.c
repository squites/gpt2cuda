#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "./tensor.h"

Tensor create_tensor(int ndim, int *dims) {
    Tensor tensor;
    tensor.ndim = ndim;
    tensor.dims = (int*)malloc(ndim * sizeof(int));
    tensor.size = 1;

    for (int i = 0; i < ndim; i++) {
        tensor.dims[i] = dims[i];
        tensor.size *= dims[i]; // calculate total number of elements
    }
    tensor.data = (float*)malloc(tensor.size * sizeof(float));

    return tensor;
}

// Initialization
void init_zeros(Tensor *t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 0.0f;
    }
}

void init_ones(Tensor *t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = 1.0f;
    }
}

void init_random(Tensor *t) {
    srand(time(NULL));
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)(rand() / RAND_MAX); // random value between 0 and 1
    }
}

// Arithmetic Ops: (add, sub, mul, div)
void add_tensor(Tensor *t1, Tensor *t2, Tensor *out, int size) {
    assert(t1->size == size && t2->size == size);
    for (int i = 0; i < size; i++) {
        out->data[i] = t1->data[i] + t2->data[i];
    }
}

void mul_tensor(Tensor *t1, Tensor *t2, Tensor *out, int size) {
    assert(t1->size == size && t2->size == size && out->size == size);
    for (int i = 0; i < size; i++) {
        out->data[i] = t1->data[i] * t2->data[i];
    }
}

/*
// CPU matmul
void matmul_tensor(Tensor *t1, Tensor *t2, Tensor *out, int size, int dim) {
    float sum = 0.0f;
    if (dim == 3) {
        assert(t1->dims[2] == t2->dims[0]);
    } else {
        assert(t1->dims[1] == t2->dims[0]);
        for (int i = 0; i < t1->dims[0]; i++) {
            for (int j = 0; j < t2->dims[1]; j++) {
                //sum += t1->data[i] * t2->data[j];
                for (int k = 0; k < t2->dims[0]; j++) {
                    sum += t1->data[i*j+k] * 
                }
            }
            out->data[i] = sum;
        } 
    }   
}*/

int main() {
    // example test of creating a initializing a Tensor
    int dims[3] = {3,3,2};
    Tensor tensor1 = create_tensor(3, dims);
    init_ones(&tensor1);
    Tensor tensor2 = create_tensor(3, dims);
    init_ones(&tensor2);
    Tensor out = create_tensor(3, dims);
    add_tensor(&tensor1, &tensor2, &out, out.size);
    
    for (int i = 0; i < out.size; i++) {
        printf("%f ", out.data[i]);
        if ((i+1) % dims[2] == 0) printf("\n");
    }



    free(tensor1.data);
    free(tensor1.dims);

    free(tensor2.data);
    free(tensor2.dims);

    free(out.data);
    free(out.dims);

    return 0;
}