#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "./tensor.h"

// allocate a tensor in memory
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
    srand(time(0));
    for (int i = 0; i < t->size; i++) {
        t->data[i] = (float)rand() / (float)RAND_MAX; // random value between 0 and 1
    }
}

// Arithmetic Ops: (add, mul)
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

// CPU matmul (2d tensor only for now!)
void matmul_tensor(Tensor *t1, Tensor *t2, Tensor *out) {
    assert(t1->dims[1] == t2->dims[0]);
    for (int i = 0; i < t1->dims[0]; i++) {
        for (int j = 0; j < t2->dims[1]; j++) {
            for (int k = 0; k < t1->dims[1]; k++) {
                out->data[i * (t2->dims[1]) + j] += t1->data[i*(t1->dims[1])+k] * t2->data[k*(t2->dims[1])+j]; 
            }
        }
    } 
} 

void print_tensor(Tensor *t, int size, int *dims) {
    for (int i = 0; i < size; i++) {
        printf("%f ", t->data[i]);
        if ((i+1) % dims[1] == 0) printf("\n");
    }
}

// CUDA matmul
/*
#define TILE_W 2
__global__ void matmul_tensor_k(Tensor *t1, Tensor *t2, Tensor *out, int width) {
    __shared__ float tile1[TILE_W][TILE_W];
    __shared__ float tile2[TILE_W][TILE_W];

    // simplify variables
    int tx = threadIdx.x; ty = threadIdx.y;
    int bx = blockIdx.x; by = blockIdx.y;

    // identify row and col of the out element to work on
    int row = by * TILE_W + ty;
    int col = bx * TILE_W + tx;

    // accumulate the results
    float Pvalue = 0;
    // loop over tile1 and tile2(faces) required to compute out
    for (int i = 0; i < width/TILE_W; i++) {
        tile1[ty][tx] = t1[row*width + i*TILE_W+tx];
        tile2[ty][tx] = t2[(i*TILE_W + ty) * width + col];
        __syncthreads();
    
        // compute dot-product
        for (int k = 0; k < TILE_W; k++) {
            Pvalue += tile1[ty][k] * tile2[k][tx];
        }
        __syncthreads();
    }
    out[row * width + col] = Pvalue;
}
*/

int main() {
    // example test of creating a initializing a Tensor
    int dims[] = {4,4};
    Tensor tensor1 = create_tensor(2, dims);
    init_ones(&tensor1);
    Tensor tensor2 = create_tensor(2, dims);
    init_random(&tensor2);
    Tensor out = create_tensor(2, dims);
    init_zeros(&out);
    //add_tensor(&tensor1, &tensor2, &out, out.size);

    matmul_tensor(&tensor1, &tensor2, &out);

    printf("tensor 1:\n");
    print_tensor(&tensor1, tensor1.size, dims);
    printf("tensor 2:\n");
    print_tensor(&tensor2, tensor2.size, dims);
    printf("tensor out:\n");
    print_tensor(&out, out.size, dims);


    free(tensor1.data);
    free(tensor1.dims);

    free(tensor2.data);
    free(tensor2.dims);

    free(out.data);
    free(out.dims);

    return 0;
}