// matmul using CUDA

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ void matmuk_k(int *a, int *b, int *c, int N) {
    // calculating global row and col for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    int tmp = 0;
    if (row < N && col < N) {
        // accumulate partial result
        for (int i = 0; i < N; i++) {
            tmp += a[row * N + i] * b[i * N + col];
        }
        // write back the result
        c[row * N + col] = tmp;
    }
}

// Initializes a matrix with random numbers 0-100
void init_matrix(int *m, int N) {
    for (int i = 0; i < N*N; i++) {
        m[i] = rand() % 100;
    }
}

// Verify the result on CPU
void verify_result(int *a, int *b, int *c, int N) {
    // c will have our resulting matrix from kernel
    int tmp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }

            // check each result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main() {
    // Set the square matrix dims (2?10 x 2?10)
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // allocate memory for the matrices
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize data (random for the example)
    init_matrix(a, N);
    init_matrix(b, N);

    // set our block and grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // setup the kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    // launch kernel
    matmul_k<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // veryfy the result
    verify_result(a, b, c, N);

    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl

    return 0;
}