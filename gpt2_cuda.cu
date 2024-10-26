#include <iostream>

// ----- kernels -----

__global__ void encoder(int B, int T, int C, const uint32_t* tokens, float* out, 
                        const float* __restrict__ wte, const float* __restrict__ wpe) {
    
    // points to the start of the tokens for the current line
    int* line_start = tokens + blockIdx.x * T;
    size_t next_line = T * C;
    // start of embedding buffer that we need to write in. The embedding of current thread within current block
    float* e_ptr = out + blockIdx.x * C + threadIdx.x;
    // how far to jump to get the start of the emb buffer for the next line
    const size_t emb_next_line = C*(gridDim.x-1);
    // offset from wte pointer for this thread
    const float* thread_wte_ptr = wte + threadIdx.x;

    for (int line = blockIdx.x; line < B; line += gridDim.x) {
        const float *p_ptr = wpe + threadIdx.x;
        for (int t = 0; t < T; t++) {
            int token = line_start[t];
            const float *t_ptr = thread_wte_ptr + token * C;
            // memory coalesced
            __syncthreads();
            *e_ptr = *t_ptr + *p_ptr;
            e_ptr += C;
            p_ptr += C;
        }
        line_start += next_line;
        e_ptr += emb_next_line;
    }
}


__global__ void matmulv2(int B, int T, int C, int outC, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict bias, float* out) {
    int C3 = C*3;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B*T && col < C3) {
        float sum = 0.0f;
        for (int i = 0; i < C; i++) {
            sum += x[row*C+i] * w[i*C3+col];
        }
        out[row*C3+col] = sum;
    }
}

__global__ void matmul(int B, int T, int C, int outC, int N,
                       const float __restrict__ *x, const float __restrict__ *w, const float __restrict__ *bias, 
                       const float *out) {
    
    int b = blockIdx.x;
    int t = threadIdx.x;
    int oc = blockIdx.y;

    if (b < B && t < T && oc < outC) {
        float sumval = (bias == NULL) ? 0.0f : bias[oc];
        for (int c = 0; c < C; c++) {
            sumval += x[b*T*C + t*C + c] * w[c + oc*C];
        }
        out[b*T*outC + t*outC + oc] = sumval;

    }


    int ix = blockIdx.x * blockDim.x + threadIdx.x; // col
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // row

    float sumval = 0.0f;
    if (ix < N && iy < N) {
        for (int i = 0; i < N; i++) {
            sumval += x[iy*N+i] * w[i*N+ix];
        }
        out[]
    } 

}

/* NOTES:
matmul: we'll use grid, block, thread hierarchy to assign each thread a unique entry in the RESULTING matrix C. Then, the thread will compute
the dot product of the corresponding row of A and column of B, and write the result to C.

*/