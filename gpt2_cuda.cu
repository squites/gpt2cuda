#include <iostream>

// ----- kernels -----

__global__ void encoder_v2(int B, int T, int C, 
                           float* __restrict__ embedding, const int* __restrict__ tokens,
                           const float* __restrict__ wte, const float* __restrict__ wpe ) {

   // const int block_id = blockIdx.x;
   // const int thread_id = threadIdx.x;  // thread id within the block
    const int n_blocks = gridDim.x;     // number of thread blocks (N diagram)
    const int n_threads = blockDim.x;

    // shared memory to pre-load tokens for a line. Allocated dynamically during kernel invocation
    extern __shared__ uint32_t* line_of_tokens;
    // points to the start of the tokens for athe current line for block
    const uint32_t* line_start_ptr = tokens + blockIdx.x * T;
    // when we're done processing current line, how far should we jump to get to the start of the next line of tokens for this block
    const size_t inc_next_line = T * n_blocks;
    // how many channels does a line of tokens generate
    const size_t channels_per_line = T * C
    // where is the start of the embedding buffer where we need to write the embedding for current thread within the current block
    float* e_ptr = embedding + blockIdx.x * C + threadIdx.x; // out
    // when we're done processing current line, how far should we jump to get to the start of the embedding buffer for the next line of
    // tokens for this thread in this block. Notice that we need to use this, we're already at the channel fo the last token in the current line
    const size_t inc_embedding_next_line = C * (n_blocks - 1);
    // offset from wte pointer for this thread, sice, no matter the token chose, this thread must always access the same channel in the weights
    const float* thread_wte_ptr = wte + threadIdx.x;

    for (int line = blockIdx.x; line < B; line += n_blocks) {
        // pre-load a line of tokens into shared memory to allow memory coalescing of token reads from global memory
        __syncthreads();
        for (int t = threadid; t < T; t += n_threads) {
            line_of_tokens[t] = line_start_ptr[t];
        }
        __syncthreads();

        const float* p_ptr = wpe + threadid;

        for (int t = 0; t < T; t++) {
            uint32_t token = line_of_tokens[t];
            const float* t_ptr = thread_wte_ptr + token * C;
            // memory access coalesced: all threads access contiguous global memory, reducing number of memory transactions
            __syncthreads();
            *e_ptr = *t_ptr + *p_ptr;
            e_ptr += C;
            p_ptr += C;
        }
        line_start_ptr += inc_next_line;
        e_ptr += inc_embedding_next_line;
    }

}


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
encoder (from yaikhom.com):
- gpt2 small has emb dim of 768. C=768. Dividing by a warp(32 threads), we have 768/32=24. so we need 24 warps. 32x24
- the ith thread of each thread block handles the ith emb. So, the thread 3 of each block, handles the embedding 3 of each token in the sequence
images:
    thread blocks               Batch of token lines
1  [][][][][][][][]             [][][][][][][][][][] 1
2  [][][][][][][][]             [][][][][][][][][][] 2
        ...                             ...
N  [][][][][][][][]             [][][][][][][][][][] B
   1 2            P             1 2               T

- In the "thread blocks", each row is a thread block, and each element of that row is a thread. Each thread will compute one embedding.
- In "Batch of token lines", each row is a batch (B), and each element of the row correspond to one token of that batch (T). Shape (B,T)

    Embeddings for the batch
1   [][][][][][][][][][][][]
2   [][][][][][][][][][][][]
              ...
T   [][][][][][][][][][][][]
    1 2                    C

- each row is a token, where each element of that row is an embedding value. So, token1 [emb1][emb2][emb3]...[embC]. Shape (T,C).
- each line of tokens is handled by a thread block of C threads. What this means is that, the first T rows, are line of tokens, and that T rows
are handled by a thread block.

    Token embedding weights               Positional encoding weights
1   [][][x][][][][][][][][][]            1 [][][][][][][][][][][][][][]
2   [][][x][][][][][][][][][]            2 [][][][][][][][][][][][][][]   
              ...                                   ...
V   [][][x][][][][][][][][][]         1023 [][][][][][][][][][][][][][]
    1 2                    C              1 2                       C

- in "token embedding weights", each row is a token, and each element is one embedding value. Shape (Vocab, C)
- in "Positional encoding weights", each row is a token, and each element is one embedding value. Shape (T, C)

- B(batch_size) is the number of token lines to process with each kernel launch.
- each token line consists of 1024 tokens
- N is the number of thread blocks in the grid when the kernel is launched
- each thread block consists of P threads
- We launch this kernel for each batch of token lines, and continue until all of the token lines have been processed.
- each thread in the block handles a channel value, so we have P = C = 768, threads per thread block.

- each token line (sequence of tokens) is processed by a thread block. All thread blocks in the grid processes multiple token lines simultaneously.
- when there's less thread blocks N in the grid than the number of token lines in the batch, each block processes more than 1 token line.
  For example, on the diagram, the 1st thread block is processing token lines 1 and N+1 in the batch. (Por isso as setas apontando para duas token lines)
- to complete a token line of T=1024 tokens per line, each thread needs to iterate 1024 times adjusting the addresses for each iteration inside
  embedding, wte, wpe.

- later, we'll see that reducing the number of threads per block can help with occupancy. The more active warps scheduled to all SMs, the higher
is the thread occupancy on the SMs, which means better throughput.


(from siboehm.com)
matmul: we'll use grid, block, thread hierarchy to assign each thread a unique entry in the RESULTING matrix C. Then, the thread will compute
the dot product of the corresponding row of A and column of B, and write the result to C.
- warp: is a group of 32 threads. A warp is assigned to a warp scheduler, which is the physical core that executes the instructions. There are
4 warp schedulers per multiprocessor.
- if we set blockDim to be multidimensional, then the threadId is calculated as: 
    threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
then threads with neighbouring threadId become part of the same warp.
- as sequential memory access by threads that are part of the same warp can be grouped and executed as one. (referred to global memory coalescing)
- global memory coalescing it's the most important thing to keep in mind when optimizing a kernel's GMEM memory access
The idea is to process threads that are part of the same warp to exploit coalescing

- threads consecutive can be of the same warp. These threads are those with consecutive threadIdx.x. But for the memory coalescing to happen,
the threadIdx.x needs to load the columns instead of the rows, because the rows are not consecutively. We can change how we assign positions 
in the resulting matrix C
*/