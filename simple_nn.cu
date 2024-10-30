// taking a step back and implementing a simple neural network
// MNIST dataset images(28x28)
// MLP: input_layer(784 neurons), hidden1(300 neurons), hidden2(100 neurons), output_layer(10 neurons)

// output of a single neuron: x0*w0 + x1*w1 + ... + xn*wn + bias
// dotproduct between the input vector * weight vector + bias
// output of all neurons: stack inputs into a matrix @ weight matrix + bias
/*
x0 * w0
x1 * w1
x2 * w2
... ...
xn * wn
 ^  
784
*/

typedef struct {
    linear(784, 784, 784, )

} Model;

__global__ void matmul_forward_k(float* x, float *w, float* bias, float* out, 
                                 int N, int B, int outC) {
    // x(B,N): input, 
    // w(N,outC): weights,
    // bias(outC): bias,
    // out(B,outC): out tensor
    // N: width. number of columns of x and rows of w
    // B: batch size. Number of rows of x
    // outC: number of cols in out tensor
    
    int col = blockIdx.x * blockDim.x + threadIdx.x; // thread col ix
    int row = blockIdx.y * blockDim.y + threadIdx.y; // thread row ix

    if (row < B && col < outC) {
        out[row * outC + col] = bias[col]; // indexing row * number_of_cols + col, to get the row-th and col-th element
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x[i * outC + col] * w[row * N + i];
        }
        out[row * outC + col] += sum;
    }
}

__global__ void relu_forward_k(int B, int outC, float* actv, float* out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < B && col < outC) {
        out[row*outC+col] = actv[row*outC+col] > 0.0f ? actv[row*outC+col] : 0.0f;
    }

}