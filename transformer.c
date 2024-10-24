#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "tensor.c"
#include "tokenizer.c"
#include "helpers.c"

#define BATCH_SZ 4 // B
#define BLOCK_SZ 8 // T
#define EMBD_SZ 4  // C

// ----- GPT2 -----
// gpt_small: 124M 
typedef struct {
    int block_size = 256,   // gpt_small:1024 (actual gpt2 config)
    int vocab_size = 65,    // gpt_small:50257  
    int n_layers = 6,       // gpt_small:12
    int n_heads = 6,        // gpt_small:12
    int n_embd = 384,       // gpt_small:768 / gpt_medium:1024 / gpt_large:1280/ gpt_extra_large:1600
} Config;

typedef struct {
    // embedding
    float *wte; // (vocab_size, n_embd) same as: (vocab_size, C)
    float *wpe; // (block_size, n_embd) same as: (T, C)
    // layernorm
    float *lw; // (n_embd,) same as: (C,)
    float *lb; // (n_embd,) same as: (C,)
    // attention
    //float *wQ;  // (n_embd, n_embd) same as: (C, C)
    //float *wK;  // (n_embd, n_embd) same as: (C, C)
    //float *wV;  // (n_embd, n_embd) same as: (C, C)
    float *QKVw;  // should be just one instead of wQ,wK,wV. qkv should be stacked on top of each other
    float *QKVb;  
} Parameters;

typedef struct {
    // embedding
    float *encoding; // (B,T,C)
    // layernorm
    float *lactv;       // (B,T,C)
    // attention
    float *qkv;      // (B,T,T)
} Activations;


// move this somewhere else (another file)
char *read_file(char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) return NULL;
    fseek(fp, 0, SEEK_END); // moves the file pointer (fp) to the end of the file
    int file_len = ftell(fp); // ftell returns the current value of the position indicator, giving us the number of characters in the file
    fseek(fp, 0, SEEK_SET); // moves fp back to the beginning of the file

    char *string = malloc(sizeof(char) * (file_len+1)); // len + '\0' 
    // read each char from the file
    char c;
    int i = 0;
    while ((c = fgetc(fp)) != EOF) { //fgetc: reads the next char of the file
        string[i] = c;
        i++;
    }
    string[i] = '\0';
    fclose(fp);

    return string;
}

// ----- Forward pass functions -----
// Combine token embedding vector + positional embedding vector to encode each token
void encoder(int B, int T, int C, float *wte, float *wpe, int *in, float *out) {
    for (int b = 0; b < B; b++) { // loop over batches
        for (int t = 0; t < T; t++) { // loop over each token in the batch
            // get token value on a specific index
            int token_value = in[b * T + t]; // this gets the token value, not its position. That's why we don't use on pos_emb
            // gets the token embedding of that token
            float *token_embd = wte + token_value * C;
            // gets the positional embedding of that token
            float *pos_embd = wpe + t * C;
            for (int c = 0; c < C; c++) {
                out[(b * T + t) * C + c] = token_embd[c] + pos_embd[c]; // add token_embd with pos_embd of that specific token.
            }
        }
    }
}
// The general formula is that if you want to retrieve any element b,t,c, you compute the offset into Storage as b*T*C + t*C + c
// Layer normalization over the embedding dimmension for each token in the sequence. For each token, calculate the mean and variance over the token embedding vector. In the end, each token of each batch will have it's own mean and variance values.
void layernorm(int B, int T, int C, float *in, float *out, float weight, float bias) { // gamma: weights, beta: bias (PyTorch)
    const float eps = 1e-5;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {               
            float *in_ptr = in + b * T * C + t * C; // b*T*C: skips over the entire batch, because the batch contains T*C elements. 
                                                    // t*C: skips over the elements of the token sequence and its embeddings.
            // calculate the mean
            float mean = 0.0f;
            for (int c = 0; c < C; c++) {
                mean += in_ptr[c];  // + c: this access the 'c' embedding of token   
            }
            mean = mean/C;

            // calculate the variance
            float var = 0.0f;
            for (int c = 0; c < C; c++) {
                var += (in_ptr[c] - mean)*(in_ptr[c] - mean); // (x-mean)?2 -> pow(embs[c]-mean, 2);
            }
            var = var/C;

            // calculate the square-root of variance+epsilon
            float sq = 1.0f / sqrtf(var + eps);

            // normalize (this works for each embedding value. We normalize each embedding value of the emb vector for each token)
            float *out_ptr = out + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                float x = sq * (in[c]-mean); // not working the right way, not sure why.
                // scale and shift
                out_ptr[c] = x * weight + bias;
            }
            // should I pass the gamma and beta as arrays? Then cache them to be easier to optimize ... 
        }
    }
}

// 2nd version of matmul
void matmul(const float* x, const float* w, const float* bias, float* out, 
               int B, int T, int C, int outC) {
    // Basically 1st layer is (B,T,C) @ (C,4*C), only on MLP
    // MLP has 2 layers ((B,T,C)@(C,C*4) = (B,T,C*4)) and ((B,T,C*4)@(C*4,C) = (B,T,C))
    float sumval = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int timeIdx = b * T + t; // get each row
            for (int oc = 0; oc < outC; oc++) {
                if (bias == NULL) sumval = 0.0f;
                else sumval = bias[oc];
                for (int c = 0; c < C; c++) {
                    sumval += x[timeIdx * C + c] * w[c + oc * C];
                }
                out[timeIdx*outC+oc] = sumval;
            }
        }
    }
}

/*
// (REMOVE) - initialize random matrix projections for Q,K,V 
float *init_rand_proj(int row, int col) {
    float *out = (float*)malloc(row * col * sizeof(float));
    for (int i = 0; i < row*col; i++) {
        out[i] = ((float)rand() / (float)RAND_MAX);
    }
    return out;
}
*/

// 2d transpose considering batch dim. Only transposes dims (-1,-2). (copy or change the original matrix)
void transpose(float *m, float *m_transpose, int B, int row, int col) { // (B,row,col) -> (B,col,row)
    for (int b = 0; b < B; b++) { // not tested with multiple batches yet
        // how to do the indexing right
        for (int i = 0; i < col; i++) {
            for (int j = 0; j < row; j++) {
                m_transpose[i * row + j] = m[j * col + i]; // I need to do m_transpose[i][j] = m[j][i]; This formula is right?
            }
        }
    }
}

// converts input to 0-1 values summing to 1. Turns to be a vector of probabilities. Input: (B,T,T). Softmax applies only on the last dimension
void softmax(int B, int T, int C, float *logits, float *out) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // ptr to get each b,t,c element
            float *p = logits + b * T * C + t * C;
            float *pout = out + b * T * C + t * C; 
            float sum = 0.0f;
            float max_val = p[0];
            
            // for numerical stability, find the max_val to subtract them when expf
            for (int c = 1; c < C; c++) {
                if (p[c] > max_val) max_val = p[c];
            }
            // expf and calculate sum
            for (int c = 0; c < C; c++) {
                // exponentiate p[c]
                pout[c] = expf(p[c] - max_val);
                // sums up to get the total value
                sum += pout[c];   
            }
            // divide by the sum and store in 'pout' tensor
            for (int c = 0; c < C; c++) {                
                pout[c] /= sum;
                //out[b*T*C+t*C+c] = pout[c] * (1/sum);
            }
        }
    }
}

// important: the input of this attention layer is the output of a linear layer, where it generates a tensor (B,T,C*3), where the C*3 dim contains wQ,wK,wV matrices.
void multihead_attention(int B, int T, int C, int NHEADS, 
                         float *qkv, float *out, float *before_soft, float *softmax_scores) {
    // in shape is (B,T,C)
    // notes llm.c parameters:
    // - qkv(B,T,C*3): contains the query,key,value vectors. 3C is the concatenated Q,K,V for each token
    // - before_soft(B,NH,T,T): pre attention scores. Holds the dotproduct between queries and keys, before is sent to softmax (unormalized)
    // - softmax_scores(B,NH,T,T): post attention scores. Stores the attention weights after the softmax
    // - out(B,T,C): holds the resulting tensor

    int C3 = C*3;
    int head_size = C/NHEADS;  

    for (int b = 0; b < B; b++) { // loop over batches
        for (int t = 0; t < T; t++) { // loop over time
            for (int h = 0; h < NHEADS; h++) { // loop over heads
                // 1) split into query, key, value vectors + split into heads (this query is for the token t on head h)
                float *query = qkv + b*T*C3 + t*C3 + h*head_size; // h*head_size is the offset for each head
                float *preatt = before_soft + b*NHEADS*T*T + h*T*T + t*T;
                float *att_scores = softmax_scores + b*NHEADS*T*T + h*T*T + t*T;

                // 2) scoring
                float maxval = 0.0f; //preatt[0];
                //for (int tok = 0; tok < T; tok++) { // this should be tok <= t?
                for (int tok = 0; tok <= t; tok++) {
                    float *key = qkv + b*T*C3 + tok*C3 + C + h*head_size; // gets the key vector of token 'tok' of head h
                    float val = 0.0f;
                    // since its divided by heads, instead of the whole C, now we loop through C/N_HEADS, which is 'head_size' 
                    for (int i = 0; i < head_size; i++) {
                        val += query[i] * key[i]; // multiply the query_i with all keys_j
                    }
                    // scale
                    // float scale = 1/sqrtf(head_size);
                    // val *= scale;
                    val *= (1/sqrtf(head_size));
                    if (val > maxval) maxval = val;
                    //} else {
                        // mask
                        //val = 0.0f;
                    //}
                    preatt[tok] = val;
                }

                // 3) softmax
                float sum = 0.0f;
                for (int tok = 0; tok <= t; tok++) { //< T; tok++) {
                    float expon = expf(preatt[tok] - maxval); // maxval for numerical stability
                    sum += expon;
                    att_scores[tok] = expon;
                }

                // 3.1) normalize
                for (int tok = 0; tok < T; tok++) {
                    if (tok <= t) att_scores[tok] *= (1.0f/sum);
                    else att_scores[tok] = 0.0f; // mask
                }

                // 4) aggregate
                float *outp = out + b*T*C + t*C + h*head_size;
                // initialize with 0s
                for (int i = 0; i < head_size; i++) {
                    outp[i] = 0.0f;
                }
                // accumulate
                for (int tok = 0; tok <= t; tok++) {
                    float *value = qkv + b*T*C3 + tok*C3 + (C*2) + h*head_size;
                    for (int i = 0; i < head_size; i++) {
                        outp[i] += att_scores[i] * value[i];
                    }
                }
            }
        }
    }
}

/*
void causal_self_attn(int B, int T, int C, float *wQ, float *wK, float *wV, float *in, float *out, float *bias) {
    // 1) for each input token create a query,key,value vectors by multiplying inputs by weight matrices wQ,wK,wV
    // 2) multiply (dot prod.) current query vector, by only key vectors of previous tokens, to get the score of how well they match
    // 3) multiply the scores by the value vectors, and then sum up
    // 4) projection
    float *query = (float*)malloc(B * T * C * sizeof(float));
    float *key   = (float*)malloc(B * T * C * sizeof(float));
    float *value = (float*)malloc(B * T * C * sizeof(float));
    // calculate query,key,value matrices
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *ix = in + b * T * C + t * C;
            float q,k,v = 0.0f;
            // aggregate the values of each embedding
            for (int cw = 0; cw < C; cw++) {
                for (int c = 0; c < C; c++) {
                    // wQ,wK,wV are the projections
                    q += ix[c] * wQ[c * C + cw]; // (B,T,C)@(C,C) -> (B,T,C)
                    k += ix[c] * wK[c * C + cw];
                    v += ix[c] * wV[c * C + cw];
                }
            }
            for (int i = 0; i < C; i++) {
                query[b*T*C+t*C+i] = q;
                key[b*T*C+t*C+i] = k;
                value[b*T*C+t*C+i] = v;
            }
        }
    }

    // compute attention scores query@key.T -> (B,T,C)@(B,C,T) = (B,T,T)
    float *attn_matrix = (float*)malloc(B * T * T * sizeof(float));   
    float *transpose_key = (float*)malloc(B * C * T * sizeof(float));    
    float attn_val = 0.0f;
    // transpose key matrix
    transpose(key, transpose_key, B, C, T); // (B,T,C) -> (B,C,T)
    // compute attention scores
    float d = 1.0f / sqrtf(C);
    for (int b = 0; b < B; b++) {
        for (int tx = 0; tx < T; tx++) {
            for (int ty = 0; ty < T; ty++) {
                float attn_val = 0.0f;
                for (int c = 0; c < C; c++) {
                    // accumulate the dot product
                    attn_val += query[b*T*C+tx*C+c] * transpose_key[b*C*T+c*T+ty];
                }
                // mask the resulting matrix with tril(attn_matrix) (maybe we don't need to create a function just for that)
                if (tx < ty) attn_val = 0.0f;
                // divide the value by sqrtf(d)
                attn_val = attn_val * d;
                // store into attn_matrix
                attn_matrix[b*T*T+tx*T+ty] = attn_val;
            }
            // softmax
            // should softmax be called here? Or should I call inside the model layers
            
        }
    }
    free(query); free(key); free(value);
    free(attn_matrix); free(transpose_key);
}
*/

// skip connection where "input" is the original input, and "layer_out" is the out tensor of the layer that we're adding
void residual_stream(int B, int T, int C, float *input, float *layer_out, float *out) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                int ix = b * T * C + t * C + c;
                out[ix] = input[ix] + layer_out[ix];
            }
        }
    }
}

// #define M_PI 3.14159265358979323846 (defined on math.h)
// GELU(x) = 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x?3)))      (more efficient! Simpler!) 
void GELU_aprox(float *x, float *out, int n) {  // refine after!
    for (int i = 0; i < n; i++) {
        float x3 = x[i] * x[i] * x[i];  // (x?3)
        float a = x[i] + 0.044715 * x3; // (x + 0.044715x?3)
        float sqroot = sqrt(2 / M_PI);  // (sqrt(2/pi))
        float b = 1 + tanh(sqroot);     // (1 + tanh(sqrt(2/pi)
        out[i] = 0.5 * x[i] * b * a;
    }
}

// function to calculate the loss over the softmax probabilities
void cross_entropy_loss(float* loss, float* x, int* y) {
    // ex: x[1, 2, 3, 4] 
    //     y[3, 4, 5, 1]
    // Loss(x,y) = -sum(xi * ln(yi))
    
}

// ----- Backward pass functions ----- 
//
// ...

// ----- utils -----
// split dataset tokens into train/test (90,10)%
void split_data(int *tokens, int sz, int *train, int *test, int train_sz, int test_sz) {
    for (int i = 0; i < train_sz; i++) {
        train[i] = tokens[i];
    }
    for (int i = 0; i < test_sz; i++) {
        test[i] = tokens[train_sz + i];
    }
}

// get a random batch of data
void get_batch(char *split, int *train_data, int *test_data, int data_sz, int *x, int *y,
               int batch_sz, int block_sz) {
    int *data = (split = "train") ? train_data : test_data;
    int ix;
    //int ixs[batch_sz]; // use this instead if I want to keep track of the random indices

    // generate (batch_sz) random indices 
    srand(time(NULL));
    for (int i = 0; i < batch_sz; i++) { 
        ix = (rand() % (data_sz - block_sz));
        //ixs[i] = ix;
        printf("ix:%d\n", ix);
        for (int j = 0; j < block_sz; j++) {
            x[i * block_sz + j] = data[ix + j]; // this is stacking each block_sz row.
            y[i * block_sz + j] = data[ix + j + 1];
        }
    }
}

int main() {
    int B = BATCH_SZ;
    int T = BLOCK_SZ;
    int C = EMBD_SZ;
    
    char *filename = "input/input.txt";
    char *input = read_file(filename);
    if (input == NULL) {
        printf("Error reading file!\n");
        return 1;
    }
    
    // get vocab from input string and sort. For character tokenization we don't actually need a vocab. But eventually we'll implement a more efficient tokenization (BPE)
    char *vocab = get_vocab(input);
    int vocab_sz = strlen(vocab); // (65)
    qsort(vocab, vocab_sz, sizeof(char), compare);
    printf("vocab:\n%s\n%d\n", vocab, vocab_sz); // 1st char of vocab is '\n'

    // encode tokens
    char *str = "First Citizen: We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them Let us revenge this with our pikes, ere we become rakes: for the gods know I ";
    //char *str = input; // input: passing the whole file
    //char *small_str; 
    //strncpy(small_str, str, 1000); small_str[1000] = '\0';
    int input_sz = strlen(str); // n
    printf("n: %d\n", input_sz);
    int *tokens = (int*)malloc(input_sz * sizeof(int));
    encode_tokens_v2(str, tokens, vocab, input_sz); // (input_sz)
    printf("tokens length: %d\n", input_sz);

    // train/test split
    int split = (int)(0.9*input_sz); // (split=90%) and (n-split=10%)
    int *train_data = (int*)malloc(split * sizeof(int)); 
    int *test_data = (int*)malloc((input_sz-split) * sizeof(int));
    int train_sz = split, test_sz = input_sz-split;
    split_data(tokens, input_sz, train_data, test_data, train_sz, test_sz);
    //print_tokens(train_data, train_sz); print_tokens(test_data, test_sz);
    
    // generate a random batch of data (inputs: x, target: y)
    int *x = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int)); // (B, T)
    int *y = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int));
    get_batch("train", train_data, test_data, train_sz, x, y, BATCH_SZ, BLOCK_SZ); // remember to do this multiple times throughout the dataset. Here's doing a single time, it works, but it has to do for the whole dataset
    /* (DEBUG)
    for (int i = 0; i < BATCH_SZ; i++) {
        printf("Batch %d:\n", i + 1);
        printf("x: ");
        for (int j = 0; j < BLOCK_SZ; j++) {
            printf("%d ", x[i * BLOCK_SZ + j]);
        }
        printf("\n");
        printf("y: ");
        for (int j = 0; j < BLOCK_SZ; j++) {
            printf("%d ", y[i * BLOCK_SZ + j]);
        }
        printf("\n");
    }
    */

    // token + positional embeddings
    float *wte = init_tok_emb_matrix(vocab_sz, C); // (65, 4). (vocab_sz, C)
    float *wpe = init_pos_emb_matrix(T, C);      // (8, 4).    (T, C)

    //print_lookup_table("token", wte, vocab_sz, C);
    //print_lookup_table("pos", wpe, T, C);

    float *output_emb = (float*)malloc(B * T * C * sizeof(float)); // (B, T, C): batches, blocksize, n_embd
    int *tokens_id = x; // need to do to 'y' as well?
    encoder(B, T, C, wte, wpe, tokens_id, output_emb); // (B, T, C)
    
    // Print the result for the first token in the first batch (for demonstration purposes) 
    /*for (int j = 0; j < T; j++) {
        printf("\nEmbedding for the each T token in the first batch:\n");
        for (int i = 0; i < C; i++) {
            printf("%f ", output_emb[j * C + i]);
        }
        printf("\n");
    }
    printf("\n");
    */

    // self-attention
    float *wQ = init_rand_proj(C, C);
    float *wK = init_rand_proj(C, C);
    float *wV = init_rand_proj(C, C);

    // implement a function to free all allocated memory. Also implement a function to allocate all necessary memory before hand, to get more compact and neat.
    // Free the allocated memory
    free(wte);
    free(wpe);
    free(output_emb);
    
    free(input);
    free(tokens);
    free(train_data);
    free(test_data);
    free(x); free(y);

    // self-attention free
    free(wQ); free(wK); free(wV);


    return 0;
}

/*
TODO:
- allocate all necessary memory before hand, into a single place
- implement a simple DataLoader
- group all the Net parameters into a struct
- do BPE encoding instead of character-level encoding. I guess that's where the sentence padding comes in.

Notes of self-attn:
- 1) create q,k,v vectors (multiply input tokens by wQKV (C,C*3), generating a vector (C*3), then split into 3 vectors of size C, vector q, k, v)
- 2) split each vector into attention heads. Each vector was size (C), now we split that vector into a matrix of (n_heads, C/n_heads)
    - 2.1) basically, each head will have a vector of size (C/n_heads). One row of the matrix is the vector of that head, for each q,k,v "vector matrix" (NOT THE MATRIX wQ, wK, wV)
- 3) score: generate scores on each head, by multiplying the q vector by all k vectors
- 4) sum: then multiply each value with its score, then sum them up, producing the result of self-attention (C/n_heads) for one of the heads
- 5) merge the attention heads: concatenate each score vector of each head, generating a a big vector of size (C)
- 6) projecting: before sending this vector for the next sublayer, we multiply the vector size (C) by wO(C, C). We need a 4th weight matrix of size (C, C) to project.
Doing all these, we have produced the vector that we can send to the next layer, which is the Feed Forward NN layer.

---- FLOW of the data -----
    * embeddings: generate token emb + pos emb, resulting in a (B,T,C) tensor
    * layernorm: the (B,T,C) tensor goes to layernorm, where it normalizes over the embeddings of each sequence,
                 resulting in (B,T,C) tensor.
    * linear: the normalized tensor (B,T,C) goes to a linear layer of (C,C*3). This is the wQ,wK,wV matrices, they're
              are glued together. So, in the linear layer we're muliplying the normalized input by the weight matrices
              (B,T,C) @ (C,C*3) resulting in (B,T,C*3). That's going to be the input in the attention block
    * Attention: 1) we split the (B,T,C*3) input into query(B,T,C) key(B,T,C) and value(B,T,C). So split by 3 
                    (NOT n_heads. n_heads is to use inside attention, which is the Number of attention heads)
                 2) split query,key,value into n_heads. Ex: if query(B,T,C=768) and we want 12 attention heads, if we
                    divide 768/12 = 64, so we'll have a matrix (12x64) where each row is a head.

Notes:
- IMPORTANT!
    big understanding: when storing the tensor in the memory, only its embeddings are being stored, and not the tokens as well. 
    I though it was stored as: [t0, e0, e1, t1, e0, e1,...]. In reality is [e0,e1,e0,e1] (just the embeddings)
- remember that each token_sequence will generate a query, key, value vector. One for each sequence 

*/