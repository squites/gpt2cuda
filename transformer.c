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

/*
typedef struct {
    int block_size = 256, // 1024 (actual gpt2 config)
    int vocab_size = 65,  // 50257  
    int n_layer = 6,      // 12
    int n_head = 6,       // 12
    int n_embd = 384,     // 768
} GPTConfig;
*/

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

// Combine token embedding vector and positional embedding vector, encoding each token
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
void layernorm(int B, int T, int C, float *in, float *out, float gamma, float beta) { // gamma: weights, beta: bias (PyTorch)
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
                out_ptr[c] = x * gamma + beta;
            }
            // should I pass the gamma and beta as arrays? Then cache them to be easier to optimize ... 
        }
    }
}

// simple cpu matmul calculation to compare with CUDA version
void matmul_cpu(float *m, float *n, float *out, int row_m, int col_m, int col_n) {
    float value = 0.0f;
    for (int i = 0; i < row_m; i++) {
        for (int j = 0; j < col_n; j++) {
            //out[i*col_n+j] = 0.0f;
            for (int k = 0; k < col_m; k++) {
                //out[i*col_n+j] += m[i*col_m+k] * n[k*col_n+j];
                value += m[i*col_m+k] * n[k*col_n+j];
            }
            out[i*col_n+j] = value;
        }
    }
}

// separate matmul into 2 functions so we can precisely calculate the time only in the matmul computation.
//void call_matmul_cpu(float *m, float *n) {
    // TODO: 
//}

// initialize random matrix projections for Q,K,V
float *init_rand_proj(int row, int col) {
    float *out = (float*)malloc(row * col * sizeof(float));
    for (int i = 0; i < row*col; i++) {
        out[i] = ((float)rand() / (float)RAND_MAX);
    }
    return out;
}

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

// creates a lower-triangular of the matrix (working!)
void tril(float *attn_matrix, int row, int col) {
    for (int x = 0; x < row; x++) {
        for (int y = 0; y < col; y++) {
            if (x < y) {
                attn_matrix[x * row + y] = 0.0f;
            }
        }
    }
}

// Implement multi-head causal self-attention, treating each head as a dimension
void causal_self_attn(int B, int T, int C, float *wQ, float *wK, float *wV, float *in, float *out, int bias) {
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
            // should softmax be called here? in llm.c is not, maybe because backward
            
        }
    }

    free(query); free(key); free(value);
    free(attn_matrix); free(transpose_key);
}

/*
// single-head (for now). Also this is only self-attention, and I need to implement the causal_self-attention.
void self_attention(int B, int T, int C, float *wQ, float *wK, float *wV,
                    float *in, float *out, int bias) { // still has more args to insert
    // remember that each token_sequence will generate a query, key, value vector. One for each sequence
    // projections
    //float *wQ = init_rand_proj(C, C); //(float*)malloc(C * C * sizeof(float)); 

    // IMPORTANT!
    // big understanding: when storing the tensor in the memory, only its embeddings are being stored, and not the tokens as well. 
    // I though it was stored as: [t0, e0, e1, t1, e0, e1,...]. In reality is [e0,e1,e0,e1] (just the embeddings)
    // query, key, value matrices
    float *query = (float*)malloc(B * T * C * sizeof(float)); // (B,T,C) @ (C,C) -> (B,T,C)
    float *key   = (float*)malloc(B * T * C * sizeof(float));
    float *value = (float*)malloc(B * T * C * sizeof(float));
    //  calculate the query, key, value matrices
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *in_x = in + b * T * C + t * C; // skips to each embedding vector starting position
            float q, k, v = 0.0f;
            // aggregate the values of each embedding of token t * each col of wQ...
            for (int cw = 0; cw < C; cw++) {
                for (int c = 0; c < C; c++) { // on in_x, this loop will go through each embedding value of the embedding vector
                    q += in_x[c] * wQ[c * C + cw]; // "c*C": skips over the row. "+cw": go to the next column. I think now it's right
                    k += in_x[c] * wK[c * C + cw];
                    v += in_x[c] * wV[c * C + cw];
                }
            }
            // store the aggregated values into the right position of the resulting query/key/value matrices
            for (int i = 0; i < C; i++) {
                query[b*T*C+t*C+i] = q; // putting this outside for efficiency, avoiding wasteful memory access
                key[b*T*C+t*C+i]   = k;
                value[b*T*C+t*C+i] = v;
            }

        }
        // computing attention scores dotproduct(query*key)
        // the dot product takes 2 vectors and return 1 value. So each resulting value will correspond to the query[i] * key[j]
        //float *att_scores = (float*)malloc(B * T * T * sizeof(float)); // attention_score is a single number for each token
        // we can compute att_scores efficiently by stacking query and key vectors into 2 matrices, and multiplying query matrix with transposed key matrix
        // compute query[i] * key[j] for j in range(n)
        float *att_matrix = (float*)malloc(B * T * T * sizeof(float));
        float *transpose_keys = (float*)malloc(B * T * T * sizeof(float));
        transpose(key, transpose_keys, B, T, T); // transpose is not right yet
        
        float att_values = 0.0f;
        for (int tx = 0; tx < T; tx++) {
            for (int ty = 0; ty < C; ty++) {
                att_values += query[b*T*C+tx*C+ty] * transpose_keys[ty*C+tx]; // (B,T,C) @ (B,C,T) = (B,T,T). (ty*C+tx) gets the column elements
            }
        }
    } // batch loop

    free(wQ); free(wK); free(wV);
    free(query); free(key); free(value);
    free(att_matrix); free(transpose_keys);
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

void GELU() {
    // TODO:
}


// function to split dataset tokens into train/test (90,10)%
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
Notes:
TODO:
- allocate all necessary memory before hand, into a single place
- implement a simple DataLoader
- group all the Net parameters into a struct
*/