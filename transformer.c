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

// Layer normalization over the embedding dimmension for each token in the sequence. For each token, calculate the mean and variance over the token embedding vector. In the end, each token of each batch will have it's own mean and variance values.
void layernorm(int B, int T, int C, float *in, float *out, float gamma, float beta, float eps) { // gamma: weights, beta: bias (PyTorch)
    //eps = 1e-5;
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
                float x = sq * (in[c]-mean);
                // scale and shift
                out_ptr[c] = x * gamma + beta;
            }
            // should I pass the gamma and beta as arrays? Then cache them to be easier to optimize ... 
        }
    }
}

// void softmax(int B, int T, int C, float *logits, float *probs){};

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

// single-head (for now)
void self_attention(int B, int T, int C, float *wQ, float *wK, float *wV,
                    float *in, float *out, int bias) { // still has more args to insert
    // If I don't pass the wMatrices as parameters. I think the right thing is to pass the weight matrices in parameters.
    // projections
    //float *wQ = init_rand_proj(C, C); //(float*)malloc(C * C * sizeof(float));
    //float *wK = init_rand_proj(C, C); //(float*)malloc(C * C * sizeof(float));
    //float *wV = init_rand_proj(C, C); //(float*)malloc(C * C * sizeof(float));

    // query, key, value matrices
    float *query = (float*)malloc(B * T * C * sizeof(float)); // (T, C). Should it be (B,T,C)?
    float *key   = (float*)malloc(B * T * C * sizeof(float));
    float *value = (float*)malloc(B * T * C * sizeof(float));
    //  calculate the query, key, value matrices
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *in_x = in + b * T * C + t * C; // skips to each embedding vector
            float q, k, v = 0.0f;
            
            for (int cw = 0; cw < C; cw++) {
                for (int c = 0; c < C; c++) {
                    q += in_x[c] * wQ[c * C + cw]; // Don't get the indexing yet. Revisit after. Let's move on for now!
                    k += in_x[c] * wK[c * C + cw];
                    v += in_x[c] * wV[c * C + cw];
                }
            }
            query[b*T*C+t*C] = q; // putting this outside for efficiency, avoiding wasteful memory access
            key[b*T*C+t*C]   = k;
            value[b*T*C+t*C] = v;

        }
        // calculate attention scores dotproduct(query*key)
        // every token has now a query and a key vector
        float *att_scores = (float*)malloc(B * T * T * sizeof(float)); // attention_score is a single number for each token
        for (int query_vec = 0; query_vec < T; query_vec++) {
            for (int key_vec = 0; key_vec < T; key_vec++) {
                att_scores
            }
        }
    }

    free(wQ); free(wK); free(wV);
    free(query); free(key); free(value);
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