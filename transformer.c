#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "tensor.c"
#include "tokenizer.c"

#define BATCH_SZ 4 // B
#define BLOCK_SZ 8 // T
#define EMBD_SZ 4  // C

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
    // wte: token embedding matrix
    // wpe: positional embedding matrix
    for (int b = 0; b < B; b++) { // loop over batches
        for (int t = 0; t < T; t++) { // loop over each token in the batch
            // get token value on a specific index
            int token_id = in[b * T + t]; // this gets the token value, not its position. That's why we don't use on pos_emb
            // gets the token embedding of that token
            float *token_embd = wte + token_id * C;
            // gets the positional embedding of that token
            float *pos_embd = wpe + t * C;
            for (int c = 0; c < C; c++) {
                out[(b * T + t) * C + c] = token_embd[c] + pos_embd[c]; // add token_embd with pos_embd of that specific token.
            }

        }
    }
}

void print_tokens(int *tokens, int n) {
    for (int i = 0; i < n; i++) {
        if (i == n-1) {
            printf("%d\n", tokens[i]);
        } else {
            printf("%d, ", tokens[i]);
        }
    }
    printf("\n");
}

// function to compare chars for qsort
int compare(const void *a, const void *b) {
    return *(char*)a - *(char*)b;
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

// print token-lookup-table or pos-lookup-table
void print_lookup_table(char *name, float *matrix, int row, int col) {
    int size = row*col;
    if (name=="train") {
        printf("token embedding matrix:\n");
    } else {
        printf("positional embedding matrix:\n");
    }
    int k = 0;
    for (int i = 0; i < size; i++) {
        if (k == col) {
            printf("\n");
            k = 0;
        }
        k++;
        printf("%f, ", matrix[i]);
    }
    printf("\n");
}

int main() {
    char *filename = "input/input.txt";
    char *input = read_file(filename);
    if (input == NULL) {
        printf("Error reading file!\n");
        return 1;
    }
    
    // get vocab from input string and sort. For character tokenization we don't actually need a vocab. But eventually we'll implement a more efficient tokenization (BPE)
    char *vocab = get_vocab(input);
    int vocab_sz = strlen(vocab);
    qsort(vocab, strlen(vocab), sizeof(char), compare);
    printf("vocab:\n%s\n%lu\n", vocab, strlen(vocab)); // 1st char of vocab is '\n'

    // encode tokens
    char *str = "First Citizen: We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them Let us revenge this with our pikes, ere we become rakes: for the gods know I ";
    //char *str = input; // input: passing the whole file
    int n = strlen(str);
    printf("n: %d\n", n);
    int *tokens = (int*)malloc(n * sizeof(int));
    //encode_tokens(str, tokens, n);
    encode_tokens_v2(str, tokens, vocab, n);
    printf("tokens length: %d\n", n);

    // train/test split
    int split = (int)(0.9*n); // (split=90%) and (n-split=10%)
    int *train_data = (int*)malloc(split * sizeof(int));
    int *test_data = (int*)malloc((n-split) * sizeof(int));
    int train_sz = split, test_sz = n-split;
    split_data(tokens, n, train_data, test_data, train_sz, test_sz);
    //print_tokens(train_data, train_sz);
    //print_tokens(test_data, test_sz);
    
    // generate a random batch of data (inputs: x, target: y)
    int *x = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int));
    int *y = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int));
    get_batch("train", train_data, test_data, train_sz, x, y, BATCH_SZ, BLOCK_SZ);
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

    // test for token_embedding
    int B = BATCH_SZ;
    int T = BLOCK_SZ;
    int C = EMBD_SZ;
    float *embeddings = init_token_emb_matrix(vocab_sz, C); // (65, 4). matrix is working fine
    float *pos_embeddings = init_pos_emb_matrix(T, C);      // ( 8, 4)

    print_lookup_table("token", embeddings, vocab_sz, C);
    print_lookup_table("pos", pos_embeddings, T, C);
     
    float *output_emb = (float*)malloc(B * T * C * sizeof(float));
    int *tokens_id = x; // need to do to 'y' as well?
    //token_embedding(output, tokens_id, embeddings, B, T, C, vocab_sz);
    encoder(B, T, C, embeddings, pos_embeddings, tokens_id, output_emb);
    
    // Print the result for the first token in the first batch (for demonstration purposes) 
    for (int j = 0; j < T; j++) {
        printf("\nEmbedding for the each T token in the first batch:\n");
        for (int i = 0; i < C; i++) {
            printf("%f ", output_emb[j * C + i]);
        }
        printf("\n");
    }
    printf("\n");
    
    
    // Free the allocated memory
    free(embeddings);
    free(output_emb);
    
    free(input);
    free(tokens);
    free(train_data);
    free(test_data);
    free(x); free(y);

    return 0;
}