#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "tensor.c"

#define BATCH_SZ 4
#define BLOCK_SZ 8
#define EMBD_SZ 4

char *read_file(char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) return NULL;
    fseek(fp, 0, SEEK_END); // moves the file pointer (fp) to the end of the file
    int file_len = ftell(fp); // ftell returns the current value of the position indicator, giving us the number of characters in the file
    fseek(fp, 0, SEEK_SET); // moves fp back to the beginning of the file

    // dynamic allocating our char array
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

// get all characters from input string
char *get_vocab(char *input) {
    bool seen[256] = {false};
    int index = 0;

    char *vocab = (char*)malloc(256 * sizeof(char));
    if (vocab == NULL) {
        printf("Memory allocation failed!");
        return NULL;
    }

    for (int i = 0; input[i] != '\0'; i++) {
        unsigned char c = input[i];
        if (!seen[c]) {
            vocab[index++] = c;
            seen[c] = true;
        }
    }
    vocab[index] = '\0';

    return vocab;
}

// Usar "char_tokenization.c" depois
// function that encodes the string input, and a function to print those tokens
void encode_tokens(char *str, int *tokens, int n) {
    for (int i = 0; i < n; i++) {
        char ch = str[i];
        tokens[i] = (int)ch;
    }
}

float *init_token_emb_matrix(int vocab_sz, int emb_dim) {
    float *embeddings = (float*)malloc(vocab_sz * emb_dim * sizeof(float));
    for (int i = 0; i < vocab_sz*emb_dim; i++) {
        embeddings[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }

    return embeddings;
}

// token embedding converts a token id into a vector of size C.
void token_embedding(float *output, int *token_ids, float *embeddings, int B, int T, int C) {
    // token is represented by an integer (NOT FLOAT)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = token_ids[b * T + t];
            float *token_emb = embeddings + token_id * C;

            for (int c = 0; c < C; c++) {
                output[(b * T + t) * C + c] = token_emb[c];
            }
        }
    }
}

float *init_pos_emb_matrix(int sequence_len, int size) {
    float *pos_embeddings = (float*)malloc(sequence_len * size * sizeof(float));
    for (int i = 0; i < sequence_len*size; i++) {
        pos_embeddings[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }
    return pos_embeddings;
}

//void encoder(int B, int T, int C, float *wte, float *wpe, float *in, float *out) {
    // wte: token embedding matrix
    // wpe: positional embedding matrix
//}


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
    int *data = (split == "train") ? train_data : test_data;
    int ix;
    int ixs[batch_sz]; // use this instead if I want to keep track of the random indices

    // generate (batch_sz) random indices 
    srand(time(NULL));
    for (int i = 0; i < batch_sz; i++) { // ex: batch_sz=4
        ix = (rand() % (data_sz - block_sz));// + 1);
        //ixs[i] = ix;
        printf("ix:%d\n", ix);
        for (int j = 0; j < block_sz; j++) {
            x[i * block_sz + j] = data[ix + j]; // this is stacking each block_sz row.
            y[i * block_sz + j] = data[ix + j + 1];
        }
    }
}

// implement linear? 
/* How it works:
    m = nn.Linear(20, 30)
    input = tensor(128, 20)
    when output = m(input). performs a matrix multiplication of (128, 20) @ (20, 30) -> (128, 30)
*/

/*
float *linear(int n_embd1, int n_embd2, bool bias) {

}

*/

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
    encode_tokens(str, tokens, n);
    printf("tokens length: %d\n", n);

    // train/test split
    int split = (int)(0.9*n);
    printf("split: %d\n", split);
    printf("(n-split): %d\n", n-split);
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
    printf("x size:%d\n", sizeof(x)/sizeof(x[0]));
    // printing (DEBUG)
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


    // test for token_embedding
    int C = 10;
    int B = 1;
    int T = n;
    float *embeddings = init_token_emb_matrix(vocab_sz, C);

    //int token_ids[10] = {12, 345, 678, 910, 11, 5678, 1234, 2345, 3456, 4567};
    float *output = (float*)malloc(n * C * sizeof(float));
    token_embedding(output, tokens, embeddings, B, T, C);
    // Print the result for the first token in the first batch (for demonstration purposes)
    printf("Embedding for the first token in the first batch:\n");
    for (int i = 0; i < C; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free the allocated memory
    free(embeddings);
    free(output);


    
    free(input);
    free(tokens);
    free(train_data);
    free(test_data);
    free(x); free(y);

    return 0;
}