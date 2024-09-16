#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// get all possible characters from input string. (NOT USING RIGHT NOW!)
char *get_vocab(char *input) {
    bool seen[256] = {0};
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
            seen[c] = 1;
        }
    }
    vocab[index] = '\0';

    return vocab;
}

int get_vocab_index_v2(char *vocab, char ch) {
    char *ptr = strchr(vocab, ch); // strchr: find the first occurence of a char in a string. Returns a pointer
    if (ptr) return ptr - vocab;
    return -1;
}

// encode tokens mapping the character to an integer
void encode_tokens_v2(char *str, int *tokens, char *vocab, int n) {
    for (int i = 0; i < n; i++) {
        char ch = str[i];
        tokens[i] = get_vocab_index_v2(vocab, ch);
        //printf("char:%c\n", ch); // prints to check if the char/token mapping is right
        //printf("token:%d\n", tokens[i]);
        if (tokens[i] == -1) {
            printf("Error: Character %c not found in vocab!\n", ch);
            exit(1);
        }
    }
}

// Initializes lookup-table for token and positional embeddings
float *init_token_emb_matrix(int vocab_sz, int emb_dim) {
    float *embeddings = (float*)malloc(vocab_sz * emb_dim * sizeof(float));
    for (int i = 0; i < vocab_sz*emb_dim; i++) {
        embeddings[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }
    return embeddings;
}

float *init_pos_emb_matrix(int seq_len, int n_embd) { // (T, n_embd)
    float *pos_embeddings = (float*)malloc(seq_len * n_embd * sizeof(float));
    for (int i = 0; i < seq_len*n_embd; i++) {
        pos_embeddings[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }
    return pos_embeddings;
}

/* Not using anymore. They're combined into "encoder" function, where we add token embedding vector with positional embedding vector.
void token_embedding(float *output, int *token_ids, float *embeddings, int B, int T, int C, int vocab_sz) {
    // token is represented by an integer (NOT FLOAT)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = token_ids[b * T + t]; // get the token value in the sequence. Ex: 2nd token of batch1 = [1 * 8 + 1] = [9]. So gets the token value at [9]. Could be 53 for example
            float *token_emb = embeddings + token_id * C;

            for (int c = 0; c < C; c++) {
                output[(b * T + t) * C + c] = token_emb[c];
            }
        }
    }
}

void pos_embedding(float *output, int *token_ids, float *pos_embeddings, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            //int token_pos_id = token_ids[b * T + t];
            float *token_pos_emb = pos_embeddings + t * C; // this gets the right row on each token. Multiplies by C because is linearized

            for (int c = 0; c < C; c++) {
                output[(b * T + t) * C + c] = token_pos_emb[c];
            }
        }
    }
}
*/