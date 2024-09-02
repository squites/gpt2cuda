#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "tensor.c"

#define BATCH_SZ 4
#define BLOCK_SZ 8

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

#define VOCAB_SIZE 65
int size = 0;
char keys[VOCAB_SIZE][VOCAB_SIZE];
int values[VOCAB_SIZE];

int get_index(char *key) {
    for (int i = 0; i < size; i++) {
        if (strcmp(keys[i], key) == 0) {
            return i;
        }
    }
    return -1;
}

void insert(char key[], int value) {
    int index = get_index(key);
    if (index == -1) {
        strcpy(keys[size], key);
        values[size] = value;
        size++;
    } else {
        values[index] = value;
    }
}

int get(char key[]) {
    int index = get_index(key);
    if (index == -1) {
        return -1;
    } else {
        return values[index];
    }
}

void printmap() {
    for (int i = 0; i < size; i++) {
        printf("%s: %d\n", keys[i], values[i]);
    }
}

// Usar "char_tokenization.c" depois
// function that encodes the string input, and a function to print those tokens
void encode_tokens(char *str, int *tokens, int n) {
    for (int i = 0; i < n; i++) {
        char ch = str[i];
        tokens[i] = (int)ch;
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

/*
int *reshape(int *x, Tuple_t *dims) {
    int *tensor = (int*)malloc()
    for (int i = 0; i < dims->x; i++) {
        for (int j = 0; j < dims->y; j++) {
            for (int k = 0; k < dims->z; k++) {

            }
        }
    }
}


typedef struct {
    int x;
    int y;
    int z;
} Tuple_t;

*/

// NEEDS TO MODIFY. THIS IS JUST A TEST!!!
void get_batch(char *split, int *train_data, int *test_data, int train_sz, int test_sz, int *x, int *y,
               int batch_sz, int block_sz) {
    int *data = NULL;
    int data_sz;
    int ix;//[batch_sz] = {0};
    //int *x = (int*)malloc(batch_sz * block_sz * sizeof(int));
    //int *y = (int*)malloc(batch_sz * block_sz * sizeof(int));
    
    if (strcmp(split, "train")) {
        data = &train_data;
        data_sz = train_sz;
    } else {
        data = &test_data;
        data_sz = test_sz;
    }

    //int **x_B = (int**)malloc(batch_sz * block_sz * sizeof(int));
    //for (int j = 0; j < batch_sz; j++) {
    //    for (int i = 0; i < block_sz; i++) {

    //    }
    //}

    // generate random indices 
    srand(time(0));
    for (int i = 0; i < batch_sz; i++) { // ex: batch_sz=4
        ix = (rand() % (data_sz - block_sz) + 1);
        for (int j = 0; j < block_sz; j++) {
            x[i * block_sz + j] = data[ix + j]; // this is stacking each block_sz row.
            //y[i * block_sz + j] = data[ix + j + 1];
        }
    }


    //x = reshape(x, )
    //for (int i = 0; i < batch_sz*block_sz; i+=block_sz) {
        
    //}

}


int main() {
    char *filename = "input.txt";
    char *input = read_file(filename);
    if (input == NULL) {
        printf("Error reading file!\n");
        return 1;
    }
    
    // get vocab from input string and sort. For character tokenization we don't actually need a vocab. But eventually we'll implement a more efficient tokenization (BPE)
    char *vocab = get_vocab(input);
    qsort(vocab, strlen(vocab), sizeof(char), compare);
    printf("vocab:\n%s\n%lu\n", vocab, strlen(vocab)); // 1st char of vocab is '\n'

    // encode tokens
    char *str = input; // input: passing the whole file
    int n = strlen(str);
    printf("n: %d\n", n);
    int *tokens = (int*)malloc(n * sizeof(int));
    encode_tokens(str, tokens, n);
    //print_tokens(tokens, n);
    printf("tokens length: %d\n", n);

    // train/test split
    int split = (int)(0.9*n);
    //printf("split: %d\n", split);
    //printf("(n-split): %d\n", n-split);
    int *train_data = (int*)malloc(split * sizeof(int));
    int *test_data = (int*)malloc((n-split) * sizeof(int));
    int train_sz = split, test_sz = n-split;
    split_data(tokens, n, train_data, test_data, train_sz, test_sz);
    //print_tokens(train_data, train_sz);
    //print_tokens(test_data, test_sz);
    
    // generate a random batch of data (inputs: x, target: y)
    int *x = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int));
    int *y = (int*)malloc(BATCH_SZ * BLOCK_SZ * sizeof(int));
    get_batch("train", train_data, test_data, train_sz, test_sz, x, y, BATCH_SZ, BLOCK_SZ);


    
    free(input);
    free(tokens);
    free(train_data);
    free(test_data);
    free(x); free(y);

    return 0;
}