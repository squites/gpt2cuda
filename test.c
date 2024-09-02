#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 65
#define ASCII_SIZE 256

/*
int *encode(char *str, int n) {
    int tokens_int[n];
    for (int i = 0; i < n; i++) {
        char ch = str[i];
        tokens_int[i] = (int)ch;
    }
    return tokens_int;
}
*/

void dynamic_tokens(const char *str, int *tokens, int n) {
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
}

int main() {
    const char *input = "large language models!";
    int n = strlen(input);
    int *tokens = (int*)malloc(n * sizeof(int));
    dynamic_tokens(input, tokens, n);
    //tokens = encode(input, strlen(input));
    print_tokens(tokens, n);
}
/*
void chartoint_map(const char *str, int *mapping) {
    int n = strlen(str);
    for (int i = 0; i < ASCII_SIZE; i++) {
        mapping[i] = -1;
    }

    for (int i = 0; i < n; i++) {
        mapping[str[i]] = i;
    }
}

int *encode(char *str, int *mapping, int *encoded_len) {
    int len = strlen(str);
    int *encoded_array = (int*)malloc(len * sizeof(int));
    if (encoded_array == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    for (int i = 0; i < len; i++) {
        if (mapping[str[i]] != -1) {
            encoded_array[count++] = mapping[str[i]];
        } else {
            fprintf(stderr, "error", str[i], i);
        }
    }
    *encoded_array = count;
    return encoded_array;
}

int main() {
    const char *uniqueChars = "abcdefg";
    const char *stringToEncode = "facebad";

    int charToIntMapping[ASCII_SIZE];
    chartoint_map(uniqueChars, charToIntMapping);

    int encodedLength = 0;
    int *encodedArray = encode(stringToEncode, charToIntMapping, &encodedLength);

    char *str = "abcabsamuel";
    int ints[strlen(str)];
    for (int i = 0; i < strlen(str); i++) {
        char ch = str[i];
        ints[i] = (int)ch;
        printf("%d ", i);
    }
    for (int i = 0; i < strlen(str); i++) {
        printf("%d ", ints[i]);
    }
    printf("\narray size:%lu\n", (sizeof(ints) / sizeof(ints[0])));

    printf("Original String: %s\n", stringToEncode);
    printf("Encoded Values: ");
    for (int i = 0; i < encodedLength; i++) {
        printf("%d ", encodedArray[i]);
    }
    printf("\n");

    // Clean up
    free(encodedArray);

    return 0;
}
*/