#include <stdlib.h>

void char_encode_tok(char *str, int *tokens, int n) {
    for (int i = 0; i < n; i++) {
        char ch = str[i];
        tokens[i] = (int)ch;
    }
}

void char_decode_tok(int *tokens, char *str, int n) {
    for (int i = 0; i < tokens; i++) {
        str[i] = (int)tokens[i];
    }
}

void char_print_tok(char *str, int *tokens, int n, int flag) {
    if (flag) { // print integer tokens
        for (int i = 0; i < n; i++) {
            if (i == n-1) {
                printf("%d\n", tokens[i]);
            } else {
                printf("%d, ", tokens[i]);
            }
        }
    } else {
        printf("%s\n", str);
    }
}