#include <stdio.h>

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