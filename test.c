#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void matmul_cpu(int *m, int *n, int *out, int row_m, int col_m, int col_n) {
    int value = 0;
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


void init_matrix(int *matrix, int row, int col) {
    for (int i = 0; i < row*col; i++) {
        matrix[i] = rand() % 100;
    }
}

void print_matrix(int *m, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d, ", m[j]);
        }
        printf("\n");
    }
}

int main() {
    int row = 2;
    int col = 3;
    
    int *m = (int*)malloc(row * col * sizeof(int));
    int *n = (int*)malloc(row * col * sizeof(int));
    int *out = (int*)malloc(row * row * sizeof(int));

    init_matrix(m, row, col);
    init_matrix(n, col, row);

    matmul_cpu(m, n, out, row, col, row);

    printf("m:\n");
    print_matrix(m, row, col);
    printf("\nn:\n");
    print_matrix(n, col, row);
    printf("\nout:\n");
    print_matrix(out, row, row);
}