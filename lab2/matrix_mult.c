#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int m = 1000;  // number of rows for A and C; also columns for B
    int n = 1000;  // number of columns for A and rows for B

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    double *a = (double *) malloc(m * n * sizeof(double));
    double *b = (double *) malloc(n * m * sizeof(double));
    double *c = (double *) malloc(m * m * sizeof(double));

    if (!a || !b || !c) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Initialize matrix A: A[i][j] = (i+1) + (j+1)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }

    // Initialize matrix B: B[i][j] = (i+1) - (j+1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }

    // Initialize matrix C to 0: C is m x m
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0.0;
        }
    }

    int chunk = 10;  
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(dynamic, chunk)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }

    double end = omp_get_wtime();
    double elapsed = end - start;
    printf("Matrix multiplication time: %f seconds\n", elapsed);


    free(a);
    free(b);
    free(c);
    return 0;
}
