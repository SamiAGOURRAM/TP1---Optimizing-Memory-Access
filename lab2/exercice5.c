#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef VAL_N
#define VAL_N 120
#endif

#ifndef VAL_D
#define VAL_D 80
#endif

void random_number(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / (double)(RAND_MAX - 1);
    }
}

int main() {
    int n = VAL_N, diag = VAL_D;
    int i, j, iteration = 0;
    double norme;

    double *a = (double *) malloc(n * n * sizeof(double));
    double *x = (double *) malloc(n * sizeof(double));
    double *x_courant = (double *) malloc(n * sizeof(double));
    double *b = (double *) malloc(n * sizeof(double));

    if (!a || !x || !x_courant || !b) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    struct timeval t_elapsed_0, t_elapsed_1;
    double t_elapsed;
    double t_cpu_0, t_cpu_1, t_cpu;

    srand(421); // For reproducibility
    random_number(a, n * n);
    random_number(b, n);

    // Strengthen the diagonal
    for (i = 0; i < n; i++) {
        a[i * n + i] += diag;
    }

    // Initial guess for the solution vector
    for (i = 0; i < n; i++) {
        x[i] = 1.0;
    }

    // Start timing
    t_cpu_0 = omp_get_wtime();
    gettimeofday(&t_elapsed_0, NULL);

    // Jacobi Iteration
    while (1) {
        iteration++;


        #pragma omp parallel for private(j) shared(x, x_courant, a, b, n)
        for (i = 0; i < n; i++) {
            double sum = 0.0;
            // Sum contributions from indices j < i
            for (j = 0; j < i; j++) {
                sum += a[j * n + i] * x[j];
            }
            // Sum contributions from indices j > i
            for (j = i + 1; j < n; j++) {
                sum += a[j * n + i] * x[j];
            }
            // Compute the new x value at index i
            x_courant[i] = (b[i] - sum) / a[i * n + i];
        }

        double absmax = 0.0;
        #pragma omp parallel for reduction(max:absmax)
        for (i = 0; i < n; i++) {
            double diff = fabs(x[i] - x_courant[i]);
            if (diff > absmax)
                absmax = diff;
        }
        norme = absmax / n;
        if ((norme <= DBL_EPSILON) || (iteration >= n))
            break;

        // Copy new solution into x for the next iteration
        memcpy(x, x_courant, n * sizeof(double));
    }

    gettimeofday(&t_elapsed_1, NULL);
    t_elapsed = (t_elapsed_1.tv_sec - t_elapsed_0.tv_sec) +
                (t_elapsed_1.tv_usec - t_elapsed_0.tv_usec) / 1e6;
    t_cpu_1 = omp_get_wtime();
    t_cpu = t_cpu_1 - t_cpu_0;

    // Print the results
    fprintf(stdout, "\n\n"
                    "   System          : %5d\n"
                    "   Iterations      : %4d\n"
                    "   Norme           : %10.3E\n"
                    "   Elapsed time    : %10.3E sec.\n"
                    "   CPU time        : %10.3E sec.\n",
                    n, iteration, norme, t_elapsed, t_cpu);

    free(a);
    free(x);
    free(x_courant);
    free(b);

    return EXIT_SUCCESS;
}
