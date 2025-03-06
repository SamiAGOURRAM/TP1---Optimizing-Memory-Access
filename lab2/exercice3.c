#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;

int main() {
    int i;
    double step, x, pi, sum = 0.0;
    double start_time, end_time;

    step = 1.0 / (double) num_steps;
    start_time = omp_get_wtime();

    // Only one line is added here: the OpenMP directive.
    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    end_time = omp_get_wtime();

    printf("Calculated pi = %f in %f seconds\n", pi, end_time - start_time);
    return 0;
}
