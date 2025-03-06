#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;

int main() {
    int i;
    double step, x, pi, sum = 0.0;
    double start_time, end_time;

    step = 1.0 / (double) num_steps;
    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nth = omp_get_num_threads();
        double local_sum = 0.0; // Each thread's private accumulation variable

        // Each thread works on a portion of the iterations.
        for (i = id; i < num_steps; i += nth) {
            x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        // Combine the local sums into the global sum atomically.
        #pragma omp atomic
        sum += local_sum;
    }

    pi = step * sum;
    end_time = omp_get_wtime();

    printf("Calculated pi = %f in %f seconds\n", pi, end_time - start_time);
    return 0;
}
