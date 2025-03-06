#include <stdio.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int main() {
    int num_threads = 1, thread_id = 0;
    
    #ifdef _OPENMP
    #pragma omp parallel private(thread_id)
    {
        thread_id = omp_get_thread_num();
        
        // Only let thread 0 print the total number of threads (once)
        if (thread_id == 0) {
            num_threads = omp_get_num_threads();
        }
    #endif
        
        printf("Hello from the rank %d thread\n", thread_id);
        
    #ifdef _OPENMP
        // Wait for all threads to finish printing before showing summary
        #pragma omp barrier
        
        // Let thread 0 print the summary
        if (thread_id == 0) {
    #endif
            printf("Parallel execution of hello_world with %d threads\n", num_threads);
    #ifdef _OPENMP
        }
    }
    #endif
    
    return 0;
}