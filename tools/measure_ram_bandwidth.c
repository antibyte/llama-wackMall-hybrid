#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MB (1024*1024)
static char *src, *dst;
static int nthreads, per_mb;

static void *worker(void *arg) {
    long id = (long)arg;
    char *s = src + id*per_mb*MB;
    char *d = dst + id*per_mb*MB;
    size_t n = (size_t)per_mb*MB;
    for (int r = 0; r < 3; ++r) memcpy(d, s, n);
    return NULL;
}

int main(int argc, char **argv) {
    nthreads = argc > 1 ? atoi(argv[1]) : 8;
    per_mb = argc > 2 ? atoi(argv[2]) : 256;
    src = aligned_alloc(4096, (size_t)nthreads*per_mb*MB);
    dst = aligned_alloc(4096, (size_t)nthreads*per_mb*MB);
    memset(src, 1, (size_t)nthreads*per_mb*MB);
    memset(dst, 2, (size_t)nthreads*per_mb*MB);
    pthread_t th[64];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < nthreads; ++i) pthread_create(&th[i], NULL, worker, (void*)i);
    for (int i = 0; i < nthreads; ++i) pthread_join(th[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double dt = (b.tv_sec - a.tv_sec) + 1e-9*(b.tv_nsec - a.tv_nsec);
    double moved = 3.0 * nthreads * per_mb * MB;
    printf("threads %d: %.1f GB/s (%.0f MiB je Thread)\n", nthreads, moved/dt/1e9, (double)per_mb);
    return 0;
}
