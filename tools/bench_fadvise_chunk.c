// bench_fadvise_chunk.c: misst den Fault-Durchgang ueber die Modell-Dateien bei
// verschiedenen Fenstergroessen fuer den posix_fadvise-Hinweis.
//
// Der Ist-Pfad (ggml-cpu.c:1847) legt je fehlendem Experten-Slab 128-KiB-Fenster
// mit POSIX_FADV_WILLNEED auf die Datei und faultet danach jede 4-KiB-Seite
// einzeln (off += page). Ohne Vorwarnung liefert dieser Durchgang nur ~82 MiB/s
// (gemessen mit tools/bench_thp_file.c), weil der Kernel 4-KiB-Seiten einzeln
// nachlaedt. Mit Vorwarnung kommen die Bytes in grossen Bloecken.
//
// Offene Frage: ist 128 KiB die richtige Fenstergroesse, oder bringt 1 MiB
// mehr, weil die SSD dann weniger, aber groessere Requests sieht?
//
// Aufruf: bench_fadvise_chunk <datei> <start_offset> <laenge> <chunk_bytes> <threads>
// Ausgabe: Rate des Durchgangs (Fault der gesamten Region) in MiB/s.

#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct job {
    const char * path;
    size_t start;
    size_t length;
    size_t chunk;
    int use_fadvise;
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Fehlert den Bereich in 4-KiB-Seiten und gibt die verstrichene Zeit zurueck.
static double fault_range(int fd, void * base, size_t start, size_t length, size_t chunk) {
    // Hinweise legen (optional).
    if (chunk > 0) {
        for (size_t off = start; off < start + length; off += chunk) {
            size_t len = length - (off - start);
            if (len > chunk) {
                len = chunk;
            }
            posix_fadvise(fd, (off_t) off, len, POSIX_FADV_WILLNEED);
        }
    }
    const long page = sysconf(_SC_PAGESIZE);
    double t0 = now_s();
    volatile unsigned char acc = 0;
    for (size_t off = start; off < start + length; off += (size_t) page) {
        acc |= ((volatile unsigned char *) base)[off];
    }
    (void) acc;
    return now_s() - t0;
}

struct targ {
    struct job * j;
    int fd;
    void * base;
    size_t part;
    int idx;
    double dt;
};

static void * worker(void * p) {
    struct targ * t = (struct targ *) p;
    size_t n = t->j->length / (size_t) t->part;
    size_t start = t->j->start + (size_t) t->idx * n;
    size_t len = (t->idx == t->part - 1) ? (t->j->length - n * (size_t) (t->part - 1)) : n;
    t->dt = fault_range(t->fd, t->base, start, len, t->j->chunk);
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <file> <offset> <length> <chunk_bytes|0> <threads>\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    long long start = strtoll(argv[2], NULL, 10);
    long long length = strtoll(argv[3], NULL, 10);
    size_t chunk = strtoull(argv[4], NULL, 10);
    int threads = atoi(argv[5]);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    if (start < 0 || start >= st.st_size) {
        fprintf(stderr, "offset %lld liegt ausserhalb der Datei (%lld Bytes)\n",
                (long long) start, (long long) st.st_size);
        return 2;
    }
    if (start + (long long) length > st.st_size) {
        length = st.st_size - start;
    }
    void * base = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    struct job j = { path, start, length, chunk, chunk > 0 };
    pthread_t th[64];
    struct targ ta[64];
    double t0 = now_s();
    for (int i = 0; i < threads; ++i) {
        ta[i].j = &j;
        ta[i].fd = fd;
        ta[i].base = base;
        ta[i].part = threads;
        ta[i].idx = i;
        ta[i].dt = 0;
        pthread_create(&th[i], NULL, worker, &ta[i]);
    }
    double wall = 0;
    for (int i = 0; i < threads; ++i) {
        pthread_join(th[i], NULL);
        if (ta[i].dt > wall) {
            wall = ta[i].dt;
        }
    }
    double total = now_s() - t0;
    printf("chunk=%-9zu threads=%-3d  %.2f s gesamt, %.2f s langsamster Thread, %.1f MiB/s\n",
           chunk, threads, total, wall,
           (double) length / (1 << 20) / total);
    munmap(base, (size_t) st.st_size);
    close(fd);
    return 0;
}
