// bench_thp_file.c: prueft, ob file-backed Huge Pages fuer das Modell-Mapping greifen.
//
// Der Prefill faultet die Modellseiten in 4-KiB-Stuecken (ggml-cpu.c:1856,
// Schleife mit off += page). Wenn THP fuer das Mapping greift, faellt ein
// Fault je 2 MiB an - das waere ein Vielfaches weniger Kernel-Overhead und
// groessere sequentielle Reads am Geraet.
//
// Aufruf: bench_thp_file <datei> [offset] [laenge_bytes]
// Ausgabe: Rate des Fault-Durchgangs mit und ohne MADV_HUGEPAGE, dazu die
//          smaps-Auszuege (AnonHugePages / FileThp / Rss) des Mapping-Bereichs.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Liest /proc/self/smaps und gibt die Zeilen aus, die zum Mapping gehoeren.
static void dump_smaps(unsigned long lo, unsigned long hi) {
    FILE * f = fopen("/proc/self/smaps", "r");
    if (!f) {
        return;
    }
    char line[512];
    int in_range = 0;
    int printed_header = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] >= '0' && line[0] <= '9' && strstr(line, "-") != NULL) {
            unsigned long a = 0, b = 0;
            if (sscanf(line, "%lx-%lx", &a, &b) == 2) {
                in_range = (a >= lo && a < hi);
                if (in_range && !printed_header) {
                    printf("  smaps-Auszug:\n");
                    printed_header = 1;
                }
            }
        }
        if (!in_range) {
            continue;
        }
        if (strstr(line, "AnonHugePages") || strstr(line, "FileThp") ||
            strstr(line, "THPStatus") || strstr(line, "Rss:") ||
            strstr(line, "VmFlags")) {
            printf("   %s", line);
        }
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [offset] [length]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    long long offset = argc > 2 ? atoll(argv[2]) : 0;
    long long length = argc > 3 ? atoll(argv[3]) : (64LL << 20);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    struct stat st;
    fstat(fd, &st);
    if (offset + length > st.st_size) {
        length = st.st_size - offset;
    }

    void * addr = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    unsigned long lo = (unsigned long) addr;
    unsigned long hi = lo + (unsigned long) st.st_size;

    const long page = sysconf(_SC_PAGESIZE);
    long long n_pages = length / page;

    printf("datei    : %s\n", path);
    printf("groesse  : %lld Bytes, Testbereich %lld .. %lld (%lld Seiten)\n",
           (long long) st.st_size, offset, offset + length, n_pages);
    printf("kernel   : %s\n", "siehe uname");
    {
        FILE * u = popen("uname -r", "r");
        char b[128];
        if (u && fgets(b, sizeof(b), u)) {
            printf("kernel   : %s", b);
        }
        if (u) {
            pclose(u);
        }
    }
    {
        FILE * t = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
        char b[256];
        if (t && fgets(b, sizeof(b), t)) {
            printf("thp      : %s", b);
        }
        if (t) {
            fclose(t);
        }
    }
    {
        char cmd[4096];
        snprintf(cmd, sizeof(cmd), "stat -f -c 'fs      : %%T' '%s'", path);
        FILE * s = popen(cmd, "r");
        char b[256];
        if (s && fgets(b, sizeof(b), s)) {
            printf("%s", b);
        }
        if (s) {
            pclose(s);
        }
    }
    printf("\n");

    // Lauf 1: ohne MADV_HUGEPAGE
    {
        double t0 = now_s();
        volatile unsigned char acc = 0;
        for (long long i = 0; i < n_pages; ++i) {
            acc |= ((volatile unsigned char *) addr)[offset + i * page];
        }
        double dt = now_s() - t0;
        printf("ohne THP : %6.2f s fuer %lld Seiten -> %.1f MiB/s\n",
               dt, n_pages, (double) n_pages * page / (1 << 20) / dt);
        (void) acc;
    }

    // MADV_HUGEPAGE setzen und den zweiten Bereich im selben Mapping faulten.
    if (madvise(addr, (size_t) st.st_size, MADV_HUGEPAGE) != 0) {
        perror("madvise(MADV_HUGEPAGE)");
    } else {
        printf("\nmadvise(MADV_HUGEPAGE) -> ok\n");
    }

    long long offset2 = offset + length;
    if (offset2 + length > st.st_size) {
        offset2 = 0;
    }
    long long n2 = (st.st_size - offset2) / page;
    if (n2 > n_pages) {
        n2 = n_pages;
    }
    {
        double t0 = now_s();
        volatile unsigned char acc = 0;
        for (long long i = 0; i < n2; ++i) {
            acc |= ((volatile unsigned char *) addr)[offset2 + i * page];
        }
        double dt = now_s() - t0;
        printf("mit THP  : %6.2f s fuer %lld Seiten -> %.1f MiB/s\n",
               dt, n2, (double) n2 * page / (1 << 20) / dt);
        (void) acc;
    }

    printf("\n");
    dump_smaps(lo, hi);
    munmap(addr, (size_t) st.st_size);
    close(fd);
    return 0;
}
