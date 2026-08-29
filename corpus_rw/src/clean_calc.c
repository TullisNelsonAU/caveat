/* clean_calc — a deterministic, linear-sweep-clean workload for the confidence-gated rewriter.
 *
 * Behaviour is a pure function of argv STRINGS (never pointers or ASLR), emitted to stdout, so the
 * reference I/O is stable across runs and machines — the behavioural oracle the rewriter is judged by.
 * Built at -O0 so .text is straight-line code with no in-section jump tables: a *clean* case where the
 * baseline's linear sweep is correct and BOTH rewriters must keep the binary working (the sanity axis).
 *
 * Several small noinline functions give the rewriter a spread of real block leaders to instrument. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t parse_u64(const char *s) {
    uint64_t v = 0;
    for (; *s >= '0' && *s <= '9'; s++) v = v * 10 + (uint64_t)(*s - '0');
    return v;
}

__attribute__((noinline)) static uint64_t mix(uint64_t a, uint64_t b) {
    a += b;
    a ^= a >> 7;
    a *= 0x9e3779b97f4a7c15ULL;
    a ^= b << 3;
    return a;
}

__attribute__((noinline)) static uint64_t fold(const uint64_t *xs, int n) {
    uint64_t acc = 0x1234567;
    for (int i = 0; i < n; i++) acc = mix(acc, xs[i]);
    return acc;
}

__attribute__((noinline)) static uint64_t digits_of(const char *s) {
    uint64_t d = 0;
    for (; *s; s++) if (*s >= '0' && *s <= '9') d++;
    return d;
}

int main(int argc, char **argv) {
    uint64_t xs[16];
    int n = 0;
    for (int i = 1; i < argc && n < 16; i++) xs[n++] = parse_u64(argv[i]);
    uint64_t f = fold(xs, n);
    uint64_t dsum = 0;
    for (int i = 1; i < argc; i++) dsum += digits_of(argv[i]);
    printf("n=%d fold=%016llx dsum=%llu mix=%016llx\n",
           n, (unsigned long long)f, (unsigned long long)dsum,
           (unsigned long long)mix(f, dsum));
    return (int)(f & 0x7f);
}
