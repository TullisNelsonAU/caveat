/* junk_desync — the SAME deterministic workload as clean_calc, but each hot function carries an
 * embedded anti-disassembly trap: a short byte blob that is jumped over at runtime (never executed)
 * yet sits in the linear byte stream of .text. A linear-sweep disassembler walks into the blob, whose
 * final opcode (`48 b8` = the start of `movabs rax, imm64`) swallows the first 8 bytes of the REAL
 * instructions that follow — desynchronising the sweep and planting block "leaders" at mid-instruction
 * addresses inside code that genuinely runs.
 *
 * A deterministic-CFG rewriter trusts those bogus leaders and patches live code → corruption. Our
 * calibrated marginals assign the desynced/junk addresses low belief (they are not real instruction
 * starts), so the confidence gate abstains there and the rewrite stays working. Behaviour (stdout) is
 * identical to clean_calc: the trap is pure disassembler poison, not a semantic change. */
#include <stdint.h>
#include <stdio.h>

/* A desync trap: `jmp past` skips the blob at runtime; linear sweep instead decodes `48 b8 …` as a
 * 10-byte `movabs`, eating the first bytes of whatever real code the compiler emits next. */
#define DESYNC_TRAP()                                        \
    __asm__ volatile goto("jmp %l[past]\n\t"                 \
                          ".byte 0x48, 0xb8\n\t"             \
                          : : : : past);                     \
    past:

static uint64_t parse_u64(const char *s) {
    uint64_t v = 0;
    for (; *s >= '0' && *s <= '9'; s++) v = v * 10 + (uint64_t)(*s - '0');
    return v;
}

__attribute__((noinline)) static uint64_t mix(uint64_t a, uint64_t b) {
    DESYNC_TRAP();
    a += b;
    a ^= a >> 7;
    a *= 0x9e3779b97f4a7c15ULL;
    a ^= b << 3;
    return a;
}

__attribute__((noinline)) static uint64_t fold(const uint64_t *xs, int n) {
    DESYNC_TRAP();
    uint64_t acc = 0x1234567;
    for (int i = 0; i < n; i++) acc = mix(acc, xs[i]);
    return acc;
}

__attribute__((noinline)) static uint64_t digits_of(const char *s) {
    DESYNC_TRAP();
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
