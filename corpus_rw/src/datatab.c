/* datatab — a deterministic workload whose hot path READS a lookup table embedded in .text.
 *
 * The table `TAB` is real read-only DATA that happens to live inside the executable section (forced
 * there with a section attribute — the same "code-in-data" shape gauntlet builds and the stack is
 * trained to suppress). A linear-sweep disassembler cannot tell the 8-byte constants from code: it
 * decodes them as instructions and reports block "leaders" inside the table. A deterministic rewriter
 * then writes a detour over those bytes — corrupting the constants the program later reads, so the
 * output changes even though the detour itself never executes.
 *
 * The stack's calibrated marginals give the table bytes low belief (no control-flow coherence — they
 * are not instruction starts), so the confidence gate abstains and the table survives intact. Behaviour
 * is a pure, deterministic function of argv content, emitted to stdout: a stable behavioural oracle. */
#include <stdint.h>
#include <stdio.h>

/* 64 pseudo-random 64-bit constants, forced into .text. `used` + volatile-free but read at runtime via
 * lookup(), so the linker keeps it and the program depends on its exact bytes. */
__attribute__((section(".text"), used, aligned(8)))
static const uint64_t TAB[64] = {
    0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
    0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
    0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL, 0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
    0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL, 0x0801f2e2858efc16ULL, 0x636920d871574e69ULL,
    0xa458fea3f4933d7eULL, 0x0d95748f728eb658ULL, 0x718bcd5882154aeeULL, 0x7b54a41dc25a59b5ULL,
    0x9c30d5392af26013ULL, 0xc5d1b023286085f0ULL, 0xca417918b8db38efULL, 0x8e79dcb0603a180eULL,
    0x6c9e0e8bb01e8a3eULL, 0xd71577c1bd314b27ULL, 0x78af2fda55605c60ULL, 0xe65525f3aa55ab94ULL,
    0x5748986263e81440ULL, 0x55ca396a2aab10b6ULL, 0xb4cc5c341141e8ceULL, 0xa15486af7c72e993ULL,
    0xb3ee1411636fbc2aULL, 0x2ba9c55d741831f6ULL, 0xce5c3e169b87931eULL, 0xafd6ba336c24cf5cULL,
    0x7a32538128958677ULL, 0x3b8f48986b4bb9afULL, 0xc4bfe81b66282193ULL, 0x61d809ccfb21a991ULL,
    0x487cac605dec8032ULL, 0xef845d5de98575b1ULL, 0xdc262302eb651b88ULL, 0x23893e81d396acc5ULL,
    0x0f6d6ff383f44239ULL, 0x2e0b4482a4842004ULL, 0x69c8f04a9e1f9b5eULL, 0x21c66842f6e96c9aULL,
    0x670c9c61abd388f0ULL, 0x6a51a0d2d8542f68ULL, 0x960fa728ab5133a3ULL, 0x6eef0b6c137a3be4ULL,
    0xba3bf0507efb2a98ULL, 0xa1f1651d39af0176ULL, 0x66ca593e82430e88ULL, 0x8cee8619456f9fb4ULL,
    0x7d84a5c33b8b5ebeULL, 0xe06f75d885c12073ULL, 0x401a449f56c16aa6ULL, 0x4ed3aa62363f7706ULL,
    0x1bfedf72429b023dULL, 0x37d0d724d00a1248ULL, 0xdb0fead349f1c09bULL, 0x075372c980991b7bULL,
};

static uint64_t parse_u64(const char *s) {
    uint64_t v = 0;
    for (; *s >= '0' && *s <= '9'; s++) v = v * 10 + (uint64_t)(*s - '0');
    return v;
}

/* the hot read: index the in-.text table. */
__attribute__((noinline)) static uint64_t lookup(unsigned i) {
    return TAB[i & 63];
}

__attribute__((noinline)) static uint64_t scramble(uint64_t x) {
    uint64_t r = 0x1234567;
    for (int k = 0; k < 8; k++) r = (r ^ lookup((unsigned)(x + k))) * 0x100000001b3ULL;
    return r;
}

int main(int argc, char **argv) {
    uint64_t acc = 0;
    for (int i = 1; i < argc; i++) acc ^= scramble(parse_u64(argv[i]));
    printf("acc=%016llx tab0=%016llx\n", (unsigned long long)acc, (unsigned long long)lookup(0));
    return (int)(acc & 0x7f);
}
