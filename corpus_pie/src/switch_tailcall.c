/* switch_tailcall — a dense switch of tail-calls to distinct noinline functions.
 *
 * The switch lowers to a `pie_rel` 4-byte-relative jump table; each case is a small
 * trampoline `jmp fN`. The target functions fN are reached ONLY through the table and
 * are never called directly, so at M3 they sit in the unconfirmed indirect tail. This
 * probes whether pie_rel resolution reaches the FUNCTION axis (the case trampolines are
 * intra-function blocks, so the honest expectation is instruction-axis coverage of the
 * trampolines, with fN confirmed only if a tail-jump edge carries confirmation). */
#include <stdint.h>

__attribute__((noinline)) uint64_t h0(uint64_t x){ return x*3 + 1; }
__attribute__((noinline)) uint64_t h1(uint64_t x){ return x ^ 0x5a5a5a5a5a5a5a5aULL; }
__attribute__((noinline)) uint64_t h2(uint64_t x){ return (x<<7) | (x>>57); }
__attribute__((noinline)) uint64_t h3(uint64_t x){ return x*x + 7; }
__attribute__((noinline)) uint64_t h4(uint64_t x){ return ~x + 99; }
__attribute__((noinline)) uint64_t h5(uint64_t x){ return x*11400714819323198485ULL; }
__attribute__((noinline)) uint64_t h6(uint64_t x){ return (x>>3) ^ (x<<11); }
__attribute__((noinline)) uint64_t h7(uint64_t x){ return x - 0xdeadbeef; }
__attribute__((noinline)) uint64_t h8(uint64_t x){ return x*7 + (x>>2); }
__attribute__((noinline)) uint64_t h9(uint64_t x){ return (x^0xff00ff00) * 3; }

__attribute__((noinline)) uint64_t dispatch(int sel, uint64_t x) {
    switch (sel) {
        case 0: return h0(x); case 1: return h1(x); case 2: return h2(x);
        case 3: return h3(x); case 4: return h4(x); case 5: return h5(x);
        case 6: return h6(x); case 7: return h7(x); case 8: return h8(x);
        case 9: return h9(x); default: return x;
    }
}

int main(int argc, char **argv) {
    uint64_t acc = 0;
    for (int i = 0; i < argc; i++) acc += dispatch(i % 10, (uint64_t)(uintptr_t)argv[i]);
    return (int)acc;
}
