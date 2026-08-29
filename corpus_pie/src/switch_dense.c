/* switch_dense — a dense switch whose cases are SUBSTANTIAL inline blocks.
 *
 * At -O2 -fPIC -pie GCC lowers the dense switch to a jump table of 4-byte SIGNED
 * self-relative offsets, addressed by `lea reg,[rip+disp]` + `movslq (base,idx,4)` +
 * `add` + `jmp reg`. That table is INVISIBLE to M3's data-anchored scan (it reads
 * 8-byte absolute words) and is exactly what the M4 `pie_rel` resolver recovers. The
 * case bodies are real, reachable-only-through-the-table code — so confirming them is
 * an instruction-axis generalization of decoy suppression to a 4-byte-relative edge. */
#include <stdint.h>

static volatile uint64_t sink;

__attribute__((noinline)) uint64_t run(int sel, uint64_t x) {
    uint64_t a = x, b = x ^ 0x9e3779b97f4a7c15ULL;
    switch (sel & 15) {
        case 0:  a = a * 6364136223846793005ULL + 1442695040888963407ULL; b ^= a >> 17; break;
        case 1:  a += b; a = (a << 13) | (a >> 51); b -= a; break;
        case 2:  a ^= b << 7; b ^= a >> 9; a += 0x1234567; break;
        case 3:  a = a * a + b; b = b * 3 + 7; a ^= b; break;
        case 4:  a -= b; a = (a >> 5) | (a << 59); b += a * 5; break;
        case 5:  a ^= 0xdeadbeefcafef00dULL; b = a - b; a = b * b; break;
        case 6:  a += b * 9; b ^= a; a = (a << 21) | (a >> 43); break;
        case 7:  a = ~a; b = b ^ (a << 3); a += b >> 2; break;
        case 8:  a = a * 11400714819323198485ULL; b ^= a >> 29; a += b; break;
        case 9:  a ^= b; a = (a << 8) | (a >> 56); b = a * 7 + 3; break;
        case 10: a += 0x0f0f0f0f; b -= a; a ^= b << 11; break;
        case 11: a = a * b + 1; b = (b << 17) | (b >> 47); a ^= b; break;
        case 12: a -= 0x55555555; a = (a >> 7) | (a << 57); b += a; break;
        case 13: a ^= b >> 4; b = b * 13; a += b ^ 0x777; break;
        case 14: a = a + b + (a << 6); b ^= a >> 15; break;
        default: a ^= 0xffffffffffffffffULL; b = a; break;
    }
    return a ^ (b * 0xff51afd7ed558ccdULL);
}

int main(int argc, char **argv) {
    uint64_t acc = 0;
    for (int i = 0; i < argc; i++) acc += run(i, (uint64_t)(uintptr_t)argv[i]);
    sink = acc;
    return (int)acc;
}
