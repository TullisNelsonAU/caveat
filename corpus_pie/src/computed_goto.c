/* computed_goto — a GNU labels-as-values (`&&label`) threaded interpreter.
 *
 * `goto *dispatch[op]` is the classic computed-goto dispatch. At -O2 -fPIC -pie the
 * label-address table is a `pie_rel` 4-byte-relative table (the same lea-rip + movslq
 * idiom), so the M4 pie_rel resolver recovers the label targets — real interpreter
 * arms that are reachable only through the indirect jump. */
#include <stdint.h>

__attribute__((noinline)) uint64_t interp(const unsigned char *code, int n, uint64_t x) {
    static const void *const tab[] = {
        &&L_add, &&L_xor, &&L_shl, &&L_mul, &&L_sub, &&L_rot, &&L_not, &&L_end,
    };
    int pc = 0;
    uint64_t acc = x;
#define NEXT() do { if (pc >= n) goto L_end; goto *tab[code[pc++] & 7]; } while (0)
    NEXT();
L_add: acc += 0x9e3779b9; NEXT();
L_xor: acc ^= (acc << 13); NEXT();
L_shl: acc = (acc << 7) | (acc >> 57); NEXT();
L_mul: acc *= 6364136223846793005ULL; NEXT();
L_sub: acc -= 0xcafef00d; NEXT();
L_rot: acc = (acc >> 11) ^ (acc << 5); NEXT();
L_not: acc = ~acc; NEXT();
L_end: return acc;
}

int main(int argc, char **argv) {
    unsigned char prog[32];
    for (int i = 0; i < 32; i++) prog[i] = (unsigned char)(argc * i + 3);
    return (int)interp(prog, 32, (uint64_t)argc);
}
