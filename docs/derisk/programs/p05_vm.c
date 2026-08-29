/* header-free; exit-code return; tiny stack-bytecode interpreter loop */
static int run(const unsigned char *code, int n) {
    int st[32], sp = 0;
    for (int ip = 0; ip < n; ip++) {
        unsigned char op = code[ip];
        switch (op & 7) {
            case 0: st[sp++] = code[++ip]; break;                 /* push imm */
            case 1: if (sp >= 2) { sp--; st[sp - 1] += st[sp]; } break;
            case 2: if (sp >= 2) { sp--; st[sp - 1] *= st[sp]; } break;
            case 3: if (sp >= 2) { sp--; st[sp - 1] ^= st[sp]; } break;
            case 4: if (sp >= 1) st[sp - 1] = -st[sp - 1]; break;
            case 5: if (sp >= 1 && st[sp - 1] == 0) ip++; break;  /* skip if zero */
            default: if (sp >= 1) st[sp - 1] &= 0xff; break;
        }
        if (sp < 0) sp = 0;
        if (sp > 31) sp = 31;
    }
    return sp > 0 ? st[sp - 1] : 0;
}
int main(void) {
    static const unsigned char prog[20] = {
        0,5, 0,9, 1, 0,3, 2, 0,7, 3, 4, 0,2, 2, 6, 0,4, 1, 6
    };
    return run(prog, 20) & 0xff;
}
