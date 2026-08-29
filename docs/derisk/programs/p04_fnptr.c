/* header-free; exit-code return; function-pointer dispatch table */
static int op_add(int a, int b) { return a + b; }
static int op_mul(int a, int b) { return a * b; }
static int op_xor(int a, int b) { return a ^ b; }
static int op_sub(int a, int b) { return a - b; }
int main(void) {
    int (*tab[4])(int, int) = { op_add, op_mul, op_xor, op_sub };
    int acc = 1;
    for (int i = 0; i < 32; i++) {
        int f = (i * 7 + 3) & 3;
        acc = tab[f](acc, i + 1) & 0xffff;
    }
    return acc & 0xff;
}
