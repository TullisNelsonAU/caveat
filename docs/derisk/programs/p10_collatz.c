/* header-free; exit-code return; Collatz step-count accumulator (data-dependent branch) */
static int collatz_len(unsigned n) {
    int steps = 0;
    while (n > 1 && steps < 1000) {
        n = (n & 1) ? (3 * n + 1) : (n >> 1);
        steps++;
    }
    return steps;
}
int main(void) {
    int acc = 0;
    for (unsigned n = 1; n <= 120; n++)
        acc = (acc + collatz_len(n)) & 0x3fff;
    return acc & 0xff;
}
