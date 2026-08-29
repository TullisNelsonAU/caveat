/* header-free; exit-code return; recursion (Ackermann, bounded) */
static int ack(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ack(m - 1, 1);
    return ack(m - 1, ack(m, n - 1));
}
int main(void) {
    int acc = 0;
    for (int m = 0; m <= 2; m++)
        for (int n = 0; n <= 4; n++)
            acc = (acc + ack(m, n)) & 0x3ff;
    return acc & 0xff;
}
