/* header-free; exit-code return; Sieve of Eratosthenes over a fixed range */
int main(void) {
    unsigned char sieve[128];
    for (int i = 0; i < 128; i++) sieve[i] = 1;
    sieve[0] = sieve[1] = 0;
    for (int p = 2; p * p < 128; p++)
        if (sieve[p])
            for (int m = p * p; m < 128; m += p) sieve[m] = 0;
    int acc = 0, cnt = 0;
    for (int i = 0; i < 128; i++)
        if (sieve[i]) { cnt++; acc = (acc + i * cnt) & 0x3fff; }
    return (acc ^ cnt) & 0xff;
}
