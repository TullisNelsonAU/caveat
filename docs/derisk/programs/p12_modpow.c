/* header-free; exit-code return; Euclid gcd + modular exponentiation (loops + recursion) */
static unsigned gcd(unsigned a, unsigned b) {
    while (b) { unsigned t = a % b; a = b; b = t; }
    return a;
}
static unsigned modpow(unsigned base, unsigned exp, unsigned mod) {
    unsigned r = 1;
    base %= mod;
    while (exp) {
        if (exp & 1) r = (r * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return r;
}
int main(void) {
    unsigned acc = 0;
    for (unsigned a = 1; a <= 30; a++) {
        acc += gcd(a * 7 + 3, a * 3 + 11);
        acc += modpow(a + 2, a, 251);
        acc &= 0x3fff;
    }
    return (int)(acc & 0xff);
}
