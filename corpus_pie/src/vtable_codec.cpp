/* vtable_codec — a second, deeper virtual hierarchy (handler chain) for the vtable
 * resolver. More virtual methods per class and a two-level hierarchy so the vtables in
 * `.data.rel.ro` are larger. Same PIE reality as vtable_shapes: the slots are
 * R_X86_64_RELATIVE relocs (M3's reloc pass sees them too). */
#include <cstdint>

struct Codec {
    virtual uint64_t encode(uint64_t) const = 0;
    virtual uint64_t decode(uint64_t) const = 0;
    virtual uint64_t checksum(uint64_t) const = 0;
    virtual uint64_t rekey(uint64_t) const = 0;
    virtual ~Codec() {}
};
struct Xor : Codec {
    uint64_t k; explicit Xor(uint64_t k) : k(k) {}
    uint64_t encode(uint64_t x) const override { return x ^ k; }
    uint64_t decode(uint64_t x) const override { return x ^ k; }
    uint64_t checksum(uint64_t x) const override { return (x + k) * 2654435761u; }
    uint64_t rekey(uint64_t x) const override { return k ^ (x << 3); }
};
struct Rot : Codec {
    uint64_t n; explicit Rot(uint64_t n) : n(n & 63) {}
    uint64_t encode(uint64_t x) const override { return (x << n) | (x >> (64 - n)); }
    uint64_t decode(uint64_t x) const override { return (x >> n) | (x << (64 - n)); }
    uint64_t checksum(uint64_t x) const override { return x ^ (n * 0x100000001b3ULL); }
    uint64_t rekey(uint64_t x) const override { return (n + x) & 63; }
};
struct Feistel : Codec {
    uint64_t k; explicit Feistel(uint64_t k) : k(k) {}
    uint64_t encode(uint64_t x) const override { uint64_t l = x >> 32, r = x & 0xffffffff; return (r << 32) | ((l ^ (r * k)) & 0xffffffff); }
    uint64_t decode(uint64_t x) const override { uint64_t r = x >> 32, l = x & 0xffffffff; return (l << 32) | ((r ^ (l * k)) & 0xffffffff); }
    uint64_t checksum(uint64_t x) const override { return x * k + 0xdeadbeef; }
    uint64_t rekey(uint64_t x) const override { return k * 6364136223846793005ULL + x; }
};

__attribute__((noinline)) uint64_t pipeline(Codec *const *c, int n, uint64_t x) {
    uint64_t acc = x;
    for (int i = 0; i < n; i++) {
        acc = c[i]->encode(acc);
        acc = c[i]->checksum(acc) ^ c[i]->decode(acc) ^ c[i]->rekey(acc);
    }
    return acc;
}

int main(int argc, char **) {
    Xor a(argc * 0x77u);
    Rot b(argc + 5);
    Feistel c(argc | 1);
    Codec *v[] = {&a, &b, &c};
    return (int)pipeline(v, 3, (uint64_t)argc);
}
