/* vtable_shapes — real C++ virtual dispatch over a small class hierarchy.
 *
 * Each concrete class has a vtable in `.data.rel.ro` whose virtual-fn slots are, in a
 * PIE image, R_X86_64_RELATIVE relocations whose addends point at the virtual methods
 * (which are FUNC symbols). This is what the M4 `vtable` resolver keys on. Note the
 * honest subtlety this specimen exists to expose: those same slots are ALSO read by
 * M3's relocation pass, so on PIE the vtable resolver re-tags rather than adds. The
 * virtual methods are only ever called virtually (no direct call site). */
#include <cstdint>

struct Shape {
    virtual uint64_t area() const = 0;
    virtual uint64_t perimeter() const = 0;
    virtual uint64_t hash() const = 0;
    virtual ~Shape() {}
};
struct Square : Shape {
    uint64_t s; explicit Square(uint64_t s) : s(s) {}
    uint64_t area() const override { return s * s; }
    uint64_t perimeter() const override { return 4 * s; }
    uint64_t hash() const override { return s * 0x9e3779b97f4a7c15ULL; }
};
struct Circle : Shape {
    uint64_t r; explicit Circle(uint64_t r) : r(r) {}
    uint64_t area() const override { return 3 * r * r; }
    uint64_t perimeter() const override { return 6 * r; }
    uint64_t hash() const override { return (r << 13) ^ (r >> 7); }
};
struct Triangle : Shape {
    uint64_t b, h; Triangle(uint64_t b, uint64_t h) : b(b), h(h) {}
    uint64_t area() const override { return b * h / 2; }
    uint64_t perimeter() const override { return 3 * b + h; }
    uint64_t hash() const override { return b * 31 + h; }
};

__attribute__((noinline)) uint64_t fold(Shape *const *v, int n) {
    uint64_t acc = 0;
    for (int i = 0; i < n; i++) acc += v[i]->area() ^ v[i]->perimeter() ^ v[i]->hash();
    return acc;
}

int main(int argc, char **) {
    Square a(argc);
    Circle b(argc + 1);
    Triangle c(argc + 2, argc + 3);
    Shape *v[] = {&a, &b, &c};
    return (int)fold(v, 3);
}
