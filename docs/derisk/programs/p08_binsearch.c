/* header-free; exit-code return; binary search over a sorted table, many queries */
static int bsearch_i(const int *a, int n, int key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] == key) return mid;
        if (a[mid] < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
int main(void) {
    static const int tab[20] = {
        2,5,9,13,18,24,31,37,44,52,61,70,80,91,103,116,130,145,161,178
    };
    int acc = 0;
    for (int q = 0; q < 200; q += 3) {
        int r = bsearch_i(tab, 20, q);
        acc = (acc + (r < 0 ? 7 : r)) & 0x7ff;
    }
    return acc & 0xff;
}
