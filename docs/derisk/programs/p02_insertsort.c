/* header-free; exit-code return; insertion sort + checksum over the sorted array */
static void isort(int *a, int n) {
    for (int i = 1; i < n; i++) {
        int k = a[i], j = i - 1;
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
        a[j + 1] = k;
    }
}
int main(void) {
    int a[16] = { 42, 7, 99, 3, 250, 17, 6, 88, 1, 200, 13, 64, 32, 9, 123, 5 };
    isort(a, 16);
    int acc = 0;
    for (int i = 0; i < 16; i++) acc = (acc * 31 + a[i]) & 0x3fff;
    return acc & 0xff;
}
