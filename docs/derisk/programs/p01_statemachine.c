/* header-free; exit-code return; switch-dispatch state machine over a fixed byte stream */
static int trans(int s, int c) {
    switch (s) {
        case 0: return (c & 1) ? 1 : 2;
        case 1: return (c > 100) ? 3 : 0;
        case 2: return (c % 3 == 0) ? 3 : 1;
        case 3: return (c & 4) ? 0 : 2;
        default: return 0;
    }
}
int main(void) {
    static const unsigned char stream[24] = {
        3,17,99,4,250,11,6,7,88,42,13,200,
        1,255,64,32,9,5,123,77,180,2,44,66
    };
    int s = 0, acc = 0;
    for (int i = 0; i < 24; i++) {
        s = trans(s, stream[i]);
        acc = (acc + s * (i + 1)) & 0x7ff;
    }
    return acc & 0xff;
}
