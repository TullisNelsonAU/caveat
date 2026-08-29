/* header-free; exit-code return; nested-loop 4x4 integer matrix multiply */
static void matmul(const int A[4][4], const int B[4][4], int C[4][4]) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int s = 0;
            for (int k = 0; k < 4; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}
int main(void) {
    int A[4][4], B[4][4], C[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) { A[i][j] = (i * 4 + j + 1); B[i][j] = ((i ^ j) + 2); }
    matmul(A, B, C);
    int acc = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc = (acc + C[i][j]) & 0x3fff;
    return acc & 0xff;
}
