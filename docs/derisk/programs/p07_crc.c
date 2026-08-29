/* header-free; exit-code return; bitwise CRC-ish accumulator loop */
static unsigned crc_step(unsigned crc, unsigned char b) {
    crc ^= b;
    for (int k = 0; k < 8; k++)
        crc = (crc & 1) ? (crc >> 1) ^ 0xedb88320u : (crc >> 1);
    return crc;
}
int main(void) {
    unsigned crc = 0xffffffffu;
    for (int i = 0; i < 64; i++)
        crc = crc_step(crc, (unsigned char)(i * 37 + 11));
    crc ^= 0xffffffffu;
    return (int)(crc & 0xff);
}
