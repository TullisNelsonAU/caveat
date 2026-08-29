/* freestanding driver, k=10 units */
int main_p01_statemachine(void);
int main_p02_insertsort(void);
int main_p05_vm(void);
int main_p06_parser(void);
int main_p07_crc(void);
int main_p08_binsearch(void);
int main_p09_matmul(void);
int main_p10_collatz(void);
int main_p12_modpow(void);
int main_p03_ackermann(void);
static int run(void){ int s=0;
  s += main_p01_statemachine();
  s += main_p02_insertsort();
  s += main_p05_vm();
  s += main_p06_parser();
  s += main_p07_crc();
  s += main_p08_binsearch();
  s += main_p09_matmul();
  s += main_p10_collatz();
  s += main_p12_modpow();
  s += main_p03_ackermann();
  return s & 0xff; }
void _start(void) {
    int r = run();
    __asm__ volatile("syscall" :: "a"(60L), "D"((long)r) : "memory");
    __builtin_unreachable();
}
