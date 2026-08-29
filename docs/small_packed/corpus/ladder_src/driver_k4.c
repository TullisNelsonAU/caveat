/* freestanding driver, k=4 units */
int main_p01_statemachine(void);
int main_p02_insertsort(void);
int main_p05_vm(void);
int main_p06_parser(void);
static int run(void){ int s=0;
  s += main_p01_statemachine();
  s += main_p02_insertsort();
  s += main_p05_vm();
  s += main_p06_parser();
  return s & 0xff; }
void _start(void) {
    int r = run();
    __asm__ volatile("syscall" :: "a"(60L), "D"((long)r) : "memory");
    __builtin_unreachable();
}
