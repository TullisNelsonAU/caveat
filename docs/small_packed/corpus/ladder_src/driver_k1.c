/* freestanding driver, k=1 units */
int main_p01_statemachine(void);
static int run(void){ int s=0;
  s += main_p01_statemachine();
  return s & 0xff; }
void _start(void) {
    int r = run();
    __asm__ volatile("syscall" :: "a"(60L), "D"((long)r) : "memory");
    __builtin_unreachable();
}
