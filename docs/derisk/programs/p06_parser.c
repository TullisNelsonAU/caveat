/* header-free; exit-code return; char-array expression tokenizer/evaluator (left-to-right) */
static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int eval(const char *s, int n) {
    int acc = 0, cur = 0, op = '+';
    for (int i = 0; i <= n; i++) {
        int c = (i < n) ? s[i] : 0;
        if (is_digit(c)) {
            cur = cur * 10 + (c - '0');
        } else {
            switch (op) {
                case '+': acc += cur; break;
                case '-': acc -= cur; break;
                case '*': acc *= cur; break;
                default: break;
            }
            op = c; cur = 0;
        }
    }
    return acc;
}
int main(void) {
    static const char expr[] = "12+7*3-4+56*2-9+100";
    int n = 0; while (expr[n]) n++;
    return eval(expr, n) & 0xff;
}
