/* vmbig — a legitimate stack-based bytecode VM with a large opcode dispatch switch.
 *
 * This is the FP-gate control the CFG-topology probe MUST NOT flag as obfuscated: a real
 * interpreter has exactly the surface features a naive flattening detector keys on — one hot
 * dispatch block, a switch/computed-jump every case routes back to, high in-degree on the loop
 * head. If the probe fires on this, it fires on every legit VM and is worthless. Written by hand
 * (not obfuscated), compiled with the same flags as the Tigress corpus. ~40 opcodes so the switch
 * is genuinely big, not a toy. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

enum {
    OP_HALT, OP_PUSH, OP_POP, OP_DUP, OP_SWAP, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG, OP_AND, OP_OR, OP_XOR, OP_NOT, OP_SHL, OP_SHR, OP_EQ, OP_NE, OP_LT,
    OP_LE, OP_GT, OP_GE, OP_JMP, OP_JZ, OP_JNZ, OP_LOAD, OP_STORE, OP_PRINT, OP_INC,
    OP_DEC, OP_ROT, OP_OVER, OP_NOP, OP_CALL, OP_RET, OP_ALLOC, OP_FREE, OP_CMP, OP_ABS,
    OP_MAX_
};

#define STK 256
#define MEM 256

static long run(const uint8_t *code, long n) {
    long stack[STK]; int sp = 0;
    long mem[MEM]; memset(mem, 0, sizeof mem);
    long rstack[STK]; int rsp = 0;
    long pc = 0;
    long steps = 0;
    while (pc < n) {
        if (++steps > 1000000) break;              /* watchdog */
        uint8_t op = code[pc++];
        switch (op) {                              /* the legit dispatcher */
        case OP_HALT:  return sp ? stack[sp-1] : 0;
        case OP_PUSH:  stack[sp++] = (int8_t)code[pc++]; break;
        case OP_POP:   if (sp) sp--; break;
        case OP_DUP:   if (sp) { stack[sp] = stack[sp-1]; sp++; } break;
        case OP_SWAP:  if (sp>=2) { long t=stack[sp-1]; stack[sp-1]=stack[sp-2]; stack[sp-2]=t; } break;
        case OP_ADD:   if (sp>=2) { sp--; stack[sp-1]+=stack[sp]; } break;
        case OP_SUB:   if (sp>=2) { sp--; stack[sp-1]-=stack[sp]; } break;
        case OP_MUL:   if (sp>=2) { sp--; stack[sp-1]*=stack[sp]; } break;
        case OP_DIV:   if (sp>=2 && stack[sp-1]) { sp--; stack[sp-1]/=stack[sp]; } break;
        case OP_MOD:   if (sp>=2 && stack[sp-1]) { sp--; stack[sp-1]%=stack[sp]; } break;
        case OP_NEG:   if (sp) stack[sp-1]=-stack[sp-1]; break;
        case OP_AND:   if (sp>=2) { sp--; stack[sp-1]&=stack[sp]; } break;
        case OP_OR:    if (sp>=2) { sp--; stack[sp-1]|=stack[sp]; } break;
        case OP_XOR:   if (sp>=2) { sp--; stack[sp-1]^=stack[sp]; } break;
        case OP_NOT:   if (sp) stack[sp-1]=~stack[sp-1]; break;
        case OP_SHL:   if (sp>=2) { sp--; stack[sp-1]<<=(stack[sp]&63); } break;
        case OP_SHR:   if (sp>=2) { sp--; stack[sp-1]>>=(stack[sp]&63); } break;
        case OP_EQ:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]==stack[sp]; } break;
        case OP_NE:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]!=stack[sp]; } break;
        case OP_LT:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]<stack[sp]; } break;
        case OP_LE:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]<=stack[sp]; } break;
        case OP_GT:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]>stack[sp]; } break;
        case OP_GE:    if (sp>=2) { sp--; stack[sp-1]=stack[sp-1]>=stack[sp]; } break;
        case OP_JMP:   pc = (uint8_t)code[pc]; break;
        case OP_JZ:    if (sp && !stack[--sp]) pc=(uint8_t)code[pc]; else pc++; break;
        case OP_JNZ:   if (sp && stack[--sp]) pc=(uint8_t)code[pc]; else pc++; break;
        case OP_LOAD:  { int a=(uint8_t)code[pc++]; stack[sp++]=mem[a&(MEM-1)]; } break;
        case OP_STORE: { int a=(uint8_t)code[pc++]; if (sp) mem[a&(MEM-1)]=stack[--sp]; } break;
        case OP_PRINT: if (sp) printf("%ld\n", stack[sp-1]); break;
        case OP_INC:   if (sp) stack[sp-1]++; break;
        case OP_DEC:   if (sp) stack[sp-1]--; break;
        case OP_ROT:   if (sp>=3) { long t=stack[sp-3]; stack[sp-3]=stack[sp-2]; stack[sp-2]=stack[sp-1]; stack[sp-1]=t; } break;
        case OP_OVER:  if (sp>=2) { stack[sp]=stack[sp-2]; sp++; } break;
        case OP_NOP:   break;
        case OP_CALL:  rstack[rsp++]=pc+1; pc=(uint8_t)code[pc]; break;
        case OP_RET:   if (rsp) pc=rstack[--rsp]; break;
        case OP_ALLOC: if (sp) { int a=(uint8_t)stack[sp-1]; mem[a&(MEM-1)]=0; } break;
        case OP_FREE:  if (sp) sp--; break;
        case OP_CMP:   if (sp>=2) { long d=stack[sp-2]-stack[sp-1]; sp--; stack[sp-1]=(d>0)-(d<0); } break;
        case OP_ABS:   if (sp) { long v=stack[sp-1]; stack[sp-1]=v<0?-v:v; } break;
        default:       return -1;                  /* bad opcode */
        }
    }
    return sp ? stack[sp-1] : 0;
}

int main(int argc, char **argv) {
    /* A tiny program: push 7, push 5, add, dup, mul, print, halt — plus a loop. */
    uint8_t prog[] = {
        OP_PUSH, 7, OP_PUSH, 5, OP_ADD, OP_DUP, OP_MUL, OP_PRINT,
        OP_PUSH, 10, OP_STORE, 0,
        OP_LOAD, 0, OP_DEC, OP_DUP, OP_STORE, 0, OP_JNZ, 12,
        OP_PUSH, 42, OP_PRINT, OP_HALT
    };
    long r = run(prog, sizeof prog);
    if (argc > 99) printf("%s", argv[0]);   /* keep argv live */
    return (int)(r & 0x7f);
}
