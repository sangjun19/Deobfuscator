// Repository: ahyangyi/flat-chipyard
// File: toolchains/riscv-tools/riscv-gnu-toolchain/riscv-gcc/gcc/testsuite/gdc.test/fail_compilation/fail3144.d

/*
TEST_OUTPUT:
---
fail_compilation/fail3144.d(12): Error: break is not inside a loop or switch
fail_compilation/fail3144.d(15): Error: break is not inside a loop or switch
---
*/

void main()
{
    switch (1)
        default: {} break;

    final switch (1)
        case 1: {} break;
}
