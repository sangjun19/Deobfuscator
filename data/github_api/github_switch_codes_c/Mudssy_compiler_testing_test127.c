
#include <stdio.h>

int main() {
    #ifdef __llvm__ // llvm specific preprocessor macro
        printf("This is LLVM compiler.\n");
        switch (__llvm_utils_count()) { // llvmutilscount specific function
            case 0:
                printf("No utils found.\n");
                break;
            default:
                printf("%d utils found.\n", __llvm_utils_count());
                break;
        }
    #else
        printf("This is not an LLVM compiler.\n");
    #endif
    
    return 0;
}
