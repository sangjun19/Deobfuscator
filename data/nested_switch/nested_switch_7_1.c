#include <stdio.h>

int main() {
    int value = 7, sub_value = 1;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 실행됨\n");
                    break;
            }
            break;
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
