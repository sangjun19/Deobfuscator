#include <stdio.h>

int main() {
    int value = 6, sub_value = 2;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 실행됨\n");
                    break;
                case 2:
                    printf("1-2 실행됨\n");
                    break;
            }
            break;
        case 2:
            switch (sub_value) {
                case 1:
                    printf("2-1 실행됨\n");
                    break;
                case 2:
                    printf("2-2 실행됨\n");
                    break;
            }
            break;
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
