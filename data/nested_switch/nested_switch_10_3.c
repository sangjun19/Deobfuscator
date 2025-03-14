#include <stdio.h>

int main() {
    int value = 10, sub_value = 3;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 실행됨\n");
                    break;
                case 2:
                    printf("1-2 실행됨\n");
                    break;
                case 3:
                    printf("1-3 실행됨\n");
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
                case 3:
                    printf("2-3 실행됨\n");
                    break;
            }
            break;
        case 3:
            switch (sub_value) {
                case 1:
                    printf("3-1 실행됨\n");
                    break;
                case 2:
                    printf("3-2 실행됨\n");
                    break;
                case 3:
                    printf("3-3 실행됨\n");
                    break;
            }
            break;
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
