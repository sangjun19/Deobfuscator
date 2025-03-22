#include <stdio.h>

int main() {
    int value = 5, sub_value = 5;
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
                case 4:
                    printf("1-4 실행됨\n");
                    break;
                case 5:
                    printf("1-5 실행됨\n");
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
                case 4:
                    printf("2-4 실행됨\n");
                    break;
                case 5:
                    printf("2-5 실행됨\n");
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
                case 4:
                    printf("3-4 실행됨\n");
                    break;
                case 5:
                    printf("3-5 실행됨\n");
                    break;
            }
            break;
        case 4:
            switch (sub_value) {
                case 1:
                    printf("4-1 실행됨\n");
                    break;
                case 2:
                    printf("4-2 실행됨\n");
                    break;
                case 3:
                    printf("4-3 실행됨\n");
                    break;
                case 4:
                    printf("4-4 실행됨\n");
                    break;
                case 5:
                    printf("4-5 실행됨\n");
                    break;
            }
            break;
        case 5:
            switch (sub_value) {
                case 1:
                    printf("5-1 실행됨\n");
                    break;
                case 2:
                    printf("5-2 실행됨\n");
                    break;
                case 3:
                    printf("5-3 실행됨\n");
                    break;
                case 4:
                    printf("5-4 실행됨\n");
                    break;
                case 5:
                    printf("5-5 실행됨\n");
                    break;
            }
            break;
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
