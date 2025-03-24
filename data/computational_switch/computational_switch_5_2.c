#include <stdio.h>

int main() {
    int value = 10;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 선택됨, 결과: %d\n", result);
            break;
        }
        case 2: {
            int result;
            result = value + 5;
            printf("2 선택됨, 결과: %d\n", result);
            break;
        }
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
