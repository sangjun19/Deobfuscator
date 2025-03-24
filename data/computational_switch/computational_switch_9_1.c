#include <stdio.h>

int main() {
    int value = 9;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 선택됨, 결과: %d\n", result);
            break;
        }
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
