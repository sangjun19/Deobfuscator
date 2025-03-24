#include <stdio.h>

int main() {
    int value = 20;
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
        case 3: {
            int result;
            result = value - 3;
            printf("3 선택됨, 결과: %d\n", result);
            break;
        }
        case 4: {
            int result;
            result = value / 2;
            printf("4 선택됨, 결과: %d\n", result);
            break;
        }
        case 5: {
            int result;
            result = value * value;
            printf("5 선택됨, 제곱 값: %d\n", result);
            break;
        }
        default:
            printf("기본값 실행됨\n");
            break;
    }
    return 0;
}
