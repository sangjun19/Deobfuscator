#include <stdio.h>

int main() {
    int value = 10;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
<<<<<<< HEAD
            printf("1 selected, result: %d\n", result);
=======
            printf("1 선택됨, 결과: %d\n", result);
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
            break;
        }
        case 2: {
            int result;
            result = value + 5;
<<<<<<< HEAD
            printf("2 selected, result: %d\n", result);
            break;
        }
        default:
            printf("default\n");
=======
            printf("2 선택됨, 결과: %d\n", result);
            break;
        }
        default:
            printf("기본값 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
            break;
    }
    return 0;
}
