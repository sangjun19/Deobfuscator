#include <stdio.h>

int main() {
    int value = 9, sub_value = 3;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
<<<<<<< HEAD
                    printf("1-1 executed\n");
                    break;
                case 2:
                    printf("1-2 executed\n");
                    break;
                case 3:
                    printf("1-3 executed\n");
=======
                    printf("1-1 실행됨\n");
                    break;
                case 2:
                    printf("1-2 실행됨\n");
                    break;
                case 3:
                    printf("1-3 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
                    break;
            }
            break;
        case 2:
            switch (sub_value) {
                case 1:
<<<<<<< HEAD
                    printf("2-1 executed\n");
                    break;
                case 2:
                    printf("2-2 executed\n");
                    break;
                case 3:
                    printf("2-3 executed\n");
=======
                    printf("2-1 실행됨\n");
                    break;
                case 2:
                    printf("2-2 실행됨\n");
                    break;
                case 3:
                    printf("2-3 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
                    break;
            }
            break;
        case 3:
            switch (sub_value) {
                case 1:
<<<<<<< HEAD
                    printf("3-1 executed\n");
                    break;
                case 2:
                    printf("3-2 executed\n");
                    break;
                case 3:
                    printf("3-3 executed\n");
=======
                    printf("3-1 실행됨\n");
                    break;
                case 2:
                    printf("3-2 실행됨\n");
                    break;
                case 3:
                    printf("3-3 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
                    break;
            }
            break;
        default:
<<<<<<< HEAD
            printf("default executed\n");
=======
            printf("기본값 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
            break;
    }
    return 0;
}
