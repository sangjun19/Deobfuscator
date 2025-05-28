#include <stdio.h>

int main() {
    int value = 7, sub_value = 1;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
<<<<<<< HEAD
                    printf("1-1 executed\n");
=======
                    printf("1-1 실행됨\n");
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
