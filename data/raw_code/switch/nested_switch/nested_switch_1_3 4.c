#include <stdio.h>

int main() {
    int value = 1, sub_value = 3;
    switch (value) {
        case 1:
<<<<<<< HEAD
            printf("Case 1 executed\n");
            break;
        case 2:
            printf("Case 2 executed\n");
            break;
        case 3:
            printf("Case 3 executed\n");
            break;
        default:
            printf("default executed\n");
=======
            printf("Case 1 실행됨\n");
            break;
        case 2:
            printf("Case 2 실행됨\n");
            break;
        case 3:
            printf("Case 3 실행됨\n");
            break;
        default:
            printf("기본값 실행됨\n");
>>>>>>> 5175742c82fb47275e31077cbfc274fefa466250
            break;
    }
    return 0;
}
