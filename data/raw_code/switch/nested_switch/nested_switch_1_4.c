#include <stdio.h>

int main() {
    int value = 1, sub_value = 4;
    switch (value) {
        case 1:
            printf("Case 1 executed\n");
            break;
        case 2:
            printf("Case 2 executed\n");
            break;
        case 3:
            printf("Case 3 executed\n");
            break;
        case 4:
            printf("Case 4 executed\n");
            break;
        default:
            printf("default executed\n");
            break;
    }
    return 0;
}
