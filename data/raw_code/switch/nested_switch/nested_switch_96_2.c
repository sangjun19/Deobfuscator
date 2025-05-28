#include <stdio.h>

int main() {
    int value = 96, sub_value = 2;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 executed\n");
                    break;
                case 2:
                    printf("1-2 executed\n");
                    break;
            }
            break;
        case 2:
            switch (sub_value) {
                case 1:
                    printf("2-1 executed\n");
                    break;
                case 2:
                    printf("2-2 executed\n");
                    break;
            }
            break;
        default:
            printf("default executed\n");
            break;
    }
    return 0;
}
