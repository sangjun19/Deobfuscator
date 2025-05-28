#include <stdio.h>

int main() {
    int value = 43, sub_value = 4;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 executed\n");
                    break;
                case 2:
                    printf("1-2 executed\n");
                    break;
                case 3:
                    printf("1-3 executed\n");
                    break;
                case 4:
                    printf("1-4 executed\n");
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
                case 3:
                    printf("2-3 executed\n");
                    break;
                case 4:
                    printf("2-4 executed\n");
                    break;
            }
            break;
        case 3:
            switch (sub_value) {
                case 1:
                    printf("3-1 executed\n");
                    break;
                case 2:
                    printf("3-2 executed\n");
                    break;
                case 3:
                    printf("3-3 executed\n");
                    break;
                case 4:
                    printf("3-4 executed\n");
                    break;
            }
            break;
        case 4:
            switch (sub_value) {
                case 1:
                    printf("4-1 executed\n");
                    break;
                case 2:
                    printf("4-2 executed\n");
                    break;
                case 3:
                    printf("4-3 executed\n");
                    break;
                case 4:
                    printf("4-4 executed\n");
                    break;
            }
            break;
        default:
            printf("default executed\n");
            break;
    }
    return 0;
}
