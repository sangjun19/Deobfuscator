#include <stdio.h>

int main() {
    int value = 140;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 selected, result: %d\n", result);
            break;
        }
        case 2: {
            int result;
            result = value + 5;
            printf("2 selected, result: %d\n", result);
            break;
        }
        case 3: {
            int result;
            result = value - 3;
            printf("3 selected, result: %d\n", result);
            break;
        }
        case 4: {
            int result;
            result = value / 2;
            printf("4 selected, result: %d\n", result);
            break;
        }
        default:
            printf("default\n");
            break;
    }
    return 0;
}
