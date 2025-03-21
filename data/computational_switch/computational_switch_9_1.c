#include <stdio.h>

int main() {
    int value = 9;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 selected, result: %d\n", result);
            break;
        }
        default:
            printf("default\n");
            break;
    }
    return 0;
}
