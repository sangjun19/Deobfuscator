#include <stdio.h>

int main() {
    int value = 448;
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
        case 5: {
            int result;
            result = value * value;
            printf("5 selected, square: %d\n", result);
            break;
        }
        case 6: {
            int result;
            result = value * 10;
            printf("6 selected, result: %d\n", result);
            break;
        }
        case 7: {
            int result;
            result = value + 5;
            printf("7 selected, result: %d\n", result);
            break;
        }
        case 8: {
            int result;
            result = value - 3;
            printf("8 selected, result: %d\n", result);
            break;
        }
        case 9: {
            int result;
            result = value / 2;
            printf("9 selected, result: %d\n", result);
            break;
        }
        case 10: {
            int result;
            result = value * value;
            printf("10 selected, square: %d\n", result);
            break;
        }
        case 11: {
            int result;
            result = value * 10;
            printf("11 selected, result: %d\n", result);
            break;
        }
        case 12: {
            int result;
            result = value + 5;
            printf("12 selected, result: %d\n", result);
            break;
        }
        case 13: {
            int result;
            result = value - 3;
            printf("13 selected, result: %d\n", result);
            break;
        }
        case 14: {
            int result;
            result = value / 2;
            printf("14 selected, result: %d\n", result);
            break;
        }
        case 15: {
            int result;
            result = value * value;
            printf("15 selected, square: %d\n", result);
            break;
        }
        case 16: {
            int result;
            result = value * 10;
            printf("16 selected, result: %d\n", result);
            break;
        }
        default:
            printf("default\n");
            break;
    }
    return 0;
}
