#include <stdio.h>

int main() {
    int value = 64;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 ���õ�, ���: %d\n", result);
            break;
        }
        case 2: {
            int result;
            result = value + 5;
            printf("2 ���õ�, ���: %d\n", result);
            break;
        }
        case 3: {
            int result;
            result = value - 3;
            printf("3 ���õ�, ���: %d\n", result);
            break;
        }
        case 4: {
            int result;
            result = value / 2;
            printf("4 ���õ�, ���: %d\n", result);
            break;
        }
        case 5: {
            int result;
            result = value * value;
            printf("5 ���õ�, ���� ��: %d\n", result);
            break;
        }
        case 6: {
            int result;
            result = value * 10;
            printf("6 ���õ�, ���: %d\n", result);
            break;
        }
        case 7: {
            int result;
            result = value + 5;
            printf("7 ���õ�, ���: %d\n", result);
            break;
        }
        case 8: {
            int result;
            result = value - 3;
            printf("8 ���õ�, ���: %d\n", result);
            break;
        }
        default:
            printf("�⺻�� �����\n");
            break;
    }
    return 0;
}
