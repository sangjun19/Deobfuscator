#include <stdio.h>

int main() {
    int value = 30;
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
        default:
            printf("�⺻�� �����\n");
            break;
    }
    return 0;
}
