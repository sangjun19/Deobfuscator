#include <stdio.h>

int main() {
    int value = 8;
    switch (value) {
        case 1: {
            int result;
            result = value * 10;
            printf("1 ���õ�, ���: %d\n", result);
            break;
        }
        default:
            printf("�⺻�� �����\n");
            break;
    }
    return 0;
}
