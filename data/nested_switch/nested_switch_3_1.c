#include <stdio.h>

int main() {
    int value = 3, sub_value = 1;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 �����\n");
                    break;
            }
            break;
        default:
            printf("�⺻�� �����\n");
            break;
    }
    return 0;
}
