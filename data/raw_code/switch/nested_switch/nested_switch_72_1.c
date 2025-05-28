#include <stdio.h>

int main() {
    int value = 72, sub_value = 1;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 executed\n");
                    break;
            }
            break;
        default:
            printf("default executed\n");
            break;
    }
    return 0;
}
