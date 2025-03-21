#include <stdio.h>

int main() {
    int value = 3, sub_value = 1;
    switch (value) {
        case 1:
            switch (sub_value) {
                case 1:
                    printf("1-1 ½ÇÇàµÊ\n");
                    break;
            }
            break;
        default:
            printf("±âº»°ª ½ÇÇàµÊ\n");
            break;
    }
    return 0;
}
