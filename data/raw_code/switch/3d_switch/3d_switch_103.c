#include <stdio.h>

int main() {
    int value_1 = 2;
    int value_2 = 1;
    int value_3 = 3;
    switch (value_1) {
        case 1:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("1-1-1\n");
                            break;
                        case 2:
                            printf("1-1-2\n");
                            break;
                        case 3:
                            printf("1-1-3\n");
                            break;
                    }
                    break;
            }
            break;
        case 2:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("2-1-1\n");
                            break;
                        case 2:
                            printf("2-1-2\n");
                            break;
                        case 3:
                            printf("2-1-3\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
