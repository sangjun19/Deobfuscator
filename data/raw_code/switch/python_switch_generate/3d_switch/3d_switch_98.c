#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 1;
    int value_3 = 10;
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
                        case 4:
                            printf("1-1-4\n");
                            break;
                        case 5:
                            printf("1-1-5\n");
                            break;
                        case 6:
                            printf("1-1-6\n");
                            break;
                        case 7:
                            printf("1-1-7\n");
                            break;
                        case 8:
                            printf("1-1-8\n");
                            break;
                        case 9:
                            printf("1-1-9\n");
                            break;
                        case 10:
                            printf("1-1-10\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
