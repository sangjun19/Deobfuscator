#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 3;
    int value_3 = 8;
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
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("1-2-1\n");
                            break;
                        case 2:
                            printf("1-2-2\n");
                            break;
                        case 3:
                            printf("1-2-3\n");
                            break;
                        case 4:
                            printf("1-2-4\n");
                            break;
                        case 5:
                            printf("1-2-5\n");
                            break;
                        case 6:
                            printf("1-2-6\n");
                            break;
                        case 7:
                            printf("1-2-7\n");
                            break;
                        case 8:
                            printf("1-2-8\n");
                            break;
                    }
                    break;
                case 3:
                    switch (value_3) {
                        case 1:
                            printf("1-3-1\n");
                            break;
                        case 2:
                            printf("1-3-2\n");
                            break;
                        case 3:
                            printf("1-3-3\n");
                            break;
                        case 4:
                            printf("1-3-4\n");
                            break;
                        case 5:
                            printf("1-3-5\n");
                            break;
                        case 6:
                            printf("1-3-6\n");
                            break;
                        case 7:
                            printf("1-3-7\n");
                            break;
                        case 8:
                            printf("1-3-8\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
