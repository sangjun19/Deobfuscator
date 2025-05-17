#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 6;
    int value_3 = 4;
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
                    }
                    break;
                case 4:
                    switch (value_3) {
                        case 1:
                            printf("1-4-1\n");
                            break;
                        case 2:
                            printf("1-4-2\n");
                            break;
                        case 3:
                            printf("1-4-3\n");
                            break;
                        case 4:
                            printf("1-4-4\n");
                            break;
                    }
                    break;
                case 5:
                    switch (value_3) {
                        case 1:
                            printf("1-5-1\n");
                            break;
                        case 2:
                            printf("1-5-2\n");
                            break;
                        case 3:
                            printf("1-5-3\n");
                            break;
                        case 4:
                            printf("1-5-4\n");
                            break;
                    }
                    break;
                case 6:
                    switch (value_3) {
                        case 1:
                            printf("1-6-1\n");
                            break;
                        case 2:
                            printf("1-6-2\n");
                            break;
                        case 3:
                            printf("1-6-3\n");
                            break;
                        case 4:
                            printf("1-6-4\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
