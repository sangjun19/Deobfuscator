#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 8;
    int value_3 = 2;
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
                    }
                    break;
                case 7:
                    switch (value_3) {
                        case 1:
                            printf("1-7-1\n");
                            break;
                        case 2:
                            printf("1-7-2\n");
                            break;
                    }
                    break;
                case 8:
                    switch (value_3) {
                        case 1:
                            printf("1-8-1\n");
                            break;
                        case 2:
                            printf("1-8-2\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
