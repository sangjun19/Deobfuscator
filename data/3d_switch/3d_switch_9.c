#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 3;
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
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("2-2-1\n");
                            break;
                        case 2:
                            printf("2-2-2\n");
                            break;
                        case 3:
                            printf("2-2-3\n");
                            break;
                    }
                    break;
                case 3:
                    switch (value_3) {
                        case 1:
                            printf("2-3-1\n");
                            break;
                        case 2:
                            printf("2-3-2\n");
                            break;
                        case 3:
                            printf("2-3-3\n");
                            break;
                    }
                    break;
            }
            break;
        case 3:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("3-1-1\n");
                            break;
                        case 2:
                            printf("3-1-2\n");
                            break;
                        case 3:
                            printf("3-1-3\n");
                            break;
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("3-2-1\n");
                            break;
                        case 2:
                            printf("3-2-2\n");
                            break;
                        case 3:
                            printf("3-2-3\n");
                            break;
                    }
                    break;
                case 3:
                    switch (value_3) {
                        case 1:
                            printf("3-3-1\n");
                            break;
                        case 2:
                            printf("3-3-2\n");
                            break;
                        case 3:
                            printf("3-3-3\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
