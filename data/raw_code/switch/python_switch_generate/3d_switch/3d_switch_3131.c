#include <stdio.h>

int main() {
    int value_1 = 4;
    int value_2 = 2;
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
                        case 4:
                            printf("2-1-4\n");
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
                        case 4:
                            printf("2-2-4\n");
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
                        case 4:
                            printf("3-1-4\n");
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
                        case 4:
                            printf("3-2-4\n");
                            break;
                    }
                    break;
            }
            break;
        case 4:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("4-1-1\n");
                            break;
                        case 2:
                            printf("4-1-2\n");
                            break;
                        case 3:
                            printf("4-1-3\n");
                            break;
                        case 4:
                            printf("4-1-4\n");
                            break;
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("4-2-1\n");
                            break;
                        case 2:
                            printf("4-2-2\n");
                            break;
                        case 3:
                            printf("4-2-3\n");
                            break;
                        case 4:
                            printf("4-2-4\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
