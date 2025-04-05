#include <stdio.h>

int main() {
    int value_1 = 5;
    int value_2 = 2;
    int value_3 = 1;
    switch (value_1) {
        case 1:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("1-1-1\n");
                            break;
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("1-2-1\n");
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
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("2-2-1\n");
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
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("3-2-1\n");
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
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("4-2-1\n");
                            break;
                    }
                    break;
            }
            break;
        case 5:
            switch (value_2) {
                case 1:
                    switch (value_3) {
                        case 1:
                            printf("5-1-1\n");
                            break;
                    }
                    break;
                case 2:
                    switch (value_3) {
                        case 1:
                            printf("5-2-1\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
