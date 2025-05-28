#include <stdio.h>

int main() {
    int value_1 = 1;
    int value_2 = 6;
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
                case 3:
                    switch (value_3) {
                        case 1:
                            printf("1-3-1\n");
                            break;
                    }
                    break;
                case 4:
                    switch (value_3) {
                        case 1:
                            printf("1-4-1\n");
                            break;
                    }
                    break;
                case 5:
                    switch (value_3) {
                        case 1:
                            printf("1-5-1\n");
                            break;
                    }
                    break;
                case 6:
                    switch (value_3) {
                        case 1:
                            printf("1-6-1\n");
                            break;
                    }
                    break;
            }
            break;
    }
    return 0;
}
