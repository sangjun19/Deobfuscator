#include <stdio.h>

int main() {
    int num;
    printf("Enter a number: ");
    scanf("%d", &num);

    switch (num % 2) {
        case 0:
            printf("The number is even.");
            break;

        case 1:
            printf("The number is odd.");
            break;
    }

    printf("\n");

    return 0;
}
