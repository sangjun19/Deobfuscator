#include <stdio.h>

int main ()
{
    int a;
    printf ("Eanter number: ");
    scanf("%d", &a);
    switch(a){
        case 1:
        printf("You entered 1\n");
        break;
        case 2:
        printf("You entered 2\n");
        break;
        case 3:
        printf("YOU entered 3\n");
        break;
        case 4:
        printf("YOU entered 4\n");
        break;
        default:
        printf("NOTHING!!!!\n");
    }

    return 0;
}