#include<stdio.h>
int main()
{
    int n;
    printf("Enter the day number of a week : ");
    scanf("%d",&n);
    switch(n)
    {
        case 1:
            printf("\nHappy Monday!\nHave a Successful week ahead.\n\n"); break;

        case 2:
            printf("\nHave a Blessed and Wonderful Tuesday!\n\n"); break;

        case 3:
            printf("\nHave a Great Wednesday!\n\n"); break;

        case 4:
            printf("\nWishing you a Fabulous Thursday!\n\n"); break;

        case 5:
            printf("\nWishing you a Pleasant Friday!\n\n"); break;

        case 6:
            printf("\nHave a Nice Saturday!\n\n"); break;

        case 7:
            printf("\nHave a Fantastic and Super Sunday!\n\n"); break;

        default:
            printf("\nEnter a number from 1-7 only.\n\n");
    }
    return 0;
}