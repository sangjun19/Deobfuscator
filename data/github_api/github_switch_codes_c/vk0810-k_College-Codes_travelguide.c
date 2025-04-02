#include<stdio.h>
#include<conio.h>
void main()
{
    char ch;
    printf("enter ch");
    scanf("%c",&ch);
    switch(ch)
    {
        case 'a':
        printf("air travel");
        break;

        case 'b':
        printf("bus timing");
        break;

        case 't':
        printf("train timing");
        break;

        default:
            prinf("skip");
            break;
    }
    getch();
}
