#include<stdio.h>
#include<string.h>
#include<stdlib.h>
void main()
{
    int ch,i;
    char s1[50],s2[50],*p;
    char s;
    printf("\n----------------MENU-------------\n");
    printf("1:Length Of the string\n");
    printf("2:Reverse a string\n");
    printf("3:Lowercase the string\n");
    printf("4:Uppercase the string\n");
    printf("5:Sompare two strings\n");
    printf("6:Find the first occurance\n");
    printf("7:Concatinate two strings\n");
    printf("8:Exit \n");

    printf("Enter Your Choice:\t");
    scanf("%d",&ch);
    switch (ch)
    {
    case 1: printf("Enter the string:\t");
            scanf("%s",s1);
            i = strlen(s1);
            printf("length = %d",i);   
        break;
    case 2: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Entered string in reverse\n%s",strrev(s1));
        break;
    case 3: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Entered string in lowercase\n%s",strlwr(s1));
        break;
    case 4: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Entered string in uppercase\n%s",strupr(s1));
        break;
    case 5: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Enter the string:\t");
            scanf("%s",s2);
            i = strcmp(s1,s2);
            if (i == 0)
                printf("BOTH STRINGS ARE EQUAL\n");
            
            else if (i>0)
                printf("STRING 2 IS GREATER\n");
            
            else
                printf("STRING 1 IS SMALLER\n");
        break;
    case 6: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Enter the character:\t");
            scanf("%c",s);
            p = strchr(s1,s);
            printf("string = %s",p);
        break;
    case 7: printf("Enter the string:\t");
            scanf("%s",s1);
            printf("Enter the string:\t");
            scanf("%s",s2);
            printf("CONCATINATED STRING\t\t %s",strcat(s1,s2));
    case 8: exit(0);
        break;
    default: printf("INVALID CHOICE");
        break;
    }
}