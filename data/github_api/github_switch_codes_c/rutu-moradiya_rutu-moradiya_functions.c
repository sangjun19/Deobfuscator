#include<stdbool.h>

int getINT()
{
    int a,b;
    
    printf("enter a:");
    scanf("%d",&a); 
    printf("enter b:");
    scanf("%d",&b);

}   
void cal(); 
{
    float cal;
    
    printf("1)for +\n");
    printf("2)for -\n");
    printf("3)for *\n");
    printf("4)for /\n");
    printf("5)for %\n");
    printf("0)for exit \n");
    scanf("%f",&cal);

    switch(cal)
    {
        
        case '1':
            printf("+\n");
        break;
        case '2':
            printf("-\n");
        break;
        case '3':
            printf("*\n");
        break;
        case '4': 
            printf("/\n");
        break;
        case '5':
            printf("%\n");
        break;
        case '0':
            printf("exit\n");
        break;
        default:
            printf("invaid");
    }

}
 