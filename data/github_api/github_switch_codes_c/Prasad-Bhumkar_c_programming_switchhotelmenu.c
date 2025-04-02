#include<stdio.h>
int main()
{
    int foodtype;
    printf("1.Punjabi food\n2.Maharashtrian food\n3.Chinese food\n4.Sea food\n");
    printf("Enter your choice:\n");
    scanf("%d",&foodtype);
    
    switch (foodtype)
    {
    case 1:
        printf("You have selected Punjabi food.");
        break;
    case 2:
        printf("You have selected Maharashtrian food.");
        break;
    case 3:
        printf("You have selected Chinese food.");
        break;
    case 4:
        printf("You have selected Sea food.");
    default:
    printf("Do you want to try some snacks!!!");
        break;
    }
    return 0;
}