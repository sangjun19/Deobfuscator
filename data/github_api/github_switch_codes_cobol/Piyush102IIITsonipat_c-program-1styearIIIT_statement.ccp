// Repository: Piyush102IIITsonipat/c-program-1styearIIIT
// File: statement.ccp

#include <stdio.h>
int main()
{   int a;

    printf("enter your age);
    scanf("%d", &a);

 switch(a){

    case 18:
    printf("your are able to give vote");
    break;

    case 20:
    printf("your are able to give vote and your vote is more valueable ");
    break;

    case 17:
    printf("you are able to give vote if you help the needy people around you);
    break;

    default:
    printf("you are not old enough to give vote );
    break;


 }  

 return 0; 

}