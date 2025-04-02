#include <stdio.h>

int main() {
    int num1,num2 ;
    float result ;
    char oper ;
    printf("enter 1st num: ") ;
    scanf("%d",&num1) ;
    printf("enter 2nd num: ") ;
    scanf("%d",&num2) ;
    printf("enter operator(+,-,*,/): ") ;
    scanf(" %c",&oper) ;
    switch(oper) {
        case '+' :
           result = num1 + num2 ;
           break ;
        case '-' :
           result = num1 - num2 ;
           break ;
        case '*' :
           result = num1 * num2 ;
           break ;
        case '/' :
           result = num1 / num2 ;
           break ;
        default :
           printf("Wrong Operator")
    }
     printf("result = %.2f",result) ;
     return 0;
}