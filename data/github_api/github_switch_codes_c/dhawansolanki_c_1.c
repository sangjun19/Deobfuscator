//Simple Calculator
#include <stdio.h>
void main(){
    float a,b,c,d;
    char op;
    printf("Enter the Operation : ");
    scanf("%f%c%f",&a,&op,&b);
    switch(op){
        case '+':
            c = a+b;
            printf("The Sum of %f & %f is %f",a,b,c);
            break;
        case '-':
            c = a-b;
            printf("The Difference of %f & %f is %f",a,b,c);
            break;
        case '*':
            c = a*b;
            printf("The Product of %f & %f is %f",a,b,c);
            break;
        case '/':
            if(b==0){
                printf("Invalid Division.");
            }else{
                c = a/b;
                printf("The Quotient of %f & %f is %f",a,b,c);
            }
            break;
        default:
            printf("Something Went Wrong...");
    }
}