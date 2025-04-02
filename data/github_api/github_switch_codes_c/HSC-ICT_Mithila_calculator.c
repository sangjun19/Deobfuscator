#include<stdio.h>
#include<math.h>
int main()
{
    char operator;
    double n1, n2, n3;
    double logx;

    printf("Enter an operator (+, -, *, /): ");
    scanf("%c", &operator);

    printf("Enter two operands: ");
    scanf("%lf %lf",&n1, &n2);

    scanf("%lf", &n3);

    switch(operator){
        case '+':
            printf("%.2lf + %.2lf = %.2lf\n", n1, n2, n1 + n2);
            break;
        case '-':
            printf("%.2lf - %.2lf = %.2lf\n", n1, n2, n1 - n2);
            break;
        case '*':
            printf("%.2lf * %.2lf = %.2lf\n", n1, n2, n1 * n2);
            break;
        case '/':
            printf("%.2lf / %.2lf = %.2lf\n", n1, n2, n1 / n2);
            break; 
    }
    return 0;
}
