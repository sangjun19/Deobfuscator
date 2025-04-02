#include <stdio.h>
#include <stdlib.h>
int main()
{
    int a, b, res;
    char op;
    printf("enter the expression : ");
    scanf("%d %c %d", &a, &op, &b);
    switch (op)
    {
    case '+':
        res = a + b;
        break;
    case '-':
        res = a - b;
        break;
    case '*':
        res = a * b;
        break;
    case '%':
        res = a % b;
        break;
    case '/':
        if (b != 0)
            res = a / b;
        else
        {
            printf("division by zero is not possible\n");
            exit(0);
        }
    }
    printf("%d %c %d = %d\n",a,op,b,res);
    return 0;
}