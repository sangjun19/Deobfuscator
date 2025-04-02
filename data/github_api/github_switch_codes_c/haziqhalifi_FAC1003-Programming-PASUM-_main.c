#include<stdio.h>
double sin(double,double);
double cos(double,double);
double fact(double);

int main()
{
    int f;
    double val,x,n;

    start:
    printf("1: sine function\t2:cosine function\nChoose your function: ");
    scanf("%d",&f);
    printf("Enter a positive integer for n : ");
    scanf("%lf",&n);
    printf("Enter a floating-point value for x in radian: ");
    scanf("%lf",&x);

    switch(f){
    case 1: val=sin(x,n);
            break;
    case 2: val=cos(x,n);
            break;
    default: printf("Function error");
             goto start;
    }

    printf("\nThe answer is %.4lf",val);
}
double cos(double x,double n)
{

    if (n==0)
        return pow(-1,n)*pow(x,2*n)/fact(2*n);

    else
        return pow(-1,n)*pow(x,2*n)/fact(2*n)+ cos(x,n-1);
}
double sin(double x,double n)
{

    if (n==0)
        return pow(-1,n)*pow(x,2*n+1)/fact(2*n+1);

    else
        return pow(-1,n)*pow(x,2*n+1)/fact(2*n+1)+ sin(x,n-1);
}
double fact(double k)
{
    if (k==0)
        return 1;
    else
        return k*fact(k-1);
}
