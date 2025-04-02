#include<stdio.h>
int main()
{
int a;
printf("Enter a number\n");
scanf("%d",&a);
switch(a%2==0)
{
case 1:printf("Given number is even\n");break;
case 0:printf("Given number is odd\n");
}
}
