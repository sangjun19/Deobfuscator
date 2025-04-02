#include<stdio.h>
void swap(int*,int*);
void main()
{
int s,x,y,ch;
int *c,*d;
printf("enter two numbers\n");
scanf("%d%d",&x,&y);
printf("1.add\n2.swap\n");
c=&x,d=&y;
printf("enter your choice\n");
scanf("%d",&ch);
switch(ch)
{
case 1 :
s=*c+*d;
printf("sum = %d\n",s);
break;
case 2 :
swap(&x,&y);
break;
default:
printf("invalid choice\n");
break;
}
}
void swap(int*a,int*b)
{
int t;
t=*a;
*a=*b;
*b=t;
printf("after swapping a=%d b=%d\n",*a,*b);
}
