#include <stdio.h>
int main()
{
    int num=10,count=1;

    num--;
    while(num>=0)
    {
    printf("%d %d\n",count,fib(num));
    num--;
    count++;
    }
    return 0;
}

fib(int n)
{
    switch(n)
    {
    case 0 : return 0;
    break;
    case 1 : return 1;
    break;  
    }
    
    return fib(n-1) + fib(n-2);

}