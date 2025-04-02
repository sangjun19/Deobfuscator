#include<stdio.h>
int main()
{
    int n;
    printf("Enter a No. : ");
    scanf("%d",&n);

    switch(n%2){
        case 0: puts("Even"); break;
        default:  puts("Odd");
    }


}