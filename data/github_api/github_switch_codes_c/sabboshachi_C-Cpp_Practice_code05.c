#include<stdio.h>
int main(){
    int x = 2, y = 5;

    switch(x){
        
        case 1+1: printf("x is 1");
        break;
        
        case 2-1: printf("x is 2");
        break;
        
        case 10*3-9: printf("x is 3");
        break;

        default: printf("x has a value other than 1 2 3");
        break;

    }

    return 0;
}