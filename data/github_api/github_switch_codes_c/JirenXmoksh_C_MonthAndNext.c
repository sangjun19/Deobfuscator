#include<stdio.h>
int main(){
    int num;
    printf("Enter a number: ");
    scanf("%d", &num);

    switch(num){
        case 1:
        printf("January\nFebruary");
        break;

         case 2:
        printf("February\nMarch");
        break;

         case 3:
        printf("March\nApril");
        break;

         case 4:
        printf("April\nMay");
        break;

         case 5:
        printf("May\nJune");
        break;

         case 6:
        printf("June\nJuly");
        break;

         case 7:
        printf("July\nAugust");
        break;

         case 8:
        printf("August\nSeptember");
        break;

         case 9:
        printf("September\nOctober");
        break;

         case 10:
        printf("October\nNovember");
        break;

         case 11:
        printf("November\nDecember");
        break;

         case 12:
        printf("December\nJanuary");
        break;
    }
    return 0;
}