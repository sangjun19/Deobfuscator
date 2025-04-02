#include<stdio.h>
int main(){
    int month;
    printf("Enter the month number: ");
    scanf("%d", &month);
    switch(month){
        case 1:
            printf("January\n");
        case 2:
            printf("February\n");
            break;
        case 3:
            printf("March\n");
        case 4:
            printf("April\n");
            break;
        case 5:
            printf("May\n");
        case 6:
            printf("June\n");
            break;
        case 7:
            printf("July\n");
        case 8:
            printf("August\n");
            break;
        case 9:
            printf("September\n");
        case 10:
            printf("October\n");
            break;
        case 11:
            printf("November\n");
        case 12:
            printf("December\n");
            break;
        default:
            printf("Invalid month number\n");
    }
    return 0;
}