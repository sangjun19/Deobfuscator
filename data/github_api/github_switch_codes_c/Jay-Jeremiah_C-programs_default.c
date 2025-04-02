// default - specifies some code to run if there is no code to run

#include <stdio.h>


int main(){
    int day = 4;

    switch (day) {
        case 6:
        printf("Today is Saturday");
        break;
        case 7:
        printf("Today is Sunday");
        break;
        default:
        printf("Looking forward for the weekend");
    }
    return 0;
}