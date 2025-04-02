#include <stdio.h>
#define Mon 1
#define Tues 2
#define Wed 3
#define Thurs 4
#define Fri 5
#define Sat 6
#define Sun 7

int main(void){
    int day;
    scanf("%d",&day);
    switch(day){
        case Mon:
            puts("周一");
            break;
        ...
    }
    
    return 0;
}
