/* C Primer Plus Ch.7 - Problem 1
 * @reproduced by yoonBot
 */

#include <stdio.h>

int main(){

    char ch;
    int space_cnt, other_cnt, newline_cnt;

    space_cnt = other_cnt = newline_cnt = 0;

    printf("Enter input (Enter # to quit):\n");
    while((ch = getchar()) != '#'){
        switch (ch){
            case '#': break;
            case ' ': space_cnt++;
                      break;
            case '\n': newline_cnt++;
                       break;
            default :
                       other_cnt++;
                       break;
        }
    }

    printf("Number of space characters entered: %d\n", space_cnt);
    printf("Number of newline characters entered: %d\n", newline_cnt);
    printf("Number of other characteres entered: %d\n", other_cnt);

    return 0;
}


