#include <stdio.h>
int main(void){
    char c;
    scanf("%c", &c);
    switch (c) {
        case 'd':
            printf("degree\n");
            break;
        case 'r':
            printf("radian\n");
            break;
       default:
            printf("Illegal\n");
            break;
    }
    return 0;
}
