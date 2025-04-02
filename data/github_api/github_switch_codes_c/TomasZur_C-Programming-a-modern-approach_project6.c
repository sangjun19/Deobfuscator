#include <stdio.h>
#include <ctype.h>

#define SIZE ((int) (sizeof(a)/sizeof(a[0])))

int main(){

    char ch[99];
    int i=0;
    char character;

    printf("Enter a message: ");

     
    while ((character=getchar()) != '\n'){
        ch[i]=character;
        i++;
    }


    printf("Message: ");
    for(int j=0;j<i;j++){
        ch[j]=toupper(ch[j]);
        switch (ch[j])
        {
        case 'A':
            ch[j]='4';
            break;
        case 'B':
            ch[j]='8';
            break;
        case 'E':
            ch[j]='3';
            break;
        case 'I':
            ch[j]='1';
            break;
        case 'O':
            ch[j]='0';
            break;
        case 'S':
            ch[j]='5';
            break;
        }
        printf("%c",ch[j]);
    }
        printf("!!!!!!!!!!");
    return 0;



}