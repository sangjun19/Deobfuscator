#include <stdio.h>
int main(){
    char ch;
    scanf("%c", &ch);
    switch(ch){
        case 'a': case 'e': case 'i': case 'o': case 'u': case 'O':
        printf("Vowel");
        break;
        case 'b': case 'c': case 'd': case 'f': case 'g': case 'h': case 'j': case 'k': case 'l': case 'm': case 'n': case 'p': case 'q': case 'r': case 's': case 't': case 'v': case 'w': case 'x': case 'y': case 'z':
        printf("Consonant");
        break;
        case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9': case '0':
        printf("Digit");
        break;
        case '%': case '$': case '@': case '&': case '?':
        printf("Special Character");
        break;
        default:
        printf("Invalid input");
    }
        return 0;
    }