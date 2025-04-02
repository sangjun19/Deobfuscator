// Write a program in C to count the total number of vowels or consonants in a string.

#include <stdio.h>
#include <ctype.h> // For tolower() function

int main() {
    char str[100];
    int vowels = 0, consonants = 0;
    int i = 0;
    char ch;

    printf("Enter a string: ");
    scanf("%[^\n]s",str);

    while (str[i] != '\0') {
        ch = tolower(str[i]); 
        if (ch >= 'a' && ch <= 'z') { 
            switch (ch) {
                case 'a':
                case 'e':
                case 'i':
                case 'o':
                case 'u':
                    vowels++;
                    break;
                default:
                    consonants++;
                    break;
            }
        }
        i++;
    }
    printf("Total vowels: %d\n", vowels);
    printf("Total consonants: %d\n", consonants);
}
