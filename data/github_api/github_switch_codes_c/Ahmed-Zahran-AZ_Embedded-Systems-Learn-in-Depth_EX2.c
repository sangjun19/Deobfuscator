#include <stdio.h>

int main(void)
{
  char c;
  printf("Enter a letter: ");
  scanf("%c",&c);
  switch (c){
    case 'i':
    case 'I':
    case 'e':
    case 'E':
    case 'o':
    case 'O':
    case 'a':
    case 'A':
    case 'u':
    case 'U':
    printf("%c is a vowel",c);
    break;
    default:
    printf("%c is a consonant",c);
  }
}

