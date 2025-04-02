#include<stdio.h>
main()
{
	int ch;
	printf("Enter a character:");
	scanf("%c", &ch);
	switch(ch)
	{
		case 'a':
		printf("Character is Vowel");
		break;
		case 'e':
		printf("Character is Vowel");
		break;
		case 'i':
		printf("Character is Vowel");
		break;
		case 'o':
		printf("Character is Vowel");
		break;
		case 'u':
		printf("Character is Vowel");
		break;
		default:
		printf("Character is Consonant");
		break;
	}
}
