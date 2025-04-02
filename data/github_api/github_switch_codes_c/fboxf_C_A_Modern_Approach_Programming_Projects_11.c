#include <stdio.h>
#include <ctype.h>

int main (void)
{
	char ch, number[20] = { 0 };

	printf("Enter phone number: ");

	for (int i = 0; (ch = getchar()) != '\n' && i < 20; i++) {

		if (isalpha(ch))
		{

			switch (toupper(ch)) 
			{
				case 'A': case 'B': case 'C':
					number[i]  = '2';
					break;
				case 'D': case 'E': case 'F':
					number[i] = '3';
					break;
				case 'G': case 'H': case 'I':
					number[i] = '4';
					break;
				case 'J': case 'K': case 'L':
					number[i] = '5';
					break;
				case 'M': case 'N': case 'O':
					number[i] = '6';
					break;
				case 'P': case 'R': case 'S':
					number[i] = '7';
					break;
				case 'T': case 'U': case 'V':
					number[i] = '8';
					break;
				case 'W': case 'X': case 'Y':
					number[i] = '9';
					break;
			}

		}
	}

	for (int i = 0; i < 20; i++){
		printf("%c", number[i]);
	}
	printf("\n");
	return 0;
}
