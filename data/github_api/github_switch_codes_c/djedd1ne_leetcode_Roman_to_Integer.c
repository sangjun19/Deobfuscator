/**
 * @file Roman_to_Integer.c
 * @author djmekki (djallal.me)
 * @brief 
 * @version 0.1
 * @date 2023-06-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdio.h>
static int roman_digit(char c)
{
	switch (c)
	{
		case 'I':
			return 1;
		case 'V':
			return 5;
		case 'X':
			return 10;
		case 'L':
			return 50;
		case 'C':
			return 100;
		case 'D':
			return 500;
		case 'M':
			return 1000;
		default:
			return 0;
	}
}

int romanToInt(char *s)
{
	int i;
	int number = roman_digit(s[0]);

	for(i = 1; s[i] != '\0'; i++)
	{
		int previous_number = roman_digit(s[i - 1]);
		int current_number = roman_digit(s[i]);
		if (previous_number < current_number)
			number = number - previous_number + (current_number - previous_number);
		else
			number += current_number;
	}
	return (number);
}
/* Uncomment to test 
int main(void)
{
	char s[100] = "III";

	printf("%d\n", romanToInt(s));
}
*/