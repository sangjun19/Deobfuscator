#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int main(void)
{
	int month, days;

	printf("input Month: ");
	scanf("%d", &month);

	switch (month)
	{
		case 2:
			days = 28; break;
		case 4:
		case 6:
		case 9:
		case 11:
			days = 30; break;
		default:
			days = 31;
	}

	printf("days of %d month => %d\n", month, days);
	return 0;
}