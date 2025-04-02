#include "stdio.h"
#include <conio.h>
void number (int);
int main()
{
int num;
//clrscr();
printf ("Enter Number (0-3) ........ ");
scanf ("%d",&num);
number(num);
getch();
}

void number (int x)
{
switch (x)
	{
	case 1:
	printf ("One");
	break;
	case 2:
	printf ("Two");
	break;
	case 3:
	printf ("Three");
	break;
	default:
	printf ("Number Out of Range.");
	break;
	}
}


