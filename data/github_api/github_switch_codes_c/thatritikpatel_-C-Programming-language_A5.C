#include <stdio.h>
#include <conio.h>

void main(){
	int x;

	clrscr();

	printf("Enter a number: ");
	scanf("%d",&x);

	switch(x){
		case 1:
			printf("~~~~~1");
			break;
		case 2:
			printf("~~~~~2 #");
			break;
		case 2:
			printf("~~~~~2 $");
		default:
			printf("####");
	}

	getch();
}