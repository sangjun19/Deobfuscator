#include<stdio.h>
#include<conio.h>
void main()
{
char ch;
clrscr();
printf("\n enter the character");
scanf("%c",&ch);
if((ch>='A'&&ch<='Z')||(ch>='a'&& ch<='z'))
{
switch(ch)
{
case 'A':
case 'E':
case 'I':
case 'O':
case 'U':
case 'a':
case 'e':
case 'i':
case 'o':
case 'u':
printf("%c is a vowel",ch);
break;
default:
printf("\n % c is  a cosonant",ch);
}
}
else
{
printf("%c is not an alphabet",ch);
}
getch();
}

