// Repository: Sidipurea-ym/Hello-word
// File: 4.c

#include<stdio.h>
#include<stdlib.h>
int main()
{
    int day,month,year;
    scanf("%d",&year);
    scanf("%d",&month);
    scanf("%d",&day);

if(year%4==0&&year%400!=0)
{
    switch(month)
{
    case 1:printf("%d",day);
    break;
case 2:printf("%d",31+day);
break;
case 3:printf("%d",60+day);
break;
case 4:printf("%d",91+day);
break;
case 5:printf("%d",121+day);
break;
case 6:printf("%d",152+day);
break;
case 7:printf("%d",182+day);
break;
case 8:printf("%d",213+day);
break;
case 9:printf("%d",244+day);
break;
case 10:printf("%d",274+day);
break;
case 11:printf("%d",305+day);
break;
case 12:printf("%d",335+day);
}
}
else
{
     switch(month)
{
case 1:printf("%d",day);
break;
case 2:printf("%d",31+day);
break;
case 3:printf("%d",59+day);
break;
case 4:printf("%d",90+day);
break;
case 5:printf("%d",120+day);
break;
case 6:printf("%d",151+day);
break;
case 7:printf("%d",181+day);
break;
case 8:printf("%d",212+day);
break;
case 9:printf("%d",243+day);
break;
case 10:printf("%d",273+day);
break;
case 11:printf("%d",304+day);
break;
case 12:printf("%d",334+day);
break;
}}

return 0;
}