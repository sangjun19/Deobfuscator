#include<stdio.h>
main()
{ int choice;
printf("The List of food items availiable here are: \n1.Pizza- Rs 239\n2.Burger- Rs 129\n3.Pasta- Rs 179\n4.French Fries- Rs 99\n5.Sandwich- Rs 149");
printf("\nselect your choice food item by number corresponding to their name(1-5)");
scanf("%d",&choice);
switch(choice)
{
    case 1:printf("Wow!Its PIZZA\nPay Rs:239");break;
    case 2:printf("Wow!Its BURGER worth Rs:129");break;
    case 3:printf("Wow!ItsPASTA\nIt Costs  Rs:179");break;
    case 4:printf("Wow!ItsFRENCH FRIES\nPay Rs:99");break;
    case 5:printf("Wow!Its SANDWITCH\nPay Rs:149");break;
    default:printf("You have made an invalid choice");
}
}
