#include<stdio.h>
int main()
{
  int num1,num2,choice,res;
  printf("Enter the number1: ");
  scanf("%d",&num1);
  printf("Enter the number2: ");
  scanf("%d",&num2);

  printf("1.addition\n2.subtract\n3.multiply\n4.division\nEnter the choice: ");
  scanf("%d",&choice);

  switch(choice)
  {
    case 1:printf("Adition: %d\n",num1+num2);
	       break;
    case 2:printf("Subtract: %d\n",num1-num2);
	       break;
    case 3:printf("Adition: %d\n",num1*num2);
	       break;
    case 4:printf("Adition: %d\n",num1/num2);
	       break;
    default:printf("Enter valid choice\n");
	       break;
  }

  return 0;
}
