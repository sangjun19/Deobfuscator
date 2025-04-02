#include<stdio.h>
int main()
{
	int num1,num2,res;
	char op;
	printf("Enter a number:");
	scanf("%d",&num1);
	printf("Enter the operator:");
	scanf("%*c%c",&op);
	printf("Enter two values:");
	scanf("%d",&num2);
	switch(op)
	{
		case'+' : res = num1+num2;
			 printf("add = %d\n",res);
			 break;
		case '-' : res = num1-num2;
			 printf("sub = %d\n",res);
			 break;
		case '*' :res = num1*num2;
			printf("mul = %d\n",res);
			break;
		case '/' :res = num1/num2;
			printf("div = %d\n",res);
			break;
		default:
			printf("Invalid operator!\n");
	}

	return 0;
}
