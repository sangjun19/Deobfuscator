#include<stdio.h>
int main(){
int a,b,c;
char ch;
printf("Enter two numbers ");
scanf("%d %d", &a, &b);
printf("Enter the operator ");
scanf(" %c", &ch);

switch(ch){
	case '+':
		printf("Result: %d\n",a+b);
		break;
	case '-':
		printf("Result: %d\n",a-b);
		break;
	case '*':
		printf("Result: %d\n",a*b);
		break;
	case '/':
		printf("Result: %d\n",a/b);
		break;
	case '%':
		printf("Result: %d\n",a%b);
		break;
}

return 0;
}
