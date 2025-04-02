#include<stdio.h>
 void main()
{
	int x,y;
	printf("Enter the numbers");
	scanf("%d%d",&x,&y);
	
	switch(x<y)
 {
		case 0:
		  if(x != y)
		  printf("%x is greater",x);break;
		case 1:
		  if(x != y)
				printf("%y is greater",y);break;
		
			 
 }
}
