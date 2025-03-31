#include <stdio.h>
int main()

{
	int a,b,d;
	
	printf("enter a and b\n");
	scanf("%d %d",&a,&b);
	
	float c;
	c=(a+b)/2;
	d=a%b;
	int x;
	      printf("enter the option:\n");
	      scanf("%d",&x);
    switch(x)
    {
    	case 1:
    		   printf("%d %d" ,a,b);
    	break;
    	
    	case 2:
    		   if(a%2 && b%2 == 1){ 
    		                       printf("%f",c);
			   }
			   else if(a%2||b%2){
			   	                 printf("%f",c+0.5);
			   }
			break;
			case 3:
				   printf("%d" ,d);
			break;
			case 4:
			break;
			
			default:
			printf("enter values between 1 and 4 only");	
	}
	return 0;
}
