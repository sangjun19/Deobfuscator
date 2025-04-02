#include<stdio.h>
#include<stdlib.h>
int main()
{
	int a,m,n,opt;
	printf("Enter the value\n");
	scanf("%d",&a);
	printf("val=%d and Hex=%02x\n",a,a);
	printf("Enter the range of bits m and n\n");
	scanf("%d%d",&m,&n);
	printf("1.Set\t2.Clear\t3.Toggle\t4.return\t5.exit\n");
	scanf("%d",&opt);
	switch(opt)
	{
		case 1:
			a = a|((0XFFFFFFFF>>(sizeof(int)*8)-(m-n+1))<<n);
			break;
		case 2:
			a = a&(~((0XFFFFFFFF>>(sizeof(int)*8)-(m-n+1))<<n));
			break;
		case 3:
			a = a^((0XFFFFFFFF>>(sizeof(int)*8)-(m-n+1))<<n);
			break;
		case 4:
			a = a&((0XFFFFFFFF>>(sizeof(int)*8)-(m-n+1))<<n);
			break;
		case 5: 
			exit(1);
			break;
		default: 
			break;
	}
	printf("val =%d and Hex=%02x\n",a,a);
	return 0;
} 
