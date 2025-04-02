#include <stdio.h>

int main()
{
	int itm, qty;
	double up, cost=0;

	printf("Enter item no. : ");
	scanf("%d", &itm);

	while (itm!=-1)
	{
		if (itm==1 || itm==2 || itm==3)
		{
			printf("Enter quantity : ");
			scanf("%d", &qty);

			switch (itm)
			{
				case 1 : up=300.25;
						 break;
				case 2 : up=145.50;
						 break;
				case 3 : up=525.00;
						 break;
				default: break;
			}

			cost = cost+(up*qty);
			
			printf("\nEnter item no. : ");
			scanf("%d", &itm);
		}
	}

	printf("\nTotal Price to Pay : %.2lf", cost);

	return 0;
}