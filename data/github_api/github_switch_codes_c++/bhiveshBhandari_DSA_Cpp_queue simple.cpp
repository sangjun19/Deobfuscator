// Repository: bhiveshBhandari/DSA_Cpp
// File: queue simple.cpp

//write program to implement leniar queue operation insertion,deletion and display
#define size 5 
#include <stdio.h>
int insert();
int rear,front,item,q[size],i;
int main()
{
	int q[5],item,ch;
	printf("enter 1.for insertion \nenter 2.for deletion \nenter 3.for display \n");
	printf("enter your choise \n");
	scanf("%d",&ch);
	do
	{
		switch (ch)
		{
			case 1:
			insert();
			break;
			default:
			printf("wrong choice entered");
		}
	}
}
	int insert()
	{
		if(rear==size)
		{
			printf("queue is FULL \n");
		}
		else
		{
			if(front=-1)
			{
				printf("\n enter the element to be inserted: \n");
				scanf("%d",&item);
				rear=rear+1;
				q[rear]=item;
			}
		}
	}
