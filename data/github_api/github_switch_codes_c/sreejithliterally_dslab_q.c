#include <stdio.h>
#include<stdlib.h> 
#define max 5

void insert();
void delete();
void display();

int Queue_array[max];
int rear=-1;
int front=-1;
int main()
{
int choice;
while(1)
{
printf("1.Insert element to a Queue\n");
printf("2.Delete elements from a queue\n");
printf("3.Display all elements \n");
printf("4.Quit\n");
printf("5.Enter your choice");
scanf("%d",&choice);
switch(choice)
{
	case 1:
		insert();
		break;
	case 2:
		delete();
		break;
	case 3:
		display();
		break;
	case 4:
		exit(0);
		break;
	default:
		printf("Wrong choice\n");
		}
	}
}

void insert()
{

int add_item;
if(rear==max-1)
	printf("Queue overflow\n");
else {
	if(front==-1)
	front = 0;
	printf("Insert the element");
	scanf("%d",&add_item);
	rear=rear+1;
	Queue_array[rear] = add_item;
	}
	printf("\n\n\n");
} 
/* End of insertion*/

void delete()
{

if(front==-1 || front>rear)
{
	printf("Queue underflow\n");
	return;
}

else
{
	printf("Element deleted from the Queue is %d\n",Queue_array[front]);
	front=front+1;
}

printf("\n\n\n");
}
/* End of deletion*/

void display(){
int i;
if(front==-1)
	printf("Queue is empty\n");
else{
	printf("queue is : \n");
	for(i=front;i<=rear;i++);
		printf("%d",Queue_array[i]);
		printf("\n");
}
	printf("\n\n\n");
}



