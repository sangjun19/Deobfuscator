#include<stdio.h>
# include <stdlib.h>
//# include <conio.h>

struct stack
{
int info;
struct stack *next;
};

struct stack *top;

struct stack* getnode (void)
{
return  (( struct stack* ) malloc (sizeof(struct stack)));
}

void freenode(struct stack *p)

{
free(p);
}

void push (int x)
{
struct stack* newnode;
newnode=getnode();

newnode->info=x;
newnode->next=top;
top=newnode;
}

void pop()
{
if(top==NULL)
{
printf("\n List is EMTPTY");
//getch();
}
else
{
struct stack* temp;

temp =top;

int x=top->info;
printf("%d element deleted successfully",x);
top=top->next;
freenode(temp);
//getch();
}
}

void display()
{
struct stack *temp;
temp=top;
printf("\ntop");
while(temp!=NULL)
{
  printf("-> %d",temp->info);
  temp=temp->next;
}
//getch();
}

void main()
{
int n1,ch;
while(1)
{
// clrscr();
printf("\nStack Operation using Linked List");
printf("\n1. Push an element into the stack.");
printf("\n2. Pop out an element from the stack.");
printf("\n3. Display an element from the stack.");
printf("\n4. Exit.");
printf("\n Enter your Choice:");
scanf("%d",&ch);
switch(ch)
{
 case 1: printf("\n Enter the element to be pused into the stack: ");
	 scanf("%d",&n1);
	 push(n1);
	 break;
 case 2: pop();
	 break;
 case 3: display();
	 break;
 case 4: exit(1);break;
 default: printf("Wrong Choice");break;
}}}