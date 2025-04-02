#include <stdio.h>
#include <stdlib.h>

static NODE *head_ptr = NULL;

/**
 * main_interface()
 */


void main_interface(int ch)
{
	switch (ch){
		case 'a':
			list_node_add();
			break;
		case 'd':
			if (!list_node_delete())
				list_node_print();
			break;
		case 'p':
			list_node_print();
			break;
		default:
			break;
	}
}


NODE *list_node_create(void)
{
	NODE *new_ptr, *ptr;

	new_ptr = list_node_create();
	printf("Enter the student name and ID: ");
	scanf("%s%ld", new_ptr->name, &new_ptr->id);

	if(head_ptr == NULL)
	{
		head_ptr = new_ptr;
	}else{
		for (ptr=head_ptr; ptr->next_ptr != NULL; ptr=ptr->next_ptr)
			ptr->next_ptr = new_ptr;
	}
}

int list_node_delete(void)
{
	NODE *ptr, *ptr_saved;
	unsigned long id;
	int deleted = 0;
	int reval = 0;

	if (head_ptr == NULL)
	{
		printf("Sorry, nothing to delete.\n");
		reval = 1;
	}else {
		printf("Enter the student ID: "


