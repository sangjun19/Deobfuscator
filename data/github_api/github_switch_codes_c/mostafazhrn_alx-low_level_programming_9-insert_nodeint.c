#include <stdlib.h>
#include "lists.h"
/**
 **insert_nodeint_at_index - this code shall inset new nodes
 *@head: this shall rerpesent head of lst
 *@idx: this shall represent ind of lst
 *@n: this shall rerpeesent int of node
 *Return: it shall return add or void if fail
 */
listint_t *insert_nodeint_at_index(listint_t **head, unsigned int idx, int n)
{
listint_t *ge;
unsigned int x = 0;
listint_t *cp = *head;
ge = malloc(sizeof(listint_t));
switch (ge == NULL)
{
case 1:
return (NULL);
break;
}
ge->n = n;
if (idx == 0)
{
ge->next = cp;
*head = ge;
return (ge);
}
while (x < (idx - 1))
{
if (cp == NULL || cp->next == NULL)
{
return (NULL);
}
cp = cp->next;
x++;
}
ge->next = cp->next;
cp->next = ge;
return (ge);
}
