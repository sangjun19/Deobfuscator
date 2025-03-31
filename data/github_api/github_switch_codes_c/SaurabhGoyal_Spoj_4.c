#include<stdio.h>
#include<string.h>
#include<stdlib.h>
struct stack{
    char* data;
    int size;
    int top;
};
struct stack* createStack(int capacity){
	struct stack *s=(struct stack *)malloc(sizeof(struct stack));
	if(!s)
		return NULL;
	s->top=-1;
	s->size=capacity;
	s->data=(char *)malloc(s->size*sizeof(char));
	if(!s->data)
		return NULL;
	return s;
}
void push(struct stack *s,char c){
	s->data[++s->top]=c;
	//printf("character %c pushed at position %d\n",s->data[s->top],s->top);
}
char pop(struct stack *s){
	if(s->top!=-1)
		return s->data[s->top--];
	printf("\nStack empty\n");
	return '$';
}

int isEmpty(struct stack *s){
	return s->top==-1;
}
int isOperand(char ch){
	return (ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z');
}
int precedence(char op){
	switch(op){
		case '+':
		case '-':
			return 1;
		case '*':
		case '/':
			return 2;
		case '^':
			return 3;
		default:
			return -1;	
	}
}

char * inToPost(char *expr){
	char *rpn;	
	rpn=(char *)malloc(strlen(expr)*sizeof(char));
    struct stack *s=createStack(strlen(expr));
    int i=0,k=0;
    for(;expr[i];i++){
		if(isOperand(expr[i])){
    		rpn[k++]=expr[i];
		}
   		else if(expr[i]=='('){
		   	push(s,expr[i]);
		}
		else if(expr[i]==')'){
   			while(!isEmpty(s)&&s->data[s->top]!='('){
			   	rpn[k++]=pop(s);
  			}
  			if(!isEmpty(s)&&s->data[s->top]!='(')
  				return -1;
			else
				pop(s);
   		}
   		else{
   			while(!isEmpty(s)&&precedence(expr[i])<=precedence(s->data[s->top])){
			   	rpn[k++]=pop(s);
			}
			push(s,expr[i]);   	
		}
    }
    while(!isEmpty(s)){
    	rpn[k++]=pop(s);
    }
    rpn[k++]='\0';
	return  rpn;
}

int main(){
	int t=0;
	scanf("%d",&t);
	while(t--){
		char *expr;
		expr=(char *)malloc(401*sizeof(char));
		//printf("enter expr\n");
		scanf("%s",expr);
		printf("%s\n",inToPost(expr));
	}
    return 0;
}
