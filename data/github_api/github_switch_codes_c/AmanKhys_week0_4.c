#include<stdio.h>
#include<string.h>

void findStrLen() {
	char str[100];
	printf("enter the string:");
	scanf("%s",str);
	int i = 0;
	while(str[i]!='\0'){
		i++;
	}
	printf("length of the string: %i \n\n", i);
}

void concatStr() {
	char str1[200], str2[100];
	printf("enter the two strings \n");
	printf("- first string:");
	scanf("%s", str1);
	printf("- second string:");
	scanf("%s", str2);


// 	while(*str1 != '\0'){
// 		str1++;
// 	}
// 	while(*str2 != '\0'){
// 		*str1 = *str2;
// 		str1++;
// 		str2++;
// 	}
// 	*str1 = '\0';

	int i = 0;
	int	j = 0;
	while(str1[i] != '\0') {
		i++;
	}
	while(str2[j] != '\0') {
		str1[i] = str2[j];
		i++;
		j++;
	}
	str1[i] = '\0';

	printf("the concated string: %s \n\n", str1);
}

void reverseStr() {
	char str[100];
	printf("enter the string: ");
	scanf("%s",str);
	char reverse[100];

	//finding the size of the array
	int len = 0;
	while(str[len]!='\0'){
		len++;
	}
	printf("length of the string: %i \n\n", len);

	for(int i = len - 1; i>= 0; i--){
		reverse[len - i - 1]= str[i];
	}
	reverse[len] = '\0';
	printf("reverse string: %s \n\n",reverse);
}

int main() {
	//repl
	while(1) {
		printf("enter the subsequent numbers for respective commands: \n\n");
		printf("1 - string length\n");
		printf("2 - string concat\n");
		printf("3 - string reverse\n");
	
		char command[100];
		printf("enter the command: ");
		scanf("%s", command);
		if (strcmp(command,"exit") == 0){
			printf("exiting the program.");
			return 1;
		} 
		switch(command[0]){
			case '1': findStrLen(); break;
			case '2': concatStr(); break;
			case '3': reverseStr(); break;
			default: printf("enter a valid command!! \n");
		}
	}

	return 0;
}
