/* 
 * File:    switch.c 
 * Author:  Student Name  0302433n@acadiau.ca 
 * Date:    2024/01/31 
 * Version: 1.0 
 * Purpose: 
 *          This program demonstrates a switch statement
 */


#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

int main(void){
	printf("input a character:\n");
	char i;
	i = getchar();
	i = toupper(i);
	switch(i) {
		case 'A':
			printf("you inputted an \"a\" or an \"A\"\n");
			break;
		case 'B':
			printf("you inputted a \"b\" or a \"B\"\n");
			break;
		default:
			printf("you didn't input an \"a\" or a \"b\"\n");
	}
	return EXIT_SUCCESS;
}
