/*
=============================================
 Name: L5T5T30016056.c
 Author: Bohan YANG
 Version: 1.0
 Copyright: Your copyright notice
 Description: Implement a program that can read and transfer a letter grade to points. Output the points to two decimal places. A waring is given for an invalid input.
 ============================================= */
#include <stdio.h>
int main() {
    
    printf("Enter a letter grade: ");
    char ch;
    scanf("%c", &ch);
    
    switch (ch) {
    case 'A':
    case 'a':
        puts("4.00");
        break;
    case 'B':
    case 'b':
        puts("3.00");
        break;
    case 'C':
    case 'c':
        puts("2.00");
        break;
    case 'D':
    case 'd':
        puts("1.00");
        break;
    case 'F':
    case 'f':
        puts("0.00");
        break;
        
    default:
        puts("warning!");
        break;
    }
    
    return 0;
}