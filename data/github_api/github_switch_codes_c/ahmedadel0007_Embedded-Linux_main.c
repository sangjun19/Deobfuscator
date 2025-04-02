// main.c

#include <stdio.h>
#include "Include/addLib.h"
#include "Include/subLib.h"
#include "Include/div.h"
#include "Include/mult.h"
#include "Include/mol.h" 
//#include "Liboperations.a"
int main() {
    int num1, num2;
    char operation;

    // Get user input for num1, num2, and operation
    printf("Enter the first number: ");
    scanf("%d", &num1);

    printf("Enter the second number: ");
    scanf("%d", &num2);

  //  printf("Enter the operation (+, -, *, /, %): ");
    scanf(" %c", &operation);

    // Perform the selected operation
    switch (operation) {
        case '+':
            printf("Result: %d\n", add(num1, num2));
            break;
        case '-':
            printf("Result: %d\n", subtract(num1, num2));
            break;
        case '*':
            printf("Result: %d\n", multiply(num1, num2));
            break;
        case '/':
            printf("Result: %d\n", divide(num1, num2));
            break;
        case '%':
            printf("Result: %d\n", modulus(num1, num2));
            break;
        default:
            printf("Invalid operation\n");
    }

    return 0;
}

