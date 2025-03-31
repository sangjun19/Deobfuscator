// C Program to Make a Simple Calculator using switch-case
// Statements Statements
#include <stdio.h>

// Function to implement simple calculator using switch
// statement
double simpleCalc(double num1, double num2, char op)
{
    double result;

    // Perform the corresponding operation using
    // switch-case
    switch (op) {
    case '+':

        // Addition
        result = num1 + num2;
        break;
    case '-':

        // Subtraction
        result = num1 - num2;
        break;
    case '*':

        // Multiplication
        result = num1 * num2;
        break;
    case '/':

        // Division
        // Check for division by zero
        if (num2 != 0) {
            result = num1 / num2;
        }
        else {
            printf("Error! Division by zero.\n");
            return -1;
        }
        break;
    default:
        printf("Error! Operator is not correct.\n");
        return -1;
    }
}

int main()
{
    char op;
    double num1, num2, result;

    // Read the operator
    printf("Enter an operator (+, -, *, /): ");
    scanf("%c", &op);

    // Read the two numbers
    printf("Enter two operands: ");
    scanf("%lf %lf", &num1, &num2);

    result = simpleCalc(num1, num2, op);

    // Print the result
    printf("Result: %.2lf\n", result);

    return 0;
}
