#include<stdio.h>

int add(int a, int b) {
  return a + b;
}

int subtract(int a, int b) {
  return a - b;
}

int multiply(int a, int b) {
  return a * b;
}

int divide(int a, int b) {
  if (b == 0) {
    printf("Error! Division by zero.\n");
    return 0;
  } else {
    return a / b;
  }
}

int main() {
  int num1, num2, choice;

  printf("Enter two numbers: ");
  scanf("%d %d", &num1, &num2);

  printf("Enter your choice:\n");
  printf("1. Addition\n");
  printf("2. Subtraction\n");
  printf("3. Multiplication\n");
  printf("4. Division\n");
  scanf("%d", &choice);

  int result;
  switch (choice) {
    case 1:
      result = add(num1, num2);
      break;
    case 2:
      result = subtract(num1, num2);
      break;
    case 3:
      result = multiply(num1, num2);
     break;
    case 4:
      result = divide(num1, num2);
      break;
    default:
      printf("Invalid choice.\n");
      return 1;
  }

  printf("Result: %d\n", result);
  return 0;
} 