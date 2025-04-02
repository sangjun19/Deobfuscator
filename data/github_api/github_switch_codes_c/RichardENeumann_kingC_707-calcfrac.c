// Calculate two fractions with any given operator
#include <stdio.h>

int main(void) {
  int num1, denom1, num2, denom2, resultNum = 0, resultDenom = 0;
  char operator;

  printf("Enter the two fractions, separated by the operator [w/x(+-*/)y/z]: ");
  scanf("%d/%d%1c%d/%d", &num1, &denom1, &operator, &num2, &denom2);

  switch (operator) {
    case '+':
      resultNum = num1 * denom2 + num2 * denom1;
      resultDenom = denom1 * denom2;
      break;
    case '-':
      resultNum = num1 * denom2 - num2 * denom1;
      resultDenom = denom1 * denom2;
      break;
    case '/':
      resultNum = num1 * denom2;
      resultDenom = denom1 * num2;
      break;
    case '*':
      resultNum = num1 * num2;
      resultDenom = denom1 * denom2;
      break;
  }

  printf("The result is %d/%d\n", resultNum, resultDenom);

  return 0;
}