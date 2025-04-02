// 3.3.4 関数ポインタを関数から返す
#include <stdio.h>
#include <string.h>

// int2つを入力に持ち、intを返す関数のポインタ
typedef int (*fptrOperation)(int, int);

int add (int num1, int num2) {
  return num1 + num2;
}

int sub(int num1, int num2) {
  return num1 - num2;
}

// opcode文字からオペレーション用の関数へのポインタを返す
fptrOperation select(char opcode) {
  switch(opcode) {
    case '+': return add;
    case '-': return sub;
  }
}

int compute(char opcode, int num1, int num2) {
  fptrOperation operation = select(opcode);
  return operation(num1, num2);
}

int main(int argc, char *argv[])
{
  char ope;
  printf("Please input operator. +: add, -: sub\n");
  ope = getchar();
  int ans = 0;

  ans = compute(ope, 1, 3);
  printf("%d\n", ans);
  return 0;
}
