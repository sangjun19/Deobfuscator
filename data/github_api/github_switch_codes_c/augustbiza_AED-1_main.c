//CALCULADORA SIMPLES
#include <stdio.h>

int main(void) {

    printf("Calculadora Simples.\n");

    int op;
    float num1, num2;

    printf("Operacoes:\n1-Adica0\n2-Subtracao\n3-Multiplicacao\n4-Divisao\nInsira a operação desejada: \n");
    scanf("%d", &op);

    printf("Insira o primeiro numero: ");
    scanf("%f", &num1);
    printf("Insira o segundo numero: ");
    scanf("%f", &num2);

    switch(op) {
        case 1: printf("%f + %f = %f\n", num1, num2, num1+num2);
            break;
        case 2: printf("%f - %f = %f\n", num1, num2, num1-num2);
            break;
        case 3: printf("%f * %f = %f\n", num1, num2, num1*num2);
            break;
        case 4: printf("%f / %f = %f\n", num1, num2, num1/num2);
            break;
        default: printf("Erro.\n\n");
    }

    return 0;
}