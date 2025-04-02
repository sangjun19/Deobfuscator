/* 4)Dado o valor do produto e a forma de pagamento. 1= à vista; 2= à prazo. 
 * Se o produto for pago à vista aplique um desconto de 10% antes de mostrar o valor final, senão informe o mesmo valor do produto.
 */

#include <stdio.h>

int main(void)
{
	int input;
	float valor;

	printf("Informe o valor do produto: ");
	scanf("%f", &valor);
	
	printf("A vista (1) | A prazo (2): ");
	scanf("%d", &input);

	switch (input) {
		case 1:
			printf("Valor a ser pago: %2.f\n", valor*0.9);
			break;
		case 2:
			printf("Valor a ser pago: %2.f\n", valor);
			break;
		default:
			printf("Opção incorreta\n");
			break;
	}

	return 0;
}
