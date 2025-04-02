#include <stdio.h>

int main(){
	float soma=0;
	float valor;
	int opcao;
	
	do{
		printf("\n M E N U  DE  O P C O E S");
		printf("\n 1. Deposito");
		printf("\n 2. Saque");
		printf("\n 3. Saldo");
		printf("\n 4. Sair");
		
		printf("\n Informe uma opcao: ");
		scanf("%d", &opcao);
		
		switch(opcao){
			
			case 1:
				printf("\m Informe o valor: ");
				scanf("%f", &valor);
				soma += valor;
				break;
			case 2:
				printf("\n Informe o valor: ");
				scanf("%f", &valor);
				soma -= valor;
				break;
			case 3:
				printf("\n Saldo atual = R$ %.2f", soma);
				break;
			case 4:
				printf("\ Saindo...");
				break;
			default:
				printf("\n Opcao invalida!");
		}
			
	} while (opcao != 4);
	
			printf("\n\n Fim das oprecoes!");
			
			return 0;
		
}
