#include <stdio.h>
#include <stdlib.h>

int main()
{
    int n, opcao, media, soma = 0, quant = 0, val_pos = 0, val_neg = 0;
    float porc_pos = 0, porc_neg = 0;
    do
    {
        printf("Escolha uma das opcoes:\n ");
        printf("1. Digitar um novo valor:\n ");
        printf("2. Sair:\n ");
        scanf("%d", &opcao);

        switch (opcao)
        {
        case 1:
            printf("Digite um novo valor:\n");
            scanf("%d", &n);
            quant++;
            soma += n;

            if (n > 0)
            {
                val_pos++;
            }
            else
            {
                val_neg++;
            }
            break;
        case 2:
            printf("Menu finalizado\n");
            break;

        default:
            printf("opcao invalida, digite outro item do menu.\n");
            break;
        }
    } while (opcao != 2);

    if (quant > 0)
    {
        media = soma / quant;
        porc_neg = (val_pos / quant) * 100;
        porc_pos = (val_neg / quant) * 100;
    }

    printf("A quatindade de valores inseridos foram de: %d\n", quant);
    printf("A media e: %d\n", media);
    printf("A quantidade de valores positivos encontrados foram de: %d\n", val_pos);
    printf("A quantidade de valores negativos encontrados foram de: %d\n", val_neg);
    printf("A porcetagem de valores positivos em relacao ao total de numeros encontrados foram de: %.2f%%\n", porc_pos);
    printf("A porcetagem de valores negativos em relacao ao total de numeros encontrados foram de: %.2f%%\n", porc_neg);
}