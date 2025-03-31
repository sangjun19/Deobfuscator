-- VETORES --

EXERCÍCIO 1:


#include <stdio.h>

int main()
{
    //variável de loop
    int loop;

    //gerando um vetor
    int A[6];

    //criando uma variável inteira
    int armazenar;

    //atribuindo valores ao vetor
     A[0] = 1;
     A[1] = 0;
     A[2] = 5;
     A[3] = -2;
     A[4] = -5;
     A[5] = 7;

    //armazenando na variável a soma de A[0] e A[1] e A[5]
    armazenar = A[0] + A[1] + A[5];

    //modificando a posição 4
     A[4] = 100;

    //mostrando os valores de A, um em cada linha
    for(loop = 0; loop < 6; loop++)
    {
        printf("%i \n", A[loop]);
    }
}


EXERCÍCIO 2:


#include <stdio.h>

int main()
{
    //declarando o vetor
    int vet[8];

    //declarando as demais variáveis
    int X, Y, loop;

    //estabelecendo valores para as posições do vetor
    for(loop = 0; loop < 8; loop++)
    {
        printf("Insira um valor para uma posicao do loop (posicao %i de 8) \n", loop + 1);
        scanf("%i", &vet[loop]);
    }

    //escolhendo X e Y
    do{
        printf("Insira um valor de posicao do vetor (de 0 a 7) \n");
        scanf("%i", &X);

        printf("Insira um valor de posicao do vetor (de 0 a 7) \n");
        scanf("%i", &Y);

        if((X < 0) || (X > 7))
        {
            printf("Valor do primeiro parametro invalido \n \n");

        }

        if((Y < 0) || (Y > 7))
        {
            printf("Valor do segundo parametro invalido \n \n");

        }
    }while((X < 0) || (X > 7) || (Y < 0) || (Y > 7));

    //imprimindo a soma
    printf("A soma das posicoes eh: %i \n", vet[X] + vet[Y]);
    
    return 0;
}

EXERCÍCIO 3:

#include <stdio.h>

int main()
{
    //declaração do vetor
    int numeros[6];

    //declaração da variável de loop
    int loop;

    //lendo os valores numericos do vetor
    for(loop = 0; loop < 6; loop++)
    {
        printf("Insira um numero inteiro para ser registrado (numero %i de 6) \n", loop + 1);
        scanf("%i", &numeros[loop]);
    }

    printf("A ordem inversa dos numeros eh: \n");

    //imprimindo os valores em ordem inversa
    for(loop = 5; loop >= 0; loop--)
    {
        printf("%i \n", numeros[loop]);
    }

    return 0;
}


EXERCÍCIO 4:

#include <stdio.h>

int main()
{
    //declarando o vetor
    int valores[5];

    //declarando as demais variáveis
    int loop, maior, menor;

    //escolhendo os valores do vetor
    for(loop = 0; loop < 5; loop++)
    {
        printf("Insira os valores para serem armazenados (valor %i de 5) \n", loop + 1);
        scanf("%i", &valores[loop]);

        //declarando maior e menor caso loop seja 0
        if(loop == 0)
        {
            maior = 0;
            menor = 0;
        }

        //declarando novo maior
        else if (valores[loop] >= valores[maior])
        {
            maior = loop;
        }

        //declarando novo menor
        else if(valores[loop] <= valores[menor])
        {
            menor = loop;
        }

    }

    //imprimindo os resultados
    printf("A posicao do maior numero eh: %i \n A posicao do menor numero eh: %i \n", maior, menor);

    return 0;
}


EXERCÍCIO 5: (VERIFICAR)


#include <stdio.h>

int main()
{
    //declarando o vetor
    int vetor[50];

    //declarando a variável de loop
    int i;

    //realizando o loop para preencher os valores do vetor
    for(i = 0; i < 50; i++)
    {
        vetor[i] =  (i + (5 * i)) % (i + 1);
    }

    //imprimindo os valores do vetor
    for(i = 0; i < 50; i++)
    {
        printf("%i ", vetor[i]);
    }

    return 0;
}	


EXERCÍCIO 6:

#include <stdio.h>

int main()
{
    int num[10];

    int primo = 0, loop1, loop2, armazenar = 0,metade, div;

    for(loop1 = 0; loop1 < 10; loop1++)
    {
        printf("Insira um numero para ser armazenado (numero %i de 10): \n", loop1 + 1);
        scanf("%i", &num[loop1]);
    }

    for(loop1 = 0; loop1 < 10; loop1++)
    {
        armazenar = num[loop1];
        metade = armazenar / 2;

        if(armazenar <= 1)
        {
            break;
        }

        if((armazenar == 2) || (armazenar == 3) || (armazenar == 5))
        {
            primo = 1;
        }
        else if(((armazenar % 2) == 0) || ((armazenar % 3) == 0) || ((armazenar % 5) == 0))
        {
            continue;
        }
        else
        {
            for(loop2 = 7; loop2 < metade; loop2++)
            {
                div = metade % loop2;
                if(div == 0)
                {
                  primo = 0;
                  break;
                }
                else
                {
                    primo = 1;
                }
            }
        }

        if(primo == 1)
        {
            printf("O numero %i na posicao %i eh primo \n", armazenar, loop1);
        }

        primo = 0;
    }

    return 0;
}

EXERCÍCIO 7:
#include <stdio.h>

int main()
{
    //declarando o vetor
    int num[6];

    //declarando as variáveis
    int loop, soma = 0, impares = 0;

    //atribuindo valores as posições do vetor
    for(loop = 0; loop < 6; loop++)
    {
        printf("Insira um valor para ser armazenado (valor %i de 6): \n", loop + 1);
        scanf("%i", &num[loop]);
    }

    //vendo se o numero eh par
    for(loop = 0; loop < 6; loop++)
    {
        if((num[loop] % 2) == 0)
        {
            printf("O numero %i eh par \n", num[loop]);
            soma = soma + num[loop];
        }
    }


    //imprimindo a soma dos pares
    printf(" \n \nA soma dos numeros pares eh: %i \n \n", soma);

    //verificando se o numero eh impar
    for(loop = 0; loop < 6; loop++)
    {
        if((num[loop] % 2) != 0)
        {
            printf("O numero %i eh impar \n", num[loop]);
            impares = impares + 1;
        }
    }

    //imprimindo a quantia de impares
    printf("\n \nFoi digitado %i numeros impares", impares);

    return 0;
}


EXERCÍCIO 8:
#include <stdio.h>

int main()
{
    //declarando o vetor
    int num[10];

    //declarando as demais variaveis
    int loop1, loop2;

    //atribuindo valores ao vetor
    for(loop1 = 0; loop1 < 10; loop1++)
    {
        printf("Insira numeros DIFERENTES para serem armazenados (numero %i de 10): \n", loop1 + 1);
        scanf("%i", &num[loop1]);

        //verificando se os numeros são iguais
        for(loop2 = 0; loop2 < loop1; loop2++)
        {
            if(num[loop2] == num[loop1])
            {
                printf("Numero igual (invalido, escreva novamente) \n \n");
                loop1--;
            }
        }
    }

    //imprimindo o vetor:
    printf("Os valores armazenados foram: ");
    for(loop1 = 0; loop1 < 10; loop1++)
    {
        printf("[%i] ", num[loop1]);
    }
    
    return 0;
}


----MATRIZES----
EXERCÍCIO 1:
#include <stdio.h>

int main()
{
    //declarando a matriz
    int mat[4][4];

    //declarando as demais variáveis
    int i, j, quant = 0;

    for(i = 0; i < 4; i++)
    {
        for(j = 0; j < 4; j++)
        {
            printf("Insira um valor para sem armazenado(valor %i de 16) \n",4 * i + j + 1);
            scanf("%i", &mat[i][j]);

            if(mat[i][j] > 10)
            {
                quant = quant + 1;
            }
        }
    }
    printf("Tem %i valores maiores que 10 na matriz", quant);
}

EXERCÍCIO 2:
#include <stdio.h>

int main()
{
    //declarando a matriz
    int mat[5][5];

    //declarando as variáveis de loop
    int loop, loop2;

    //preenchendo a matriz
    for(loop = 0; loop < 5; loop++)
    {
        for(loop2 = 0; loop2 < 5; loop2++)
        {
            if(loop == loop2)
            {
                mat[loop][loop2] = 1;
            }
            else
            {
                mat[loop][loop2] = 0;
            }
            printf("%i  ", mat[loop][loop2]);
        }
        printf("\n");
    }

}


EXERCÍCIO 3:

#include <stdio.h>

int main()
{
    //declarando a matriz
    int mat[4][4];

    //declarando as demais variaveis
    int loop, loop2, loc_coluna, loc_linha;

    //atribuindo valores na matriz e comparando eles
    for(loop = 0; loop < 4; loop++)
    {
        for(loop2 = 0; loop2 < 4; loop2++)
        {
            printf("Insira um valor para ser armazenado (valor %i de 16): \n", 4 * loop + loop2 + 1);
            scanf("%i", &mat[loop][loop2]);

            if((loop == 0) && (loop2 == 0))
            {
                loc_linha = loop;
                loc_coluna = loop2;
            }
            else if(mat[loc_linha][loc_coluna] < mat[loop][loop2])
            {
                loc_linha = loop;
                loc_coluna = loop2;
            }
        }
    }

    //imprimindo o local do maior numero
    printf("O maior numero esta na linha %i e coluna %i", loc_linha, loc_coluna);
}

EXERCÍCIO 4:
#include <stdio.h>

int main()
{
    //declarando a matriz
    int mat[5][5];

    //declarando as demais variaveis
    int loop, loop2, X, verif = 0;

    //declarando valores na matriz
    for(loop = 0; loop < 5; loop++)
    {
        for(loop2 = 0; loop2 < 5; loop2++)
        {
            printf("Insira um valor para ser armazenado na matriz (valor na linha %i e coluna %i) \n",loop, loop2);
            scanf("%i", &mat[loop][loop2]);
        }
    }

    //declarando o valor X
    printf("Insira um valor para ser buscado na matriz: \n");
    scanf("%i", &X);

    //procurando o valor de X na matriz
    for(loop = 0; loop < 5; loop++)
    {
        for(loop2 = 0; loop2 < 5; loop2++)
        {
            if(mat[loop][loop2] == X)
            {
                verif = 1;
                printf("O valor %i esta na linha %i e coluna %i \n", X, loop, loop2);
            }
        }
    }

    //imprimindo se não existir X na matriz
    if(verif == 0)
    {
        printf("Nao encontrado");
    }
    
    return 0;
}

EXERCÍCIO 5:
#include <stdio.h>

int main()
{
    //declarando matriz
    int mat[10][10];

    //declarando as demais variavies
    int i, j;

    //gerando a matriz
    for(i = 0; i < 10; i++)
    {
        for(j = 0; j < 10; j++)
        {
            if(i < j)
            {
                mat[i][j] = 2 * i + 7 * j - 2;
            }
            else if(i == j)
            {
                mat[i][j] = ( 3 * i * i) - 1;
            }
            else
            {
                mat[i][j] = (4 * (i * i * i)) - ( 5 * (j * j));
            }
            printf("[%i   ]", mat[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}

EXERCÍCIO 6:
#include <stdio.h>

int main()
{
    int mat[4][4];

    int loop, loop2;

    for(loop = 0; loop < 4; loop++)
    {
        for(loop2 = 0; loop2 < 4; loop2++)
        {
            printf("Insira um numero no intervalo [1,20] para ser armazenado (numero %i de 16): \n", 4 * loop + loop2 + 1);
            scanf("%i", &mat[loop][loop2]);

            if((mat[loop][loop2] < 1) || (mat[loop][loop2] > 20))
            {
                printf("Numero invalido (menor que 1 ou maior que 20) \n \n");
                loop2--;
            }
        }
    }

    printf("\n \nA matriz inserida eh: \n");

    for(loop = 0; loop < 4; loop++)
    {
        for(loop2 = 0; loop2 < 4; loop2++)
        {
            printf("[%i]  ", mat[loop][loop2]);
        }
        printf("\n");
    }

    //transformando a matriz em triangular superior
    for(loop = 0; loop < 4; loop++)
    {
        for(loop2 = 0; loop2 < 4; loop2++)
        {
            if(loop2 > loop )
            {
                mat[loop][loop2] = 0;
            }
        }
    }

    printf("\n \nA matriz transformnada em triangular superior eh: \n \n");

    //printando a matriz transformada

    for(loop = 0; loop < 4; loop++)
    {
        for(loop2 = 0; loop2 < 4; loop2++)
        {
            printf("[%i]  ", mat[loop][loop2]);
        }
        printf("\n");
    }
    
    return 0;
}

EXERCÍCIO 7:
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int mat[5][5];

    int loop, loop2, loop3, loop4;

    //certificando que a repetição de um mesmo numero não ocorra
    srand(time(NULL));

    //linha
    for(loop = 0; loop < 5; loop++)
    {
        //coluna
        for(loop2 = 0; loop2 < 5; loop2++)
        {
            //atribuindo valor a matriz
            mat[loop][loop2] = rand() % 100;

            //reverificando os valores
            reverificar:

            //verificando linhas ateriores ao atibuido
            for(loop3 = 0; loop3 <= loop; loop3++)
            {

                //caso loop3 for diferente de loop (pode ler todos os valores em todas as linhas)
                if(loop3 != loop)
                {
                    for(loop4 = 0; loop4 <= loop2; loop4++)
                    {
                        while(mat[loop][loop2] == mat[loop3][loop4])
                        {
                            mat[loop][loop2] = rand() % 100;
                            loop3 = 0;
                            loop4 = 0;
                            goto reverificar;
                        }
                    }
                }

                //caso loop3 for igual loop1 (não pode ler a ultima coluna (coluna onde esta o numero inserido)
                else
                    for(loop4 = 0; loop4 < loop2; loop4++)
                    {
                        while(mat[loop][loop2] == mat[loop3][loop4])
                        {
                            mat[loop][loop2] = rand() % 100;
                            loop3 = 0;
                            loop4 = 0;
                            goto reverificar;
                        }
                    }
            }
            //imprimir os numeros nas colunas de uma mesma linha
            printf("[%i] ", mat[loop][loop2]);
        }
        //mudar de linha
        printf("\n");
    }
    return 0;
}


EXERCÍCIO 8:
#include <stdio.h>

int main()
{
    //declarando as matrizes
    int mat1[2][2], mat2[2][2], res[2][2];

    //declarando as demais variaveis
    int loop, loop2, con;
    char escolha;

    //atribuindo valores a matriz 1
    for(loop = 0; loop < 2; loop++)
    {
        for(loop2 = 0; loop2 < 2; loop2++)
        {
            printf("Insira um numero para ser armazenado na primeira matriz (numero %i de 4):", 2 * loop + loop2 + 1);
            scanf("%i", &mat1[loop][loop2]);
        }
    }

    //atribuindo valores a matriz 2
    for(loop = 0; loop < 2; loop++)
    {
        for(loop2 = 0; loop2 < 2; loop2++)
        {
            printf("Insira um numero para ser armazenado na segunda matriz (numero %i de 4):", 2 * loop + loop2 + 1);
            scanf("%i", &mat2[loop][loop2]);
        }
    }

    //abrindo o menu
    do
    {
        fflush(stdin);
        printf("Escolha uma opcao : \n a) somar as duas matrizes \n b) subtrair a primeira matriz da segunda \n c) adicionar uma constante as duas matrizes \n d) imprimir as matrizes \n ");
        scanf("%c", &escolha);

        //caso a escolha seja invalida
        if((escolha != 'a') && (escolha != 'b') && (escolha != 'c') && (escolha != 'd'))
        {
            printf("Escolha invalida, realize novamente \n \n");
        }
    }
    while((escolha != 'a') && (escolha != 'b') && (escolha != 'c') && (escolha != 'd'));


    //switch case
    switch(escolha)
    {


    case 'a':
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                res[loop][loop2] = mat1[loop][loop2] + mat2[loop][loop2];
                printf("%i ", res[loop][loop2]);
            }
            printf(" \n");
        }
        break;


    case 'b':
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                res[loop][loop2] = mat2[loop][loop2] - mat1[loop][loop2];
                printf("%i ", res[loop][loop2]);
            }
            printf(" \n");
        }
        break;


    case 'c':
        printf("Insira a constante: ");
        scanf("%i", &con);


        //matriz 1
        printf("matriz 1: \n");
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                res[loop][loop2] = mat1[loop][loop2] + con;
                printf("%i ", res[loop][loop2]);
            }
            printf(" \n");
        }

        //matriz 2;
        printf("\nMatriz 2: \n");
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                res[loop][loop2] = mat2[loop][loop2] + con;
                printf("%i ", res[loop][loop2]);
            }
            printf(" \n");
        }
        break;

    case 'd':
        printf("matriz 1: \n");
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                ;
                printf("%i ", mat1[loop][loop2]);
            }
            printf(" \n");
        }
        printf("\nMatriz 2: \n");
        for(loop = 0; loop < 2; loop++)
        {
            for(loop2 = 0; loop2 < 2; loop2++)
            {
                printf("%i ", mat2[loop][loop2]);
            }
            printf(" \n");
        }
        break;

    }
    return 0;
    
}

-----STRINGS----	
EXERCÍCIO 1:
#include <stdio.h>
#include <string.h>

int main()
{
    char  str[20];
    printf("Escreva uma frase com no MAXIMO 19 simbolos (pontuacao e espaco sao considerados simbolos):  ");

    setbuf(stdin, NULL);
   gets(str);


    printf("Voce digitou: %s", str);

    return 0;
}

EXERCÍCIO 2:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];
    int loop, qnt = 0;

    printf("Insira uma sequencia de numeros (de 0 a 9)  para retornar a quantidade de '1' nessa sequencia ( o numero maximo de numeros admitidos eh 19)");
    setbuf(stdin, NULL);
    gets(str);

    for(loop = 0; loop < 20; loop++)
    {
        if(str[loop] == '1')
        {
            qnt = qnt + 1;
        }
    }

    printf("Foi inserido %d  numeros 1", qnt);
}

EXERCÍCIO 3:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[50];
    int loop;

    printf("Insira uma palavra para ser exibida de tras para frente (numero maximo decarracteres admitidos eh 19): \n");
    setbuf(stdin, NULL);
    gets(str);

    for(loop = strlen(str); loop >= 0; loop--)
    {
        printf("%c", str[loop]);
    }

    return 0;
}

EXERCÍCIO 4:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];
    int loop, calc = 0;

    printf("Insira uma palava (maximo 19 caracteres):  \n");
    setbuf(stdin, NULL);

    gets(str);

    for(loop = 0; loop < 20; loop++)
    {
        if((str[loop] == 'a') || (str[loop] == 'A') || (str[loop] == 'e') || (str[loop] == 'E') || (str[loop] == 'i') || (str[loop] == 'I')|| (str[loop] == 'o') || (str[loop] == 'O') || (str[loop] =='u') || (str[loop] == 'U'))
        {
            calc++;
            str[loop] = 'z';
        }
    }

    printf("Foi inserido %d vogais \n", calc);

    printf("A palavra com as vogais trocadas por z eh: %s \n", str);
    
    return 0;
}

EXERCÍCIO 5:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];
    int loop;

    printf("insira uma frase para as letras maiusculas serem convertidas em minusculas: \n");
    setbuf(stdin, NULL);
    gets(str);

    for(loop = 0; loop < 20; loop++)
    {
        if((str[loop] >= 65) && (str[loop] <= 90))
        {
            str[loop] = str[loop] + 32;
        }
    }

    printf("A conversao eh: %s", str);


    return 0;
}

EXERCÍCIO 6:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];
    int loop;

    printf("insira uma frase para as letras minusculas serem convertidas em maiusculas: \n");
    setbuf(stdin, NULL);
    gets(str);

    for(loop = 0; loop < 20; loop++)
    {
        if((str[loop] >= 97) && (str[loop] <= 122))
        {
            str[loop] = str[loop] - 32;
        }
    }

    printf("A conversao eh: %s", str);


    return 0;
}

EXERCÍCIO 7:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];
    int loop, loop2;

    printf("insira uma frase para as os espacos serem removidos: \n");
    setbuf(stdin, NULL);
    gets(str);

    for(loop = 0; loop < 20; loop++)
    {
        if(str[loop] == 32)
        {
            continue;
        }
        else
        {
            printf("%c", str[loop]);
            if(str[loop] == '\0')
            {
                break;
            }
        }
    }

    return 0;
}

EXERCÍCIO 8:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[50];

    char L1, L2;

    int loop;

    printf("Insira uma palavra ou frase: \n");

    //limpando o buffer do teclado
    setbuf(stdin, NULL);
    gets(str);

    //decidindo qual letra deve ser mudada
    printf("Insira a leta que deseja ser modificada: ");
    scanf("%c", &L1);

    setbuf(stdin, NULL);
    printf("Insira a leta que deseja substituir a anterior: ");
    scanf("%c", &L2);

    for(loop = 0; loop < strlen(str); loop++)
    {
        if(str[loop] == L1)
        {
            str[loop] = L2;
        }
        else if(str[loop] == L1 - 32)
        {
            str[loop] = L2 - 32;
        }
    }

    printf("A frase modificada eh: \n %s", str);

}

EXERCÍCIO 9:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[5][80];

    int consumo[5];

    int loop, economico;

    for(loop = 0; loop < 5; loop++)
    {
        setbuf(stdin, NULL);

        printf("Insira um modelo de carro: ");
        gets(str[loop]);

        printf("Insira o consumo desse modelo: ");
        scanf("%i", &consumo[loop]);
        printf("\n");
    }

    for(loop = 0; loop < 5; loop++)
    {
        if(loop == 0)
        {
            economico = 0;
        }

        else if(consumo[loop] < consumo[economico])
        {
            economico = loop;
        }
    }

    printf("O modelo mais economico eh o %s \n \n", str[economico]);

    for(loop = 0; loop < 5;loop++)
    {
        printf("O consumo do modelo %s em 1000 km eh: %i \n", str[loop], consumo[loop] * 1000);
    }

    return 0;
}

EXERCÍCIO 10:
#include <stdio.h>
#include <string.h>

int main()
{
    char nome[1000];
    float desconto, preco_incial, preco_vista;

    printf("Insira o nome da mercadoria: ");
    setbuf(stdin, NULL);
    gets(nome);

    printf("Insira o preco da mercadoria sem desconto: ");
    scanf("%f", &preco_incial);

    desconto = preco_incial * 0.1;
    preco_vista = 0.9 * preco_incial;

    printf("O produto de nome: %s \nPreco do produto: %0.2f \nPreco do produto a vista: %0.2f \nDesconto a vista: %0.2f", nome, preco_incial, preco_vista, desconto);
    return 0;
}



EXERCÍCIO 11:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[100101];
    int i, j, verif, loop;

    printf("Insira uma frase e/ou palavra para ser armazenado na string: ");
    setbuf(stdin, NULL);
    gets(str);

    do
    {
        printf("Insira dois valores numericos inteiros nao negativos e menores que 100100:");
        scanf("%i %i", &i, &j);

        if((i > 100100) || (i < 0) || (j > 100100) || (j < 0))
        {
            printf("Ao menos um dos valores digitados nao seguiu as delimitacoes especificadas (nao negativo e menor que 100100). Insira novamente os numeros \n \n");
        }
    }while((i > 100100) || (i < 0) || (j > 100100) || (j < 0));

    if(i > j)
    {
        verif = 1;
    }
    else
    {
        verif = 2;
    }

    switch(verif)
    {
    case 1:
        for(loop = j; loop <= i; loop++)
        {
            printf("%c", str[loop]);
        }
        break;

    case 2:
        for(loop = i; loop <= j; loop++)
        {
            printf("%c", str[loop]);
        }
        break;
    }
    return 0;
}


EXERCÍCIO 12:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[20];

    int k, i;

    printf("Insira a string: \n");
    gets(str);

    k = strlen(str);

    for(i = 0; i < strlen(str); i++)
    {
        if((str[i] >= 'a') && (str[i] < 'x'))
        {
            str[i] = str[i] - 29;
        }
        if((str[i] >= 'x') && (str[i] <= 'z'))
        {
            str[i] = (str[i] - 32) - 23;
        }
        if((str[i] >= 'A') && (str[i] < 'X'))
        {
            str[i] = str[i] + 3;
        }
        if((str[i] >= 'X') && (str[i] <= 'Z'))
        {
            str[i] = str[i] - 23;
        }
    }

    printf("%s", str);
}

EXERCÍCIO 13:
#include <stdio.h>
#include <string.h>

int main()
{
    char str[1000];
    char inv[1000];

    int i, max, k, m, l;

    printf("Insira uma frase e / ou palavra: \n");
    setbuf(stdin, NULL);
    gets(str);

    max = strlen(str);

    //retirando espaços, - etc
    for(i = 0; i < max; i++)
    {
        if((str[i] < 65) || ((str[i] > 90) && (str[i] < 97)) || (str[i] > 122))
        {
            for(m = i + 1; m <= max; m++)
            {
                l = m  - 1;
                str[l] = str[m];
            }
        }
    }

    //colocando letras maiusculas em minusculas
    for(i = 0; i < max; i++)
    {
        if((str[i] >= 65) && (str[i] <= 90))
        {
            str[i] = str[i] + 32;
        }
    }

    max = strlen(str);
    k = max - 1;

    //invertendo a matriz
    for(i = 0; i < max; i++)
    {
        inv[k] = str[i];
        k--;
    }
    inv[max] = '\0';

    //comparando as matrizes
    if(strcmp(str, inv) == 0)
    {
        printf("Eh palindromo.");
    }
    else
    {
        printf("Nao eh palindromo.");
    }
    return 0;
}

EXERCÍCIO 14:
#include <stdio.h>
#include <string.h>

int main()
{
    char str1[1000];
    char str2[1000];
    int N, i, max, conc, k;

    printf("Insira a primeira string: \n");
    setbuf(stdin, NULL);
    gets(str1);

    printf("Insira a segunda string: \n");
    setbuf(stdin, NULL);
    gets(str2);

    printf("Insira quantos caracteres da primeira string deve ser concatenado na segunda: \n");
    scanf("%i", &N);

    max = strlen(str1);
    conc = max + N;
    k = 0;

    for(i = max; i < conc; i++)
    {
        str1[i] = str2[k];
        k++;
    }
    str1[i] = '\0';
    printf("%s", str1);

    return 0;
}

EXERCÍCIO 15: VERIFICAR H E I
#include <stdio.h>
#include <string.h>

int main()
{
    //declarar variáveis
    char S1[20], S2[20], cat[1000], inv[1000], sub[10000];
    char car, car1, car2;
    int i, k = 0, tam1, tam2, qant = 0, numeros = 0, pos;

    //exercício A
    printf("Insira uma string S1: \n");
    setbuf(stdin, NULL);
    gets(S1);

    //exercício B
    tam1 = strlen(S1);
    printf("O tamanho da string S1 eh: %i \n \n ", tam1);

    //Exercício C
    printf("Insira uma string S2: \n");
    setbuf(stdin, NULL);
    gets(S2);

    if(strcmp(S1, S2) == 0)
    {
        printf("As strings sao iguais. \n \n");
    }
    else
    {
        printf("As strings nao sao iguais. \n \n");
    }

    //Exercício D
    strcpy(cat, S1);
    strcat(cat, S2);
    printf("A string apos a concatenar eh: %s \n \n", cat);

    //exercício E
    for(i = tam1 - 1; i >= 0; i--)
    {
        inv[k] = S1[i];
        k++;
    }
    inv[k] = '\0';
    printf("A string s1 invertida eh: %s \n \n", inv);

    //exercício F
    printf("Insira um caractere para ser procurado na string S1: ");
    setbuf(stdin, NULL);
    scanf("%c", &car);
    for(i = 0; i < strlen(S1); i++)
    {
        if(S1[i] == car)
        {
            qant += 1;
        }
    }
    printf("O caractere %c aparece %i vezes na string S1. \n \n", car, qant);

    //Exercício G
    printf("Insira um caractere para ser substituido em sua primeira ocorrencia em S1: \n");
    setbuf(stdin, NULL);
    scanf("%c", &car1);

    printf("Insira um caractere para substituir o digitado anteriormente: \n");
    setbuf(stdin, NULL);
    scanf("%c", &car2);

    for(i = 0; i <= strlen(S1); i++)
    {
        if(i == strlen(S1))
        {
            printf("Caractere nao encontrado. \n \n");
        }

        if(S1[i] == car1)
        {
            S1[i] = car2;
            printf("A nova string S1 eh: %s \n \n", S1);
            S1[i] = car1;
            break;
        }
    }

    //Exercício H
    printf("Insira uma string S2 para verificar se ela eh substring de S1: \n");
    setbuf(stdin, NULL);
    gets(S2);

    k = 0;
    for(i = 0; i < strlen(S1); i++)
    {
        if(numeros == strlen(S2))
        {
            break;
        }
        if(S1[i] == S2[k])
        {
            numeros++;
            k++;
        }
        else if(S1[i] != S2[k])
        {
            numeros = 0;
            k = 0;
        }
    }
    if(numeros == strlen(S2))
    {
        printf("S2 eh substring de S1. \n \n");
    }
    else
    {
        printf("S2 nao eh substring de S1. \n \n");
    }

    //exercício I
    printf("Insira em qual posicao deve se iniciar a formacao da substring de S1: \n");
    scanf("%i", &pos);

    printf("Insira o tamanho da string a ser formada: \n");
    scanf("%i", &tam1);

    tam1--;
    k = 0;
    for(i = pos; i <= (pos + tam1); i++)
    {
        if(i == strlen(S1))
        {
            break;
        }
        sub[k] = S1[i];
        k++;
    }
    sub[i] = '\0';
    printf("A substring eh %s \n", sub);
}

EXERCÍCIO 16:
#include <stdio.h>
#include <string.h>

int main()
{
    char string[] = "7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530";
    int i, j, k, l;
    int maior = 0, produto;
    k = strlen(string);

    for(i = 0; i < k - 4; i++)
    {
        produto = 1;
        for(j = i; j <= i + 4; j++)
        {
            produto = produto * (string[j] - '0');
        }
        if(produto > maior)
        {
            maior = produto;
        }
    }
    printf(" O maior produto eh: %i", maior);
    return 0;
}



