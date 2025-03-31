#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#define LIM 5
#define TAM_CAD 224
#define TAM_EMPR 678
#define PRAZO_DEV 14

typedef struct{
    char nome[50];
    int num;
    int senha;
    int data_nasc[3];
    char rua[40];
    int num_casa;
    char complemento[20];
    char bairro[40];
    char cidade[30];
    char estado[20];
} t_cadastro;

typedef struct{
    char nome[50];
    int num;
    int qtd;
    int dia_empr[LIM];
    int mes_empr[LIM];
    int ano_empr[LIM];
    int dia_dev[LIM];
    int mes_dev[LIM];
    int ano_dev[LIM];
    char nome_livro[LIM][50];
    char nome_autor[LIM][50];
}t_emprestimo;

int menu(){
    char aux[3];
    int opcao;
    printf("\n\tBIBLIOTECA\n");
    printf("\t>>Menu<<\n");
    printf("1. Novo cadastro\n");
    printf("2. Emprestimo\n");
    printf("3. Busca de historico\n");
    printf("4. Devolver Livro\n");
    printf("5. Sair\n");
    while(1){
        printf("Digite sua opcao: \n");
        fflush(stdin);
        fgets(aux, 3, stdin);
        if((aux[0]=='1')||(aux[0] == '2')||(aux[0] == '3')||(aux[0] == '4')||(aux[0]=='5')){
            opcao = atoi(aux);
            return opcao;
        }
    }
}

int verifica_cad(t_cadastro *usuario, int i, char temp[50], char *url_cad){
    int endereco, flag;
   	FILE *Arq = fopen(url_cad, "r+b");

    if (Arq == NULL) {
            printf("Erro, nao foi possivel abrir o arquivo\n");
            return -1;
    }
    while(1){
        fread(usuario->nome, sizeof(char), 50, Arq);
        if(strcmp(temp, usuario->nome)==0){ //sao iguais
            endereco = ftell(Arq); //indica onde esta o nome no arquivo
            fclose(Arq);
            return endereco;
        }else{
            fseek(Arq, TAM_CAD-50, SEEK_CUR); //pula para o proximo cadastro
            if(fread(&flag, 1, 1, Arq) == 0) //verifica fim do arquivo
                break;
            else
                fseek(Arq, -1, SEEK_CUR);
        }
    }
    fclose(Arq);
    return 0;

}

int cadastro(t_cadastro *usuario, int i, char *url_cad){
    char temp[50];
    int senhatemp;
    printf("Nome: ");
    fflush(stdin);
    fgets(temp, 50, stdin);
    int flag = verifica_cad(usuario, i, temp, url_cad);

    if(flag != 0){
        if(flag == -1){
            return -1; //arquivo nao aberto
        }else{
            printf("Ja existe um cadastro nesse nome\n");
            return 1;
        }
    }else{ //nao existe cadastro
        strcpy(usuario->nome,temp);

		usuario->num= i+100000;
        printf("Numero de cadastro: %d\n", usuario->num);

        //verifica senha
        while(1){
            printf("Senha numerica [pelo menos 4 algarismos]: ");
            fflush(stdin);

            fgets(temp, 10, stdin);
            senhatemp = atoi(temp);
            while(senhatemp < 1000){
                printf("Digite uma senha de pelo menos 4 algarismos\n");
                printf("Senha numerica: ");
                fflush(stdin);
    
                fgets(temp, 10, stdin);
                senhatemp = atoi(temp);
            }
            printf("Digite a senha novamente: ");
            fflush(stdin);

            fgets(temp, 10, stdin);
            usuario->senha= atoi(temp);
            if(senhatemp == usuario->senha)
                break;
        }

        printf("Data de nascimento:\n");
        printf("Dia: ");
        fflush(stdin);
        fgets(temp, 5, stdin);
        usuario->data_nasc[0] = atoi(temp);
        printf("Mes: ");
        fflush(stdin);
        fgets(temp, 5, stdin);
        usuario->data_nasc[1] = atoi(temp);
        printf("Ano: ");
        fflush(stdin);
        fgets(temp, 5, stdin);
        usuario->data_nasc[2] = atoi(temp);

        printf("Endereco:\n");
        fflush(stdin);
        printf("Rua: ");
        fflush(stdin);
        fgets(temp, 50, stdin);
        strcpy(usuario->rua,temp);

        printf("Numero: ");
        fflush(stdin);
        fgets(temp, 5, stdin);
        usuario->num_casa=atoi(temp);

        printf("Complemento: ");
        fflush(stdin);
        fgets(temp, 50, stdin);
        strcpy(usuario->complemento,temp);

        printf("Bairro: ");
        fflush(stdin);
        fgets(temp, 50, stdin);
        strcpy(usuario->bairro,temp);

        printf("Cidade: ");
        fflush(stdin);
        fgets(temp, 50, stdin);
        strcpy(usuario->cidade,temp);

        printf("Estado: ");
        fflush(stdin);
        fgets(temp, 50, stdin);
        strcpy(usuario->estado,temp);
        return 0;
    }
}

void salva_cadastro(t_cadastro *usuario, char *url_cad){
    FILE *Arq=fopen(url_cad, "ab");

    if(Arq == NULL){
        printf("Erro, nao foi possivel abrir o arquivo\n");
    }else{
        fwrite(&usuario->nome, sizeof(char), 50, Arq);
        fwrite(&usuario->num, sizeof(int), 1, Arq);
        fwrite(&usuario->senha, sizeof(int), 1, Arq);
        fwrite(&usuario->data_nasc[0], sizeof(int), 1, Arq);
        fwrite(&usuario->data_nasc[1], sizeof(int), 1, Arq);
        fwrite(&usuario->data_nasc[2], sizeof(int), 1, Arq);
        fwrite(&usuario->rua, sizeof(char), 40, Arq);
        fwrite(&usuario->num_casa, sizeof(int), 1, Arq);
        fwrite(&usuario->complemento, sizeof(char), 20, Arq);
        fwrite(&usuario->bairro, sizeof(char), 40, Arq);
        fwrite(&usuario->cidade, sizeof(char), 30, Arq);
        fwrite(&usuario->estado, sizeof(char), 20, Arq);

    }
    fclose(Arq);
}

int confirma_senha(int endereco, char *url_cad, char nome[50], int modo){
    char aux[50];
    int senha1, senha2;
    FILE *Arq = fopen(url_cad, "rb");

    if(Arq == NULL){
        printf("Erro, nao foi possivel abrir o arquivo\n");
    } else {
        if(modo==1){
            fseek(Arq, endereco + sizeof(int), SEEK_SET);
            fread(&senha1, sizeof(int),1, Arq);
            while(1){
                printf("Confirme sua senha (digite 0 para sair): ");
                fflush(stdin);
    
                fgets(aux, 10, stdin);
                senha2 = atoi(aux);
                if(senha1 == senha2){
                    fclose(Arq);
                    return 1;
                }else if (senha2 == 0){ //sair da funcao
                    fclose(Arq);
                    return 0;
                }else{
                    printf("Senha incorreta\n");
                }
            }
        }
        if(modo==2){
            while(1){
                fread(aux,sizeof(char),50,Arq);
                if(strcmp(aux,nome)==0){
                    fseek(Arq, sizeof(int), SEEK_CUR);
                    fread(&senha1, sizeof(int),1, Arq);
                    while(1){
                        printf("Confirme sua senha (digite 0 para sair): ");
                        fflush(stdin);
            
                        fgets(aux, 10, stdin);
                        senha2 = atoi(aux);
                        if(senha1 == senha2){
                            fclose(Arq);
                            return 1;
                        }else if (senha2 == 0){ //sair da funcao
                            fclose(Arq);
                            return 0;
                        }else{
                            printf("Senha incorreta\n");
                        }
                    }
                } else {
                    fseek(Arq, TAM_CAD-(50*sizeof(char)), SEEK_CUR); //pula para o proximo cadastro
                        if(fread(aux, 1, 1, Arq) == 0) //verifica fim do arquivo
                            break;
                        else
                            fseek(Arq, -1, SEEK_CUR); //volta 1 byte
                }
            }
        }
    fclose(Arq);
    return 0;

    }
}

void devolucao(int dia, int mes, int ano, int *dia_dev, int *mes_dev, int *ano_dev){
    int dia_mes = 31;
    switch (mes){
        case 2: //fevereiro
            dia_mes = dia_mes - 3;
            break;
        case 4: //abril
        case 6: //junho
        case 9: //setembro
        case 11: //novembro
            dia_mes--;
            break;
    }
    if(dia + PRAZO_DEV > dia_mes){ //se ha mudanca de mes
        if(mes == 12){
            *ano_dev = ano++;
            *mes_dev = 1;
            *dia_dev = dia + PRAZO_DEV - dia_mes;
        }else{
            *ano_dev = ano;
            *mes_dev = mes++;
            *dia_dev = dia + PRAZO_DEV - dia_mes;
        }
    }else{ //continua no mesmo mes
        *ano_dev = ano;
        *mes_dev = mes;
        *dia_dev = dia + PRAZO_DEV;
    }
}


int verifica_emprestimo(char nome[50], char *url_empr){
    t_emprestimo emprestimo;
    int flag;

    FILE *Arq = fopen(url_empr, "rb");

    if (Arq == NULL){
        printf("Erro ao abrir o arquivo de emprestimo\n");
        return -1;
    }else{
        while (1){
            fread(&emprestimo.nome, sizeof(char), 50, Arq);
            if (strcmp(nome, emprestimo.nome) == 0){
                fclose(Arq);
                return 1; //ja tem emprestimo
            }else{
                fseek(Arq, TAM_EMPR - 50, SEEK_CUR); //proximo emprestimo
                if(fread(&flag, sizeof(char), 1, Arq) == 0) //verifica o fim do arquivo
                    return 0; //nao tem emprestimo
                else
                    fseek(Arq, -1, SEEK_CUR);
            }
        }
    }

    fclose(Arq);
    return 0;
}
void novo_emprestimo(t_cadastro usuario, int endereco, int dia, int mes, int ano, char *url_cad, char *url_empr){
    t_emprestimo emprestimo;
    int cont, dia_dev, mes_dev, ano_dev;
    FILE *Arq = fopen(url_cad, "rb");

    if(Arq == NULL){
        printf("Erro ao abrir o arquivo de cad\n");
    }else{
        fseek(Arq, endereco - 50, SEEK_SET); //vai ate o fim do nome da pessoa e volta novamente para le-lo
        fread(&emprestimo.nome, 50, 1, Arq);
        fread(&emprestimo.num, sizeof(int), 1, Arq);
    }
    fclose(Arq);


    Arq = fopen(url_empr, "ab");
    if(Arq == NULL){
        printf("Erro ao abrir o arquivo de empr\n");
    }else{
        printf("Digite a quantidade de livros a serem emprestados: ");
        scanf("%d", &emprestimo.qtd);
        fflush(stdin);
        while(emprestimo.qtd > LIM){
            printf("Erro: quantidade de livros acima da permitida\n");
            printf("Digite a quantidade de livros a serem emprestados: ");
            scanf("%d", &emprestimo.qtd);
        }
        for(cont = 0; cont < emprestimo.qtd; cont++){
            
            printf("Digite o nome do livro %d: ", cont+1);
            fflush(stdin);

            fgets(emprestimo.nome_livro[cont], 50, stdin);
            printf("Digite o nome do autor: ");
            fflush(stdin);

            fgets(emprestimo.nome_autor[cont], 50, stdin);
        }
        //salvar no arquivo nome, num de id e quantidade de livros
        fwrite(&emprestimo.nome, sizeof(emprestimo.nome), 1, Arq);
        fwrite(&emprestimo.num, sizeof(int), 1, Arq);
        fwrite(&emprestimo.qtd, sizeof(int), 1,Arq);

        //saber o dia da devolucao
        devolucao(dia, mes, ano, &dia_dev, &mes_dev, &ano_dev);
        for(cont = 0; cont < emprestimo.qtd; cont++){
            emprestimo.dia_dev[cont]=dia_dev;
            emprestimo.mes_dev[cont]=mes_dev;
            emprestimo.ano_dev[cont]=ano_dev;
        }

        //reservar o espaco no binario de uma vez para LIM*livros
        for(cont=0; cont<LIM; cont++){
            fwrite(&dia, sizeof(int), 1, Arq);
            fwrite(&mes, sizeof(int), 1, Arq);
            fwrite(&ano, sizeof(int), 1, Arq);
            fwrite(&emprestimo.dia_dev[cont], sizeof(int), 1, Arq);
            fwrite(&emprestimo.mes_dev[cont], sizeof(int), 1, Arq);
            fwrite(&emprestimo.ano_dev[cont], sizeof(int), 1, Arq);
            fwrite(&emprestimo.nome_livro[cont], sizeof(emprestimo.nome_livro[cont]), 1, Arq);
            fwrite(&emprestimo.nome_autor[cont], sizeof(emprestimo.nome_autor[cont]), 1, Arq);
        }
    }
    fclose(Arq);
}

//adiciona os livros novos e altera a quantidade
void altera_emprestimo(int dia, int mes, int ano, char *url_empr, char *url_cad, int modo){
    t_emprestimo emprestimo;
    int qtd, cont, dia_dev, mes_dev, ano_dev, flag, *endereco;
    char livro[LIM-1][50],autor[LIM-1][50], nome[50], temp[10], rmvlivro='$', testchar;
    FILE *Arq = fopen(url_empr, "r+b");
    endereco = (int*)calloc(LIM,sizeof(int));

    if(Arq == NULL ){
        printf("Erro, nao foi possivel abrir o arquivo\n");
    }else{
        printf("Digite o nome: ");
        fflush(stdin);
        fgets(nome, 50, stdin);
        confirma_senha(endereco[0], url_cad, nome, 2);
        if (modo==1){
            printf("Digite a quantidade de livros: ");
		fflush(stdin);
        scanf("%d", &qtd);
        }
		fflush(stdin);
        while(1){
            fread(emprestimo.nome, sizeof(char), 50, Arq);
            if(strcmp(nome, emprestimo.nome)==0){ //verifica se encontrou o nome
                fread(&emprestimo.num, sizeof(int), 1, Arq);
                fread(&emprestimo.qtd, sizeof(int), 1, Arq);

                //verfica multa
                if (modo == 1){
                    flag=0;
                    for (cont=0; cont < emprestimo.qtd; cont++){
                        fseek(Arq, 3*sizeof(int), SEEK_CUR);
                        fread(&emprestimo.dia_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.mes_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.ano_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.nome_livro[cont], sizeof(char), 50, Arq);
                        fseek(Arq, 50 * sizeof(char), SEEK_CUR);
                        if(emprestimo.ano_dev[cont] < ano){
                            printf("Livro %d atrasado\n",cont+1);
                             flag++;
                        }
                        if(emprestimo.ano_dev[cont] == ano){
                            if(emprestimo.mes_dev[cont]<mes){
                                printf("Livro %d atrasado\n",cont+1);
                                flag++;
                            }
                            if(emprestimo.mes_dev[cont] == mes)
                                if(emprestimo.dia_dev[cont]<dia){
                                    printf("Livro %d atrasado\n",cont+1);
                                    flag++;
                                }
                        }
                        fseek(Arq, 50*sizeof(char), SEEK_CUR);
                    }

                    if(flag != 0){
                        printf("Nao pode realizar emprestimos devido aos atrasos\n");
                        break;
                    }


                    while(qtd + emprestimo.qtd > LIM && modo ==1){ //verificacao valida apenas para insercao de novos livros
                        printf("Erro: quantidade acima da permitida\n");
                        printf("Digite a quantidade de livros (max %d): ",(LIM-emprestimo.qtd));
                        scanf("%d", &qtd);
                    }

                    for(cont=0;cont<qtd;cont++){ //nomes dos novos livros a serem adicionados
                        fflush(stdin);
                        printf("Digite o nome do livro %d: ",cont+1);
                        fflush(stdin);
            
                        fgets(livro[cont],50,stdin);
                        printf("Digite o nome do autor do livro %d: ",cont+1);
                        fflush(stdin);
            
                        fgets(autor[cont],50,stdin);
                    }

                    devolucao(dia, mes, ano, &dia_dev, &mes_dev, &ano_dev);//saber os dias de devolucao dos livros novos
                    for(cont=emprestimo.qtd; cont < LIM; cont++){
                        emprestimo.dia_dev[cont]=dia_dev;
                        emprestimo.mes_dev[cont]=mes_dev;
                        emprestimo.ano_dev[cont]=ano_dev;
                    }
                    if(emprestimo.qtd<5){
                        for(cont=0; cont < qtd; cont++){
                                fwrite(&dia, sizeof(int), 1, Arq);
                                fwrite(&mes, sizeof(int), 1, Arq);
                                fwrite(&ano, sizeof(int), 1, Arq);
                                fwrite(&emprestimo.dia_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&emprestimo.mes_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&emprestimo.ano_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&livro[cont], sizeof(char), 50, Arq);
                                fwrite(&autor[cont], sizeof(char), 50, Arq);
                        }
                    } else {
                        fseek(Arq, -emprestimo.qtd*(6*sizeof(int)+sizeof(emprestimo.nome_livro)+sizeof(emprestimo.nome_autor)),SEEK_CUR);
                        fseek(Arq,6*sizeof(int),SEEK_CUR);
                        for(cont=0; cont<LIM; cont++){
                            fread(&testchar, sizeof(char), 1,Arq);
                            fseek(Arq,-sizeof(char),SEEK_CUR);
                            if(testchar == rmvlivro)
                                endereco[cont] = ftell(Arq);

                            fseek(Arq,50*sizeof(char)+6*sizeof(int),SEEK_CUR);
                        }
                        for(cont=0;cont<LIM;cont++){
                            if(endereco[cont]!=0){
                                //posiciona o ponteiro no lugar certo para sobrescrever o livro devolvido
                                fseek(Arq, endereco[cont],SEEK_SET);
                                fseek(Arq,-6*sizeof(int),SEEK_CUR);

                                fwrite(&dia, sizeof(int), 1, Arq);
                                fwrite(&mes, sizeof(int), 1, Arq);
                                fwrite(&ano, sizeof(int), 1, Arq);
                                fwrite(&emprestimo.dia_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&emprestimo.mes_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&emprestimo.ano_dev[cont+emprestimo.qtd], sizeof(int), 1, Arq);
                                fwrite(&livro[cont], sizeof(char), 50, Arq);
                                fwrite(&autor[cont], sizeof(char), 50, Arq);
                            }
                        }
                    }
                    //mudanca na quantidade de livros
                    emprestimo.qtd += qtd;
                    fseek(Arq, (-1)*emprestimo.qtd*(6*sizeof(int)+sizeof(emprestimo.nome_livro)+sizeof(emprestimo.nome_autor)), SEEK_CUR);
                    fseek(Arq, -(long)sizeof(int), SEEK_CUR);
                    fwrite(&emprestimo.qtd, sizeof(int), 1, Arq);
				// modo 2
                } else {
                    printf("Escolha o livro devolvido:\n");
                    for(cont=0; cont<emprestimo.qtd; cont++){
                        fseek(Arq, 3*sizeof(int), SEEK_CUR);
                        fread(&emprestimo.dia_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.mes_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.ano_dev[cont], sizeof(int), 1, Arq);
                        fread(&emprestimo.nome_livro[cont], sizeof(char), 50, Arq);
						fread(&emprestimo.nome_autor[cont], sizeof(char), 50, Arq);
                        printf("%d. Nome: %s   Autor: %s(%d/%d/%d)\n",cont+1,emprestimo.nome_livro[cont],emprestimo.nome_autor[cont],emprestimo.dia_dev[cont],emprestimo.mes_dev[cont],emprestimo.ano_dev[cont]);
                    }
                    fflush(stdin);
        
                    fgets(temp, 5, stdin);
                    flag = atoi(temp);
                    fseek(Arq, (-1)*emprestimo.qtd*(6*sizeof(int)+100*sizeof(char)), SEEK_CUR);//volta o ponteiro para o primeiro livro
                    fseek(Arq, -sizeof(int), SEEK_CUR); //volta o ponteiro para sobrescrever qtd

                    emprestimo.qtd--;
                    fwrite(&emprestimo.qtd, sizeof(int), 1, Arq);
					fseek(Arq,(flag-1)*(6*sizeof(int)+100*sizeof(char)),SEEK_CUR); // posiciona o ponteiro no livro que sera apagado
					fseek(Arq, 6*sizeof(int), SEEK_CUR);
					fwrite(&rmvlivro, sizeof(char), 1, Arq);
                    fseek(Arq,-sizeof(char),SEEK_CUR);

                }
            } else { //nao achou o nome
                fseek(Arq, TAM_EMPR - 50*sizeof(char), SEEK_CUR); //le ate o proximo emprestimo
                if(fread(&flag, 1, 1, Arq) == 0) //verifica o fim do arquivo
                    break;
                else
                    fseek(Arq, -1, SEEK_CUR);
            }
        }
    fclose(Arq);
    }
}

void imprimir_emprestimo(char *nome, char *url_empr){
    t_emprestimo emprestimo;
    char flag;
    int cont;
    FILE *Arq = fopen(url_empr, "r+b");

    if(Arq == NULL){
        printf("Erro ao abrir o arquivo\n");
    }else{
        while(1){
            fread(&emprestimo.nome, 50, 1, Arq);
            if(strcmp(nome, emprestimo.nome)==0){ //iguais
                //ler arquivo
                fread(&emprestimo.num, sizeof(int), 1, Arq);
                fread(&emprestimo.qtd, sizeof(int), 1, Arq);
                for(cont=0; cont < emprestimo.qtd; cont++){ //nao le o lixo (se tiver)
                    fread(&emprestimo.dia_empr[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.mes_empr[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.ano_empr[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.dia_dev[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.mes_dev[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.ano_dev[cont], sizeof(int), 1, Arq);
                    fread(&emprestimo.nome_livro[cont], sizeof(char), 50, Arq);
                    fread(&emprestimo.nome_autor[cont], sizeof(char), 50, Arq);
                    if(emprestimo.nome_livro[cont][0]== '$'){
                        cont--;
                        continue;
                    }
                }

                //printa na tela
                printf("Quantidade de livros: %d\n", emprestimo.qtd);
                for(cont=0; cont < emprestimo.qtd; cont++){
                    printf("Livro %d:Data de emprestimo: %d/%d/%d\n",cont+1, emprestimo.dia_empr[cont], emprestimo.mes_empr[cont], emprestimo.ano_empr[cont]);
                    printf("Livro %d:Data de devolucao: %d/%d/%d\n",cont+1, emprestimo.dia_dev[cont], emprestimo.mes_dev[cont], emprestimo.ano_dev[cont]);
                    printf("Livro %d: %sAutor %s\n", cont+1, emprestimo.nome_livro[cont], emprestimo.nome_autor[cont]);
                }
                break;
            }else{ //nao achou o nome
                fseek(Arq, TAM_EMPR - (50*sizeof(char)), SEEK_CUR); //vai ate o proximo emprestimo
                if(fread(&flag, 1, 1, Arq) == 0){ //verifica fim do arquivo
                    fclose(Arq);
                    break;
                }else
                    fseek(Arq, -1, SEEK_CUR);
            }
        }
    }
}

void imprimir_cadastro(char *url_cad, char *url_empr){
    t_cadastro usuario;
    char nome[50], flag;

    FILE *Arq = fopen(url_cad, "rb");
    if(Arq == NULL){
        printf("Erro ao abrir o arquivo\n");
    }else{
        printf("Digite o nome a ser procurado: ");
        fflush(stdin);
        fgets(nome, 50, stdin);

        while(1){
            fread(usuario.nome, 50, 1, Arq);
            if(strcmp(nome, usuario.nome)==0){ //sao iguais
                fread(&usuario.num, sizeof(int), 1, Arq);
                fseek(Arq, sizeof(int), SEEK_CUR);
                fread(&usuario.data_nasc[0], sizeof(int), 1, Arq);
                fread(&usuario.data_nasc[1], sizeof(int), 1, Arq);
                fread(&usuario.data_nasc[2], sizeof(int), 1, Arq);
                fread(&usuario.rua, sizeof(char), 40, Arq);
                fread(&usuario.num_casa, sizeof(int), 1, Arq);
                fread(&usuario.complemento, 20, 1, Arq);
                fread(&usuario.bairro, sizeof(char), 40, Arq);
                fread(&usuario.cidade, sizeof(char), 30, Arq);
                fread(&usuario.estado, sizeof(char), 20, Arq);
                    //printa na tela
                printf("Nome: %s\n", usuario.nome);
                printf("Numero de cadastro: %d\n", usuario.num);
                printf("Data de nascimento: %d/%d/%d\n", usuario.data_nasc[0], usuario.data_nasc[1], usuario.data_nasc[2]);
                printf("Rua %s numero %d, %s", usuario.rua, usuario.num_casa, usuario.complemento);
                printf("Bairro: %s\n", usuario.bairro);
                printf("Cidade: %s\n", usuario.cidade);
                printf("Estado: %s\n", usuario.estado);
                fclose(Arq);

                imprimir_emprestimo(usuario.nome, url_empr);
                break;
            }else{
                fseek(Arq, TAM_CAD-50, SEEK_CUR); //pula para o proximo cadastro
                if(fread(&flag, 1, 1, Arq) == 0) //verifica fim do arquivo
                    break;
                else
                    fseek(Arq, -1, SEEK_CUR); //volta 1 byte
            }
        }
    }
}
int verifica_id(char *url_cad){
    int num;
    char flag;
    FILE *Arq=fopen(url_cad,"rb");
    if (fread(&flag,1,1,Arq)==0)
        return 0;
    fseek(Arq,-(TAM_CAD-50*sizeof(char)),SEEK_END);
    fread(&num,sizeof(int),1,Arq);
    return (num - 100000 + 1);
}


int main(){
	char *url_cad = "Cadastro.bin";
	char *url_empr = "Emprestimo.bin";
	char temp[5],nome[50];
    int i = 0, endereco, flag, opcao;
    int dia, mes, ano;

    t_cadastro usuario;

    FILE *Arq = fopen(url_cad, "ab");
    if (Arq == NULL) {
        printf("Erro, nao foi possivel abrir o arquivo de cadastro\n");
        return -1;
    }
    fclose(Arq);

    Arq = fopen(url_empr, "ab");
    if (Arq == NULL) {
        printf("Erro, nao foi possivel abrir o arquivo de emprestimo\n");
        return -1;
    }
    fclose(Arq);

    i = verifica_id(url_cad);

    //Data do dia
    printf("Data de hoje\n");
    printf("Dia: ");
    fflush(stdin);
    fgets(temp,5,stdin);
    dia = atoi(temp);
    printf("Mes: ");
    fflush(stdin);
    fgets(temp,5,stdin);
    mes = atoi(temp);
    printf("Ano: ");
    fflush(stdin);
    fgets(temp,5,stdin);
    ano = atoi(temp);

    while(1){
        opcao = menu();
        switch(opcao){
            case 1: //cadastro
                flag = cadastro(&usuario, i, url_cad);
                if(flag == 0){ //nao tinha cadastro mas fez
                    salva_cadastro(&usuario, url_cad);
                    i++;
                }
                break;

            case 2: //emprestimo
                printf("Nome: ");
                fflush(stdin);
    
                fgets(nome, 50, stdin);
                endereco = verifica_cad(&usuario, i, nome, url_cad);
                if(endereco !=0 && endereco !=(-1)){ //tem cadastro
                    if(confirma_senha(endereco, url_cad,NULL, 1)){
                        if(verifica_emprestimo(nome, url_empr)==0){ //nao fez emprestimo ainda
                            novo_emprestimo(usuario, endereco, dia, mes, ano, url_cad, url_empr);
                            break;
                        }else if(verifica_emprestimo(nome, url_empr)==1){ //j fez emprestimo
                            altera_emprestimo(dia, mes, ano, url_empr, url_cad, 1);
                            break;
                        }
                        break;
                    }else//saiu sem preencher a senha
                        break;
                }else if(endereco == 0){ //nao tem cadastro
                    printf("Nao existe cadastro nesse nome\n");
                    break;
                }

            case 3: //busca historico
                imprimir_cadastro(url_cad, url_empr);
                break;
            case 4: //devolucao
                altera_emprestimo(dia, mes, ano, url_empr, url_cad, 2);
                break;
            case 5:
                return 0;
            default:
                continue;
        }
        continue;
    }
}