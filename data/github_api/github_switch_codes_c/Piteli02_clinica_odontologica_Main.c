/*
PROJETO FINAL DISCIPLINA SI200-PROGRAMAÇÃO 2

UNIVERSIDADE ESTADUAL DE CAMPINAS - FACULDADE DE TECNOLOGIA (UNICAMP-FT)

Link para o github - https://github.com/Piteli02/clinica_odontologica

Codigo desenvolvido por::
	-Andre Gomes de Lima Braga 234444
	-Caio Gomes Piteli 234451
	-Eduardo Longhi 237468
	-Henrique Bexiga Eulalio 255002
*/

/*
O usuario e senha pedidos no começo do sistema devem ser setados em um arquivo chamado
"login.txt", o qual deve estar armazenado na mesma pasta que o "Main.c"
No arquivo deve conter o usuario, seguido por um espaço e uma senha

Exemplo:
adm 123
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

//Estrutura da hora
typedef struct {
	int hora;
	int min;
} Hora;

//Estrutura da Data
typedef struct {
	int ano;
	int mes;
	int dia;
} Data;

//Estrutura dos pacientes
typedef struct {
	char nome[50];
	char cpf[15];
	char telefone[15];
} Paciente;

//Estrutura da consulta
typedef struct {
	int dia;
	int mes;
	int ano;
	int hora; 
	int minuto;
	char cpf[15];
} Consulta;

//Variaveis com o usuario e senha do txt
char usuario_setado[6], senha_setada[6];

//Recebo os dados de login
void fazer_login();

//Opcoes de escolha para o que fazer no software
void menu_principal();

//Sequencia depois de ter decidido o que sera feito no sofware
void escolha_menu(int operador);

void cadastrar_clientes();

int marcar_consulta(Consulta **c, int quantidade, int tamanho);

void salvar(Consulta **c, int quantidade, char arq[]);

void alterar_consulta();

//Procuro e exibo todas as consultas daquele dia
void achar_consultas_dia();

//Procuro e exibo todas as consultas daquele cpf
void achar_consultas_cpf();

void excluir_consulta();

//Vejo se o usuario digitado eh o mesmo que o usuario presente no arquivo "file_login"
void verificar_login(char usuario_entrada[20], char senha_entrada[20]);

//Verifico se a entrada do cpf esta correta
void verificar_cpf(char cpf[15]);

//Verifica se a entrada do telefone esta correta
void verificar_tel(char tel[15]);

//Verifica se o paciente que vai marcar a consulta ja esta cadastrado
void verificar_paciente_cadastrado(char cpf[15]);


int main() {
	int operador = 0;

	//Seleciono o arquivo com o login e senha_setada	
	FILE *file_login;
	file_login = fopen("login.txt", "r"); // "r" - porque quero pegar informacao do arquivo
	//Caso o usuario n tenha colocado o txt na mesma pasta que o executavel
	if(file_login == NULL){
		system("cls"); //limpar terminal windows
		printf("\nERRO: Insira o arquivo 'login.txt' na mesma pasta que o executavel!!\n");
		exit(0);
	}

	fscanf(file_login, "%s %s", usuario_setado, senha_setada); //recebo a senha e o usuario do arquivo txt
	fclose(file_login);

	fazer_login();

	menu_principal();

	return 0;
}

void fazer_login() {
	system("cls"); //limpar terminal windows

	char usuario_entrada[20], senha_entrada[20];

	printf("--------------------BEM VINDO(A)--------------------\n\n");
	printf("LOGIN:\n");

	printf("Digite o nome de usuario: ");
	fgets(usuario_entrada, 20, stdin);
	usuario_entrada[strcspn(usuario_entrada, "\n")] = 0; //tirando o "\n" da string
	printf("Digite a sua senha: ");
	fgets(senha_entrada, 20, stdin);
	senha_entrada[strcspn(senha_entrada, "\n")] = 0; //tirando o "\n" da string

	verificar_login(usuario_entrada, senha_entrada);
}

void menu_principal() {
	int escolha_menu_func = 0;
	int controlador_menu_principal = 0;

	system("cls"); //limpar terminal windows
	
	printf("--------------------MENU PRINCIPAL--------------------\n");

	//Usuario escolhe o que ele vai fazer no software a partir de um numero
	while(escolha_menu_func == 0 || escolha_menu_func != 1 && escolha_menu_func != 2 && escolha_menu_func != 3 && escolha_menu_func != 4 && escolha_menu_func != 5  && escolha_menu_func != 6 && escolha_menu_func != 7){
		if(controlador_menu_principal!=0) {
			printf("--------------------OPCAO INVALIDA, TENTE NOVAMENTE--------------------\n");
		}
		
		printf("\nAPERTE:\n");
		printf("1 para CADASTRAR NOVO CLIENTE\n");
		printf("2 para AGENDAR NOVA CONSULTA\n");
		printf("3 para ALTERAR CONSULTA EXISTENTE\n");
		printf("4 para VISUALIZAR CONSULTAS DO DIA\n");
		printf("5 para VISUALIZAR CONSULTAS DO PACIENTE\n");
		printf("6 para EXCLUIR CONSULTA\n");
		printf("7 para ENCERRAR O SISTEMA \n");
		printf("Escolha: ");
		scanf("%d", &escolha_menu_func);

		controlador_menu_principal++;
		system("cls"); //limpar terminal windows
	}
	
	escolha_menu(escolha_menu_func);
}

void escolha_menu(int operador) {
	Consulta *agenda[100];   
	// a agenda sera um vetor de Consultas
	char arq[] = {"agenda.txt"};
	int tamanho = 100, quantidade = 0;
	
	switch(operador) {
		case 1:
			cadastrar_clientes();
			break;
		case 2:
			quantidade += marcar_consulta(agenda, quantidade, tamanho);
			salvar(agenda, quantidade, arq);
			menu_principal();
			break;
		case 3:
			alterar_consulta();
			menu_principal();
			break;
		case 4:
			achar_consultas_dia();
			menu_principal();
			break;
		case 5:
			achar_consultas_cpf();
			menu_principal();
			break;
		case 6:
			excluir_consulta();
			menu_principal();
			break;
		case 7:
			exit(0);
	}
}

void cadastrar_clientes() {

	//Seleciono o arquivo com o login e senha_setada	
	FILE *file_pacientes;
	file_pacientes = fopen("pacientes.txt", "a"); // "a" - porque quero "append" informa��es no arquivo

	//Caso o usuario n tenha colocado o txt na mesma pasta que o executavel
	if(file_pacientes == NULL) {
		system("cls"); //limpar terminal windows
		printf("\nERRO: Insira o arquivo 'pacientes.txt' na mesma pasta que o executavel!!\n");
		exit (0);
	}

	Paciente paciente;

	system("cls");
	printf("--------------------CADASTRO NOVO PACIENTE--------------------\n");

	fflush(stdin);
	printf("Insira o nome do paciente a ser cadastrado: ");
	fgets(paciente.nome, 50, stdin);
	paciente.nome[strcspn(paciente.nome, "\n")] = 0; //tirando o "\n" da string

	fflush(stdin);
	printf("Insira o cpf do paciente no modelo ***.***.***.** : ");
	fgets(paciente.cpf,15,stdin);
	paciente.cpf[strcspn(paciente.cpf, "\n")] = 0; //tirando o "\n" da string		
		
	verificar_cpf(paciente.cpf);

	fflush(stdin);
	printf("Insira o telefone do paciente no modelo (**)*****-****: ");
	fgets(paciente.telefone, 15, stdin);
	paciente.telefone[strcspn(paciente.telefone, "\n")] = 0; //tirando o "\n" da string
		
	verificar_tel(paciente.telefone);
	
	fprintf(file_pacientes, "%s %s %s\n", paciente.nome, paciente.cpf, paciente.telefone); //colocando as informa��es no arquivo

	system("cls");
	fclose(file_pacientes);
		
	fflush(stdin);
	
	menu_principal();
}

int marcar_consulta(Consulta **c, int quantidade, int tamanho) {
    if(quantidade < tamanho) {
        Consulta *novo = malloc(sizeof(Consulta));
		printf("--------------------AGENDAR CONSULTA--------------------\n");
        printf("\nDigite o CPF do paciente: ");
        scanf("%s", &novo->cpf);
        verificar_cpf(novo->cpf);
		verificar_paciente_cadastrado(novo->cpf);
        printf("\nDigite a data da consulta dd/mm/aaaa: ");
        scanf("%d/%d/%d", &novo->dia, &novo->mes, &novo->ano);
        printf("\nDigite o horario da consulta hh:mm: ");
        scanf("%d:%d", &novo->hora, &novo->minuto);
        getchar();
        c[quantidade] = novo;
        return 1;
    } else {
        printf("\n\tImpossivel novo cadastro. Vetor cheio!\n");
        return 0;
    }
}

void salvar(Consulta **c, int quantidade, char arq[]) {
	FILE *file = fopen(arq, "a");
	int i;
	
	if(file) {
	    for(i = 0; i < quantidade; i++) {
	    	fputs(c[i]->cpf, file);
	        fprintf(file, " %d/%d/%d ", c[i]->dia, c[i]->mes, c[i]->ano);
	        fprintf(file, "%d:%d\n", c[i]->hora, c[i]->minuto);
	    }
	    fclose(file);
	} else {
	    printf("\n\tERRO: Nao foi possivel ABRIR/CRIAR o arquivo!\n");
	}
}

/*
---------------------------------------------------------------------------------------------------
Essa função vai ser responsável por receber uma consulta a qual deverá ser alterada
Todas as outras consultas serão escritas em um arquivo chamado "temp_agenda"
Desse moodo, no final, o "temp_agenda" vai posuir todas as consultas, menos a que eu quero alterar
Eu excluo o arquivo que tinha a consulta que eu selecionei para alterar
Faço o arquivo que já possui a exclusão passar a ser o meu "agenda" de verdade
Printo a nova consulta (que substitui a excluida) no arquivo
*/
void alterar_consulta() {
	system("cls");
	
	//variaveis para chamar a função de agendar consulta
	Consulta *agenda[100];   
	char arq[] = {"agenda.txt"};
	int tamanho = 100, quantidade = 0;

	int vezes_errado = 0; //variavel que fala se o cara errou a digitação da consulta que ele queria
	bool consulta_encontrada = false; 
	char consulta_procurar[100];
	char consulta_do_arquivo[100];
	bool continuar_lendo_arq = true;

	//abrindo arquivos
	FILE *file_agenda;
	file_agenda = fopen("agenda.txt", "r"); // "r" - porque quero pegar informacao do arquivo
	FILE *temp_file_agenda;
	temp_file_agenda = fopen("temp_agenda.txt","w");

	//apontando erro caso os arquivos não sejam abertos corretamente
	if(file_agenda == NULL || temp_file_agenda == NULL) {
		system("cls");
		printf("ERRO: Nao foi possivel abrir o arquivo agenda.txt ou temp_agenda.txt!!");
		exit(1);
	} 

	//percorrer linha por linha do txt, procurando pela consulta inserida
	while(consulta_encontrada == false) {
		system("cls");

		//se tiver inserido a consulta a ser procurada erroneamente, ele printa que escreveu errado e manda escerver dnv
		if(vezes_errado == 0) {
			printf("Nessa tela, todas as consultas do cpf a ser procurado serao exibidas, COPIE A CONSULTA A SER ALTERADA");
			printf("\nAperte uma tecla para continuar...");
			getch();
			achar_consultas_cpf();
			system("cls");
			printf("Insira a consulta a ser alterada: ");
		} else {
			printf("**Consulta nao encontrada\n\n");
			printf("--------------------TENTE NOVAMENTE--------------------\n");
			printf("FORMATO: ***.***.***.** dd/mm/aaaa hr:min\n");

			//devo fechar os arquivos e abrir de novo, para ele recomeçar a passar pelas linhas, caso contrario ele tenta continuar a partir do final do arquivo
			fclose(file_agenda);
			fclose(temp_file_agenda);
			
			//abrindo os arquivos após ter fechado
			file_agenda = fopen("agenda.txt", "r"); // "r" - porque quero pegar informacao do arquivo
			temp_file_agenda = fopen("temp_agenda.txt","w");
		}

		//recebendo a consulta a ser procurada, para ser alterada
		fflush(stdin);
		fgets(consulta_procurar,100,stdin);

		//de fato, achando a consulta a ser procurada
		do {
			fgets(consulta_do_arquivo,100,file_agenda);
	
			if(feof(file_agenda)) {
				continuar_lendo_arq = false;
			} else {
				if(strcmp(consulta_procurar,consulta_do_arquivo)==0) {
					consulta_encontrada = true;
					continue;
				} else {
					fputs(consulta_do_arquivo, temp_file_agenda);
				}
			}
		} while(continuar_lendo_arq);
	
	vezes_errado++;
	continuar_lendo_arq=true;
	
	}

	fclose(file_agenda);
	fclose(temp_file_agenda);

	//eu exluo a agenda, e renomeio o arquivo que era temporario, assim, ele passa a ser o de verdade
	remove("agenda.txt");
	rename("temp_agenda.txt", "agenda.txt");

	system("cls");

	printf("Consulta excluida com sucesso! Agora, vamos marcar a consulta que substituira a anterior!");
	printf("\nAperte uma tecla para continuar...");
	getch();
	system("cls");
	//chamando a  função pra marcar a consulta a qual substituira a que eu acabei de excluir
	quantidade += marcar_consulta(agenda, quantidade, tamanho);
	salvar(agenda, quantidade, arq);

	//voltar para o menu principal
	menu_principal();
}

void achar_consultas_dia() {
	char data[10];

	system("cls");

	printf("--------------------DATA A SER PROCURADA--------------------\n");
	printf("Insira a data:\n");
	printf("FORMATO - dd/mm/aaaa: ");
	fflush(stdin);
	fgets(data,10,stdin);

	system("cls");
	printf("--------------------CONSULTAS NESSE DIA--------------------\n");
	FILE *file_agenda;
	file_agenda = fopen("agenda.txt", "r");

	char consulta_do_arquivo[50];

	if(file_agenda == NULL) {
		system("cls");
		printf("ERRO: Nao foi possivel abrir o arquivo agenda.txt!!");
		exit(1);
	}

	while(!feof(file_agenda)) {
		fgets(consulta_do_arquivo,50,file_agenda);
		if(strstr(consulta_do_arquivo,data) != NULL) {
			printf("%s", consulta_do_arquivo);
		}
	}

	fclose(file_agenda);
	fflush(stdin);
	printf("Clique alguma coisa para continuar...");
	getch();
}

void achar_consultas_cpf() {
	char cpf[15];
	
	system("cls");

	printf("--------------------CPF A SER PROCURADO--------------------\n");
	printf("Insira o cpf: ");
	fflush(stdin);
	fgets(cpf,15,stdin);
	verificar_cpf(cpf);
	verificar_paciente_cadastrado(cpf);

	system("cls");
	printf("--------------------CONSULTAS DESSE CPF--------------------\n");
	FILE *file_agenda;
	file_agenda = fopen("agenda.txt", "r");

	char consulta_do_arquivo[50];

	if(file_agenda == NULL) {
		system("cls");
		printf("ERRO: Nao foi possivel abrir o arqiivo agenda.txt!!");
		exit(1);
	}

	while(!feof(file_agenda)) {
		fgets(consulta_do_arquivo,50,file_agenda);
		if(strstr(consulta_do_arquivo,cpf) != NULL){
			printf("%s", consulta_do_arquivo);
		}
	}

	fclose(file_agenda);
	
	printf("Clique alguma coisa para continuar...");
	getch();
}

void excluir_consulta() {
	system("cls");

	int vezes_errado = 0; //variavel que fala se o cara errou a digitação da consulta que ele queria
	bool consulta_encontrada = false; 
	char consulta_procurar[100];
	char consulta_do_arquivo[100];
	bool continuar_lendo_arq = true;

	//abrindo arquivos
	FILE *file_agenda;
	file_agenda = fopen("agenda.txt", "r"); // "r" - porque quero pegar informacao do arquivo
	FILE *temp_file_agenda;
	temp_file_agenda = fopen("temp_agenda.txt","w");

	//apontando erro caso os arquivos não sejam abertos corretamente
	if(file_agenda == NULL || temp_file_agenda == NULL) {
		system("cls");
		printf("ERRO: Nao foi possivel abrir o arquivo agenda.txt ou temp_agenda.txt!!");
		exit(1);
	} 

	//percorrer linha por linha do txt, procurando pela consulta inserida
	while(consulta_encontrada == false ){
		system("cls");

		//se tiver inserido a consulta a ser procurada erroneamente, ele printa que escreveu errado e manda escerver dnv
		if(vezes_errado==0) {
			printf("Nessa tela, todas as consultas do cpf a ser procurado serao exibidas, COPIE A CONSULTA A SER EXCLUIDA");
			printf("\nAperte uma tecla para continuar...");
			getch();
			achar_consultas_cpf();
			system("cls");
			printf("Insira a consulta a ser EXCLUIDA: ");
		} else {
			printf("**Consulta nao encontrada\n\n");
			printf("--------------------TENTE NOVAMENTE--------------------\n");
			printf("FORMATO: ***.***.***.** dd/mm/aaaa hr:min\n");

			//devo fechar os arquivos e abrir de novo, para ele recomeçar a passar pelas linhas, caso contrario ele tenta continuar a partir do final do arquivo
			fclose(file_agenda);
			fclose(temp_file_agenda);
			
			//abrindo os arquivos após ter fechado
			file_agenda = fopen("agenda.txt", "r"); // "r" - porque quero pegar informacao do arquivo
			temp_file_agenda = fopen("temp_agenda.txt","w");
		}

		//recebendo a consulta a ser procurada, para ser alterada
		fflush(stdin);
		fgets(consulta_procurar,100,stdin);

		//de fato, achando a consulta a ser procurada
		do {
			fgets(consulta_do_arquivo,100,file_agenda);
	
			if(feof(file_agenda)) {
				continuar_lendo_arq = false;
			} else {
				if(strcmp(consulta_procurar,consulta_do_arquivo) == 0) {
					consulta_encontrada = true;
					continue;
				} else {
					fputs(consulta_do_arquivo, temp_file_agenda);
				}
			}
		} while(continuar_lendo_arq);

		vezes_errado++;
		continuar_lendo_arq=true;
	
	}

	fclose(file_agenda);
	fclose(temp_file_agenda);

	//eu exluo a agenda, e renomeio o arquivo que era temporario, assim, ele passa a ser o de verdade
	remove("agenda.txt");
	rename("temp_agenda.txt", "agenda.txt");

	system("cls");

	printf("Consulta excluida com sucesso!");
	printf("\nAperte uma tecla para continuar...");
	getch();
	system("cls");
}

void verificar_login(char usuario_entrada[20], char senha_entrada[20]) {
	int verificar_usuario, verificar_senha;
	
	verificar_usuario = strcmp(usuario_setado, usuario_entrada);
	verificar_senha = strcmp(senha_setada, senha_entrada);
	
	while(verificar_senha != 0 || verificar_usuario != 0) {
		system("cls"); //limpar terminal windows

		printf("--------------------LOGIN OU SENHA INCORRETO--------------------\n");

		printf("Digite o nome de usuario novamente: ");
		fgets(usuario_entrada,20,stdin);
		usuario_entrada[strcspn(usuario_entrada, "\n")] = 0; //tirando o "\n" da string
		printf("Digite a sua senha novamente: ");
		fgets(senha_entrada,20,stdin);
		senha_entrada[strcspn(senha_entrada, "\n")] = 0; //tirando o "\n" da string

		verificar_usuario = strcmp(usuario_setado, usuario_entrada);
		verificar_senha = strcmp(senha_setada, senha_entrada);
	}
}

void verificar_cpf(char cpf[15]) {
	int i, cont = 0;
	
	do {	
		cont = 0;
		for(i = 0; i < 15; i++) {
			//checando se possui alguma letra (percorrendo caracter por caracter) de acordo com a tabela ASCII
			if(cpf[i] > 58 ) {
				cont++;
				printf("CPF invalido, digite apenas numeros e siga o modelo: ***.***.***.**: ");
				fflush(stdin);
				fgets(cpf,15,stdin);
				cpf[strcspn(cpf, "\n")] = 0; //tirando o "\n" da string	
				system("cls");
			}
		}	
		
		if(cpf[3] != '.' || cpf[7] != '.' || cpf[11] != '.') {
			cont++;
				printf("CPF invalido, digite apenas numeros e siga o modelo: ***.***.***.**: ");
				fflush(stdin);
				fgets(cpf,15,stdin);
				cpf[strcspn(cpf, "\n")] = 0; //tirando o "\n" da string	
				system("cls");
		}
		
	} while(cont > 0);
}

void verificar_tel(char tel[15]) {
	int i, cont = 0;
	
	do {	
		cont = 0;
		for(i = 0; i < 15; i++) {
			//checando se possui alguma letra (percorrendo caracter por caracter) de acordo com a tabela ASCII
			if(tel[i] > 58 ) {
				cont++;
				printf("Telefone invalido, digite apenas numeros e siga o modelo (**)*****-****: ");
				fflush(stdin);
				fgets(tel,15,stdin);
				tel[strcspn(tel, "\n")] = 0; //tirando o "\n" da string	
				system("cls");
			}
		}
		
		if(tel[0] != '(' || tel[3] != ')' || tel[9] != '-') {
			cont++;
			printf("Telefone invalido, digite apenas numeros e siga o modelo (**)*****-****: ");
			fflush(stdin);
			fgets(tel,15,stdin);
			tel[strcspn(tel, "\n")] = 0; //tirando o "\n" da string	
			system("cls");
		}
	} while(cont > 0);
}
	
void verificar_paciente_cadastrado(char cpf[15]) {
	int paciente_existe = 0;
	int controlador = 0;

	FILE *file_paciente;
	file_paciente = fopen("pacientes.txt", "r");

	char consulta_do_arquivo[100];

	if(file_paciente == NULL) {
		printf("ERRO: Nao foi possivel abri o arquivo pacientes.txt");
		exit(1);
	}

	do {
		if(controlador!=0) {
			system("cls");
			printf("--------------------PACIENTE NAO EXISTE--------------------\n");
			printf("Insira o cpf novamente:");
			fflush(stdin);
			fgets(cpf,15,stdin);
			verificar_cpf(cpf);
			fclose(file_paciente);
			FILE *file_paciente;
			file_paciente = fopen("pacientes.txt", "r");
		}
		
		while(!feof(file_paciente)) {
			fgets(consulta_do_arquivo,100,file_paciente);
			if(strstr(consulta_do_arquivo,cpf) != NULL){
				paciente_existe = 1;
			}
		}
		
		controlador++;
		
	} while(paciente_existe==0);

	fclose(file_paciente);

}