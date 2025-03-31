#include "list-generation.c"
#include "merge.c"
#include "sort.c"
#include "search.c"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void showMenuOptions();
void getMenuOption(int* option);
void handleMenuOptions(int option);
void showList(StudentList list);

StudentList studentsA, studentsB, mergedStudents;

void main() {
    srand(time(NULL));

    printf("Estrutura de Dados I\n");
    printf("Mesclagem de Listas Ordenadas\n\n");

    studentsA = createRandomStudentList(5);
    studentsB = createRandomStudentList(5);

    quick_sort(studentsA.items, 0, studentsA.size - 1);
    quick_sort(studentsB.items, 0, studentsB.size - 1);

    mergedStudents = mergeSortedLists(studentsA, studentsB);

    int option;

    while (option != -1)
    {
        system("clear");
        showMenuOptions();
        getMenuOption(&option);
        printf("\n");
        handleMenuOptions(option);
        if (option != -1) getchar();
    }
}

void showMenuOptions() {
    printf("Menu\n");
    printf("Opção 1 - Exibir as listas originais\n");
    printf("Opção 2 - Exibir a lista mesclada\n");
    printf("Opção 3 - Pesquisar por nome na lista mesclada\n");
    printf("Digite -1 para encerrar o programa\n");
}

void getMenuOption(int* option) {
    printf("Digite a opção desejada: ");
    scanf("%d", option);
    getchar();
}

char* readString() {
    char* string = (char*) malloc(100 * sizeof(char));
    scanf("%[^\n]", string);
    getchar();
    return string;
}

void handleMenuOptions(int option) {
    switch (option)
    {
        case 1:
            printf("Lista A:\n");
            showList(studentsA);
            printf("\n");
            printf("Lista B:\n");
            showList(studentsB);
            break;

        case 2:
            printf("Lista Mesclada:\n");
            showList(mergedStudents);
            break;

        case 3:
            printf("Digite o nome a ser pesquisado: ");
            char* name = readString();
            StudentList result = searchAllOcurrences(mergedStudents, name);
            showList(result);
            break;

        case -1:
            printf("Saindo...\n");
            break;
        
        default:
            printf("Opção inválida\n");
            break;
    }
}

void showList(StudentList list) {
    for(int i = 0; i < list.size; i++) {
        printf("Aluno [%d]:\n", i);
        printf("\t-> Nome: %s\n", list.items[i].name);
        printf("\t-> Telefone: %s\n", list.items[i].phone);
        printf("\t-> Idade: %d\n", list.items[i].age);
    }
}