#include <stdio.h>

int main()
{
    char strefa;
    printf("Wybierz strefe [A] [B] [C] [D]: ");
    scanf("%c", &strefa);

    switch (strefa)
    {
    case 'A':
        printf("Koszt postoju 2 zl/h");
        break;
    case 'B':
        printf("Koszt postoju 4 zl/h");
        break;
    case 'C':
        printf("Koszt postoju 6 zl/h");
        break;
    case 'D':
        printf("Koszt postoju 8 zl/h");
        break;
    
    default:
        break;
    }



    return (0);
}