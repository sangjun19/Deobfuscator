#include <stdio.h>

/* A gas station wants to determine which of their products is 
the preference of their customers. Write a program to read the 
type of fuel supplied (coded as follows: 1. Alcohol 2. Gasoline 3. 
Diesel 4. End). If you enter an invalid code (outside the range of 
1 to 4) a new code must be requested. The program will end when 
the inserted code is the number 4. */

int main() {
    
    int num;
    int alcool = 0, gasolina = 0, diesel = 0;

    do {

        scanf("%d", &num);

        switch (num) {

        case 1:
            alcool++;
            break;

        case 2:
            gasolina++;
            break;

        case 3:
            diesel++;
            break;

        default:
            break;
        }

    } while (num != 4);

    printf("MUITO OBRIGADO\n");
    printf("Alcool: %d\n", alcool);
    printf("Gasolina: %d\n", gasolina);
    printf("Diesel: %d\n", diesel);

    return 0;
}
