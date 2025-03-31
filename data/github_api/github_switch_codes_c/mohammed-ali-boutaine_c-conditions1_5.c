#include <stdio.h>

int main() {

    int anne,option;
    printf("entre un anne:");
    scanf("%d",&anne);

    printf("chose un option:\n\n");
    printf("1- Mois\n");
    printf("2- Jours\n");
    printf("3- Heures\n");
    printf("4- Minutes\n");
    printf("5- Secondes\n");

    printf("entre votre choix:");
    scanf("%d",&option);

    switch(option){
        case 1:
            printf("%d an = %d mois",anne,anne*12);
            break;
        case 2:
            printf("%d an = %d jours",anne,anne*365);
            break;
        case 3:
            printf("%d an = %d h",anne,anne*365*24);
            break;
        case 4:
            printf("%d an = %d min",anne,anne*365*24*60);
            break;
        case 5:
            printf("%d an = %d s",anne,anne*365*24*60*60);
            break;
        default:
            printf("choix valide\n");
    }




    return 0;
}