#include <stdio.h>

int main() {
    
    float quilometragem, total;
    char opcao;
    printf("informe o percurso a ser percorrido em KM do carro: ");
    scanf("%f",&quilometragem);
    printf("\nInforme o tipo de carro: \nA = 8km/litro \nB = 9km/litro \nC = 12km/litro\n");
    scanf("%s",&opcao);
    
    switch (opcao) {
    case 'a' :
        total = quilometragem * 8;
        printf("O consumo estimado é %.0fkm/l",total);
        break;
    case 'b' :
        total = quilometragem * 9;
        printf("O consumo estimado é %.0fkm/l",total);
        break;
    case 'c' :
        total = quilometragem * 12;
        printf("O consumo estimado é %.0fkm/l",total);
        break;
    default : 
        printf("Carro invalido");
    }    
    return 0;
}
