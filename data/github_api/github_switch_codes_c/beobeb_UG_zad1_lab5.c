#include <stdio.h>

void odejmowanie (int x,int y){
printf("%d",x-y);
}
void dodawanie (int x,int y){
printf("%d",x+y);
}
void mnozenie (int x,int y){
printf("%d",x*y);
}
void dzielenie (int x,int y){
printf("%d",x/y);
}


int main(){

    int x,y;

    printf("Podaj dwie liczby: \n");
    scanf("%d %d",&x,&y);
    int w;
    printf("Podaj rodzaj działania: \n1-odejmowanie \n2-dodawanie \n3-mnozenie \n4-dzielenie\n");
    scanf("%d",&w);
    switch(w){
        case 1:
    printf("wynik odejmowania: ");
    odejmowanie(x,y);
    break;
        case 2:
    printf("wynik dodawania: ");
    dodawanie(x,y);
    break;
        case 3:
    printf("wynik mnozenia: ");
    mnozenie(x,y);
    break;
        case 4:
    printf("wynik dzielenia: ");
    dzielenie(x,y);
    break;
    }

    return 0;
}
