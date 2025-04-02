#include <stdio.h>
#include <string.h>
#define N_MAX 100000

int led(char str){
  int led;
  switch (str){
  case '1':
    led = 2;
    break;
  case '2':
    led = 5;
    break;
  case '3':
    led = 5;
    break;
  case '5':
    led = 5;
    break;
  case '4':
    led = 4;
    break;
  case '6':
    led = 6;
    break;
  case '9':
    led = 6;
    break;
  case '0':
    led = 6;
    break;
  case '7':
    led = 3;
    break;
  case '8':
    led = 7;
    break;
  }
  return led;
}
void verifica(char str[]){
  int tam = strlen(str);
  int i,j, soma = 0;
  for(i=0;i<tam;i++){
    soma += led(str[i]);
  }
  printf("%d leds\n",soma);
}

int main(void){

  int entradas;

  scanf("%d",&entradas);

  while(entradas--){
    char str[N_MAX];
    scanf("%s", str);
    verifica(str);
  }
  return 0;
}