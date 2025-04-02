#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sorting.h"

int main(int argc, char *argv[]) {
  int *a, N, i, j, new_number;

  char *sorting = argv[1];
  int checker = typeCheck(sorting);

  N = argc - 2;
  a = (int*)malloc(sizeof(int) * N);
  for (i = 0; i < N; i++)
  {
    a[i] = atoi(argv[i+2]);
  }


  display(a, N);
  switch (checker)
  {
  case 1:
    bubbleSort(a, N);
    break;
  case 2:
    selectionSort(a, N);
    break;
  case 3:
    insertion(a, N);
    break;
  
  default:
    break;
  }
  printf("\n");
  display(a, N);

 return 0;
}