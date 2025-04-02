#include<stdio.h>
int main () {
    int n[3] = {73,95,82};  //배열 선언 
    int i, sum = 0;

    for (i = 0 ; i< 3; i++) {  //i가 3보다 작으면 sum+=n[i] 실행 이후 i++실행 (i를 1씩 증가)
         sum +=n[i];
    }

    switch (sum/30 )  //Sum/ 30 = 250을 30으로 나눴을 때 몫 = 8 
    { 
    case 10:
    case 9:
    printf("A");
    case 8:
    printf("B");  //Case8에 해당되는 코드 실행 "B" 출력 이후 break 없어서 default까지 실행
    case 7:
    case 6:
    printf("C");

    default:
        printf("D");
    }
}