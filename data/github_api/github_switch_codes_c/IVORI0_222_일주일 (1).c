#include <stdio.h>

int main() {
    int day;

    printf("숫자(1-7)를 입력하세요: ");
    scanf("%d", &day);

    switch(day) {
        case 1:
            printf("일요일\n");
            break;
        case 2:
            printf("월요일\n");
            break;
        case 3:
            printf("화요일\n");
            break;
        case 4:
            printf("수요일\n");
            break;
        case 5:
            printf("목요일\n");
            break;
        case 6:
            printf("금요일\n");
            break;
        case 7:
            printf("토요일\n");
            break;
        default:
            printf("잘못된 입력입니다. 1부터 7까지의 숫자를 입력하세요.\n");
    }

    return 0;
}