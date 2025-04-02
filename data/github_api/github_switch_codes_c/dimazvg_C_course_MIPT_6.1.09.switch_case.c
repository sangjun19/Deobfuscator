/* пример использования конструкции switch-case
"Склоняем" коров по падежам.
*/
#include <stdio.h>

int main()
{
    int n;  // количество коров
    scanf("%d", &n);

    switch (n % 10) {
        case 1:
            printf("%d корова", n);
            break;
        case 2:
        case 4:     // константы в любом порядке
        case 3:
            printf("%d коровы", n);
            break;
        default:
            printf("%d коров", n);
    }
    // сюда переходим после срабатывания break

    return 0;
}