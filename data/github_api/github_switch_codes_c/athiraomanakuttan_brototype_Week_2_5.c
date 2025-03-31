#include <stdio.h>
#include <stdlib.h>
#include <math.h>
char GradeCheck(float mark);
void main()
{
    float mark;
    printf("Please enter your mark\n");
    scanf("%f", &mark);
    GradeCheck(mark);
}
char GradeCheck(float mark)
{

    const int grade = round(mark) / 10;

    switch (grade)
    {

    case 9:
        printf("Grade A");
        break;
    case 8:
        printf("Grade B");
        break;
    case 7:
        printf("Grade C");
        break;
    case 6:
        printf("Grade D");
        break;
    case 5:
        printf("Grade E");
        break;
    case 4:
        printf("Sorry yu failed");
        break;
    case 3:
        printf("Sorry yu failed");
        break;
    case 2:
        printf("Sorry yu failed");
        break;
    case 1:
        printf("Sorry yu failed");
        break;
    case 0:
        printf("Sorry yu failed");
        break;
    default:
        printf("Wrong input.");
        break;
    }
}