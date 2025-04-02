//
// Created by Doomf on 7/26/2023.
//
#include "stdio.h"
#include "ctype.h"
char* grader()
{
    int grade;
    scanf("%c", &grade);
    grade = toupper(grade);
    switch (grade) 
    {
        case 65: return "Genius";
        case 66: return "Good";
        case 67: return "Try Harder";
        case 68: return "Very Bad";
        case 70: return "Fail";
        default: return "Invalid Input";
    }
}

int main()
{
    printf("%s", grader());
    return 0;
}