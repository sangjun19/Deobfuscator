// Repository: Melearner777/Guvi-Ds-with-C
// File: 8.c

/*Enumeration (enum) in C
Problem Statement:
Create a C program that uses an enumeration to represent the days of the week. The program should take a list of integers from the user, convert them to the corresponding days of the week using the enum, and then print the days in reverse order.

Description:
You will define an enumeration for the days of the week starting with Sunday as 0 and ending with Saturday as 6. The program should read a list of integers from the user, map them to the corresponding days, and print the days in reverse order.

Input Format:

An integer
𝑛
n representing the number of days.
A list of
𝑛
n integers representing the days of the week.
Output Format:

The days of the week in reverse order.

Private Testcase Input 1:
5
0 1 2 3 4
Private Testcase Output 1:
Days in reverse order:
Thursday Wednesday Tuesday Monday Sunday*/


#include <stdio.h>

typedef enum {
    SUNDAY,
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY
} DayOfWeek;

const char* getDayOfWeekString(DayOfWeek day) {
    switch(day) {
        case SUNDAY:    return "Sunday";
        case MONDAY:    return "Monday";
        case TUESDAY:   return "Tuesday";
        case WEDNESDAY: return "Wednesday";
        case THURSDAY:  return "Thursday";
        case FRIDAY:    return "Friday";
        case SATURDAY:  return "Saturday";
        default:        return "Invalid";
    }//717580
}

int main() {
    int n;

    scanf("%d", &n);

    int days[n];

    for (int i = 0; i < n; i++) {
        scanf("%d", &days[i]);
    }

    printf("Days in reverse order:\n");
    for (int i = n - 1; i >= 0; i--) {
        printf("%s", getDayOfWeekString(days[i]));
        if (i > 0) {
            printf(" ");
        }
    }
    printf("\n");

    return 0;
}
