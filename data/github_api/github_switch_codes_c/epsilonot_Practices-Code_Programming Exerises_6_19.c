#include <stdio.h>
#include <string.h>

int main() {
    char month[10];
    int days = -1;

    printf("Enter the name of a month: ");
    scanf("%s", month);

    switch (month[0]) {
        case 'J':
        case 'j':
            if (strcmp(month, "January") == 0 || strcmp(month, "january") == 0)
                days = 31;
            else if (strcmp(month, "June") == 0 || strcmp(month, "june") == 0)
                days = 30;
            else if (strcmp(month, "July") == 0 || strcmp(month, "july") == 0)
                days = 31;
            break;

        case 'F':
        case 'f':
            if (strcmp(month, "February") == 0 || strcmp(month, "february") == 0)
                days = 28;
            break;

        case 'A':
        case 'a':
            if (strcmp(month, "April") == 0 || strcmp(month, "april") == 0)
                days = 30;
            else if (strcmp(month, "August") == 0 || strcmp(month, "august") == 0)
                days = 31;
            break;

        case 'M':
        case 'm':
            if (strcmp(month, "March") == 0 || strcmp(month, "march") == 0)
                days = 31;
            else if (strcmp(month, "May") == 0 || strcmp(month, "may") == 0)
                days = 31;
            break;

        case 'S':
        case 's':
            if (strcmp(month, "September") == 0 || strcmp(month, "september") == 0)
                days = 30;
            else if (strcmp(month, "November") == 0 || strcmp(month, "november") == 0)
                days = 30;
            break;

        case 'O':
        case 'o':
            if (strcmp(month, "October") == 0 || strcmp(month, "october") == 0)
                days = 31;
            break;

        case 'D':
        case 'd':
            if (strcmp(month, "December") == 0 || strcmp(month, "december") == 0)
                days = 31;
            break;

        default:
            days = -1;
            break;
    }

    if (days == -1) {
        printf("Invalid month name.\n");
    } else {
        printf("%s has %d days.\n", month, days);
    }

    return 0;
}

