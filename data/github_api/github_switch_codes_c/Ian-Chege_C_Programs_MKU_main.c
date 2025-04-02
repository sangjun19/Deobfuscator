// switch statement format 1 to evalute student's grades

#include <stdio.h>

int main()
{
    int a, b, c, d, e, marks;
    printf("Enter your grades: ");
    scanf("%d%d%d%d%d", &a, &b, &c, &d, &e);
    marks = (a + b + c + d + e) / 5;
    printf("Average marks: %d\n", marks);

    switch (marks / 10)
    {
    case (10):
    case (9):
    case (8):
    case (7):
        printf("A\n");
        break;
    case (6):
        printf("B\n");
        break;
    case (5):
        printf("C\n");
        break;
    case (4):
        printf("D\n");
        break;
    default:
        printf("Fail\n");
    }

    return 0;
}

// switch statement format 2 to evalute student's grades

#include <stdio.h>

int main()
{
    int a, b, c, d, e, marks;
    printf("Enter your grades: ");
    scanf("%d%d%d%d%d", &a, &b, &c, &d, &e);
    marks = (a + b + c + d + e) / 5;
    printf("Average marks: %d\n", marks);

    switch (marks)
    {
    case 70 ... 100:
        printf("A\n");
        break;
    case 60 ... 69:
        printf("B\n");
        break;
    case 50 ... 59:
        printf("C\n");
        break;
    case 40 ... 49:
        printf("D\n");
        break;
    default:
        printf("Fail\n");
    }

    return 0;
}

// switch statement to evalute 1st character of color

#include <stdio.h>

int main()
{
    char color;
    printf("Enter the first chracter of your color: ");
    scanf("%c", &color);

    switch (color)
    {
    case 'b':
    case 'B':
        printf("Blue\n");
        break;
    case 'o':
    case 'O':
        printf("Orange\n");
        break;
    default:
        printf("Invalid\n");
    }

    return 0;
}

// while loop example
#include <stdio.h>
int main()
{
    int i;
    printf("Enter the value of i\n");
    scanf("%d", &i);
    while (i <= 100)
    {
        printf("%d\n", i);
        i++;
    }
    return 0;
}

// program to display even numbers between 50 to 80 using while loop
#include <stdio.h>
int main()
{
    int i = 50;
    while (i <= 80)
    {
        if (i % 2 == 0)
        {
            printf("%d\n", i);
        }
        i++;
    }
    return 0;
}

// program to display numbers between 0 to 100 that are divisible by 3 using while loop
#include <stdio.h>
int main()
{
    int i = 0;
    while (i <= 100)
    {
        if (i % 3 == 0)
        {
            printf("%d\n", i);
        }
        i++;
    }
    return 0;
}

// program to display numbers between 0 to 10 using do...while loop
#include <stdio.h>
int main()
{
    int i = 0;
    do
    {
        printf("%d\n", i);
        i++;
    } while (i <= 10);
    return 0;
}

// arrays
#include <stdio.h>

int main()
{
    int values[5] = {3, 5, 7, 9, 11};
    printf("%d", values[3]);
    return 0;
}

// arrays using loops
#include <stdio.h>

int main()
{
    int value[5], i;
    for (i = 0; i < 5; i++)
    {
        scanf("%d", &value[i]);
    }
    return 0;
}

// sum of values of a 1-D array
#include <stdio.h>

int main()
{
    int value[7];
    int sum = 0, i;
    printf("Enter the 7 values:\n");
    for (i = 0; i < 7; i++)
    {
        scanf("%d", &value[i]);
        sum = sum + value[i];
    }
    printf("The sum is %d", sum);

    return 0;
}

// program that takes a string, reverses it, and returns the reversed string
#include <stdio.h>
#include <string.h>

// Function to reverse a string
void reverseString(char *str)
{
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++)
    {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

// Function to return a reversed string
char *getReversedString(char *str)
{
    // Create a static variable to hold the reversed string
    static char reversed[100];
    // Copy the input string to the static variable
    strcpy(reversed, str);
    // Reverse the copied string
    reverseString(reversed);
    // Return the reversed string
    return reversed;
}

int main()
{
    char input[100];
    printf("Enter a string: ");
    fgets(input, 100, stdin);
    // Remove the newline character from the input string
    input[strcspn(input, "\n")] = 0;

    char *reversed = getReversedString(input);
    printf("Reversed string: %s\n", reversed);

    return 0;
}
