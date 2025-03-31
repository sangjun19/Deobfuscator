/* operators:

Unary Operators:  +, -, ++, --, sizeof()

Arithmetic Operators: *, /, %,  (High in terms of below all) +, -

Note:
1. If mul operator are use in expression with same prior follow associativity  rule lhs to rhs.
2. Modulo is great to check divisibility.
3. When divide low to high ans: rem = lhs.
4. % not work on real constant ex: 3.5 % 2 = error , coz we need both int operand.
Ex: -5 %2 = -1 (Here solve normally  & assign sigh of lhs to rem).

5. / Gave quotient, also if both operand is int, Q is int, if any one real Q is real, also we need to change data type

Bitwise Operator: &, |, ^(xor), ~(not), >>(right-shift), <<(left-shift)
Note:
1. ~ also unary operator.
Ex:  5= 00000000 00000000 00000000 00000101
1's  5= 11111111 11111111 11111111 11111010
    ~5= 11111111 11111111 11111111 11111010

    Add just +1 in 5 result is = -ve
    If msb 0=+ve & 1=-ve, this represent in memory using 2's compliment

Relational Operators: <, >, <=, >=, (1st 3 is of high prior) ==, !=
Note:
1. Relational operator always gave result either 0 or 1
2. Every Non-zero Value is True and Zero is False
3. True is 1 and false is 0

Logical Operators: !, (high prior as unary) &&, ||

Conditional/Selective assignment is (ternary) Operator: -?-:-
Note: Here we cant wrote like: 5>4?return (5):return(4): // Its not allowed

Assignment Operators: =

post inncrement: x++

Compound Assignment Operators: +=, -=, *=, /=,%=, &=, |=, ^=

*/

/* Control Instruction in C:
1. Decision / Selection Control Instruction : if, if-else, nested, ladder, ?:
2. Iterative Control Instruction : while (entry loop), do while (exit loop), for
3. Switch Case Control instruction
4. goto control Instruction

*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// #include <conio.h> win lib
// #include <windows.h> for gotoxy() in modern IDE

// void add(int, int); // Global-declaration
// int add(void);
int add(int, int);
void swap(int *, int *);
void input(int *);
void display(int *);
void sort(int *);
int length(char *);
char *reverse(char *);

/* structure def */
struct book struct_input(void);
void struct_display(struct book);

/* structure */
struct date /*global-def of struct*/
{
    /* data */
    int d, m, y;
} d2;

struct book
{
    int id;
    char title[20];
    float price;
};

// struct functions
struct book struct_input()
{
    struct book b;
    printf("\nEnter BookID, Title & Price: ");
    scanf("%d %s %f", &b.id, &b.title, &b.price);
    fflush(stdin);
    // gets(b.title);
    return (b);
}
void struct_display(struct book b)
{
    printf("BookID= %d, Title= %s & Price= %f\n", b.id, b.title, b.price);
}

int main(int argc, char const *argv[]) // Call by o.s
{
    /* code */
    /*
    clrscr(); win lin
    getch(); win lib
    gotoxy(40,30); win lib
    */

    int a = ~5, b, c, choice, d = 1, e, f[10], i, j, s = 1;
    // int f[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 101};

    int A[3][3], B[3][3], C[3][3], k[5];
    float avg;
    // char g[10] = {'S', 'U', 'R', 'A', 'J', '\0'};
    char g[10] = "SURAJ", h[10];

    // struct date today = {15, 9, 2024};
    struct date today, d1;
    today.d = 15;
    today.m = 9;
    today.y = 2024;

    // d1.d = today.d;
    // d1.m = today.m;
    // d1.y = today.y;

    // d1 = today;

    // struct prg1
    printf("Enter today's date: ");
    scanf("%d/%d/%d", &d1.d, &d1.m, &d1.y);
    printf("Date: %d/%d/%d\n", d1.d, d1.m, d1.y);

    // struct prg2
    struct book b1;
    b1 = struct_input();
    struct_display(b1);

    // void add(void); // declaration
    // void add(int, int); // Loacl-declaration
    void isEven(void);
    void Nnatural(void);
    // system("clear");

    // a = sizeof(int);
    printf("\nValue of ~a is: %d\n", a);

    // printf("Enter a number to find its square: \n");
    // printf("square of %d is: %d\n", a, a * a);

    printf("\nEnter a number to chech max: ");
    scanf("%d %d", &b, &c);
    printf("max no. is: %d\n", b > c ? b : c);

    // if (/* condition */ b > 0)
    // {
    //     /* code */
    //     printf("No. %d is +ve:\n", b);
    // }
    // else
    // {
    //     printf("No. %d is -ve:\n", b);
    // }

    // else if (/* condition */)
    // {
    //     /* code */
    // }

    // while (/* condition */ d <= 5)
    // {
    //     /* code */
    //     printf("Suraj\n");
    //     d++; // d=d+1flow or increment
    // }

    // do
    // {
    //     /* code */
    //     printf("SurajGW\n");
    //     d++; // d=d+1flow or increment
    // } while (/* condition */ d <= 5);

    // for (size_t i = 0; i < count; i++)

    for (i = 1; i <= 5; i++)
    {
        /* code */
        printf("\nSurajGW");
    }

    // switch constant is int,char not real
    while (/* condition */ 1)
    {
        // below is Menu-Driven Program

        printf("\n\n1. Addition.\n2. Odd-Even.\n3. Print N Natural No.\n4. Array-Sum.\n5. String.\n6. Swap\n7. Sort\n8. StrLen & StrRev.\n9. Exit.");

        printf("\nEnter your choice number: ");
        scanf("%d", &choice);
        switch (/* expression */ choice)
        {
        case /* constant-expression */ 1:
            /* code */

            // add();     // call
            // add(b, c); // call by value, Actual arg / Parameters
            // printf("\nSum of %d & %d is: %d", b, c, add());

            // TSRS
            printf("\nEnter Two No. to Add: ");
            scanf("%d %d", &b, &c);
            printf("\nSum of %d & %d is: %d", b, c, add(b, c));
            break;

        case 2:
            isEven();
            break;
        case 3:
            Nnatural();
            break;

        case 4:
            /* Array:
            • Array is a linear collection of similar elements
            • Array is also known as Subscript variable

            Note: We can declare empty arrat as: a[] but a[]=[2,3]; accept
            We can't declare array of beyond size, less size accept
            */
            printf("Enter 10 No: ");
            for (i = 0; i <= 9; i++)
            {
                scanf("%d", &f[i]);
                s = s + f[i];
            }
            printf("Sum of above is: %d and Avg is: %f\n", s, s / 10.0);

            // two-dimentional array
            printf("\nEnter 9 Number to fill 1st 3*3 matrix: \n");
            for (i = 0; i <= 2; i++)
                for (j = 0; j <= 2; j++)
                    scanf("%d", &A[i][j]);

            printf("Enter 9 Number to fill 2nd 3*3 matrix: \n");
            for (i = 0; i <= 2; i++)
                for (j = 0; j <= 2; j++)
                    scanf("%d", &B[i][j]);

            for (i = 0; i <= 2; i++)
            {
                for (j = 0; j <= 2; j++)
                {
                    printf(" %d", C[i][j] = A[i][j] + B[i][j]);
                }
                printf("\n");
            }
            break;

        case 5:
            /* String"
            • Sequence of characters terminated at null character.
            • ASCIl code of null character is 0 (zero)
            */
            for (i = 0; g[i] != '\0'; i++)
                printf("%c", g[i]);
            // printf("%s", g);
            // puts(g);

            // take i/p from user
            printf("\nEnter ur name: ");
            scanf("%s", h); // It can't take mul-word as i/p coz it consider space as delimeter
            // gets(h); //fgets(h, 10, stdin); // Here wrote name of s-array represent its 1st-block-addr
            puts(h);
            break;

        case 6:
            printf("Entet two No, to swap: \n");
            scanf("%d%d", &b, &c);
            swap(&b, &c);
            printf("Swap Values is: b=%d & c=%d\n", b, c);
            break;

        case 7:
            input(k);
            display(k);
            sort(k);
            display(k);
            break;

        case 8:
            printf("Length of sting is: %d\n", length("Suraj"));
            // printf("Reverse of sting is: %s\n", reverse("Suraj"));
            break;

        case 9:
            exit(0);
        default:
            printf("\nInvalid Choice\n");
            break;
        }
    }

    return 0;
}

/* Function in C
void add() // Defination
{
    int b, c, s;
    printf("\nEnter Two No. to Add: ");
    scanf("%d %d", &b, &c);
    // s = b + c;
    printf("\nSum of %d & %d is: %d", b, c, s = b + c);
}

void add(int b, int c) // formal arg/parameters
{
    printf("\nEnter Two No. to Add: ");
    scanf("%d %d", &b, &c);
    printf("\nSum of %d & %d is: %d", b, c, b + c);
}

int add() // defination
{
    int b, c;
    printf("\nEnter Two No. to Add: ");
    scanf("%d %d", &b, &c);
    return (b + c); // use of () is good.
    // printf("hello"); // After return no any line execute
}
*/

int add(int b, int c)
{
    return (b + c);
}

void isEven()
{
    int b;
    printf("\nEnter a No. to check odd or even: ");
    scanf("%d", &b);
    if (/* condition */ b % 2 == 0)
        printf("\nNo. %d is even.", b);

    else
        printf("\nNo. %d is odd.", b);
}

void Nnatural()
{
    int b, c;
    printf("\nEnter a no. to print N Narural: ");
    scanf("%d", &c);
    printf("\nN narural no. of %d is: ", c);
    for (b = 1; b <= c; b++)
        printf("%d ", b);
}

/* Recursion: function calling itself
int k;
k = fun(3);
printf("Recursion of k=3 is %d", k);
int fun(int a)
{
    int s;
    if (a == 1)
        return (a);
    s = a + fun(a - 1);
    return (s);
}
*/

/* String related functions
strlen()
strrev()
strlwr()
strupr)
strcpy() strcpy(s,"BHOPAL")
strcmp() strcmp("AMAR","AMI|
strcat)
*/

/* Pointer in C:
• Pointer basics

Note:
1. Here '&' is address of/refrencing operator, it take unari variable name as arg.
2. '*' is indirection/derefrencing operator, it take unari addr as arg.

Ex: printf("\nAddre of %d is %u: \nactual value is: %d", s, &s, *&s);

3. %d in the range of: -32768 to 32767
4. %u in the range of: 0 to 65535
5. &x=7; not work but to store addr in any var we define as: int *j; j=&s;

6. Pointer is a variable that contains address of another variable.
Ex: printf("\nAddre of %d is %u: \nactual value is: %d", s, *&j, *j); // Here *& we cancel , we get j

7. pointer can hold value of same data type only

*/

/* Level of pointer
void main()
{
    int x = 5, *p, **q; // here q levelOfIndirection is 2 so i can store level 1 pointer addr
    p = &x;
//  We can read **q jargon as: q  is Pointer, to a pointer, to an int
//  Ex: **q=5; it will as: x=7
}

*/

/* Pointer's Arithmetic
1. We can add or subtract integer to / from an address, ex : p + 1;
• Pointer + n = pointer + sizeof(type of pointer) *n
Ex : p + 1 = 1000 + 2 * 1 = 1002

2. We can subtract two addresses of same type.
• Pointer1 - pointer2 = Literal subtraction / sizeof(type of pointer)

*/

void swap(int *x, int *y)
// • When formal arguments are pointer variables, it is call by reference
// • Reference means address
{
    int t;
    t = *x;
    *x = *y;
    *y = t;
}

// Application of Pointers:
void input(int *p)
{
    int i;
    printf("\nEnter five No. to Sort: ");
    for (i = 0; i <= 4; i++)
        scanf("%d", p + i);
}

void display(int *p)
{
    int i;
    for (i = 0; i <= 4; i++)
        printf("%d ", *(p + i));
}

// Bubble Sort
void sort(int *p)
{
    int round, t, i;
    for (round = 1; round <= 4; round++)
    {
        for (i = 0; i <= 3; i++)
            if (*(p + i) > *(p + i + 1))
            {
                t = *(p + i); // a[i] *(p + i)
                *(p + i) = *(p + i + 1);
                *(p + i + 1) = t;
            }
    }
}

int length(char *p)
{
    int i;
    for (i = 0; *(p + i) != '\0'; i++)
        ;
    return (i);
}

char *reverse(char *p)
{
    int l, i;
    char t;
    for (l = 0; *(p + l) != '\0'; l++)
        ;
    for (i = 0; i < l / 2; i++)
    {
        t = *(p + i);
        *(p + i) = *(p + l - 1 - i);
        *(p + l - 1 - i) = t;
    }
    return (p);
}

// What is structure?
// • Structure is a way to group variables
// • Structure is a collection of dissimilar elements
// • Defining structure means creating new data-type
// • No memory is consumed for definition of structure.

/*
SMA / DMA
• SMA : Static Memory Allocation
• DMA : Dynamic Memory Allocation

SMA Ex : int a; // Decide how much mem will alocate in compile time to vars.

DMA - Decide how much mem will alocate in Run time.
-- > dma vars have only addr not name


imp use case:
1. when we know, imput 10, 100 no.from user use Sma
2. when we don't know input vars size use Dma.

Dma, can implement using :
1. malloc();
its return type is void, its, mem block having garbage val.
Ex : int main()
{
    float *p;
    if ()
        ;
    p = (float *)malloc(4) // size of mem-block is 4 bytes, we are typecasting to float
    *p = 3.4;
}
*/

/*
2. calloc(5, 2);
-- > it will create a array of 5 block &each of size 2byte with 0 val.3. realloc();
--> it can work on only malloc() & calloc()

Syntax : void *realloc(void *block, int size);

4. free();
--> Dma vars mem will not free(leak) after func end also, to free it we use free(p);
*/
