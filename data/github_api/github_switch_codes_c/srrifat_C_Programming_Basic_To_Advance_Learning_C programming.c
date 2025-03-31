///Print your name
/*
#include<stdio.h>
int main()
{
    printf("My name is Md.Saydur Rahman ");
    return 0;

}
*/
///write a program that print your name ,phone number, your university name.
/*
#include<stdio.h>
int main()
{
    printf("My name is Md.Saydur Rahman\n");
    printf("01703639091\n\a");
    printf("Bangladesh University Of Business and Technology.\n");
    getch();


    return 0;

}
*/
//data type 1
/// Write a program that print a integer number
/*
#include<stdio.h>
int main()
{
    int num1=10,num2=20;

    printf("Numbers are  is %d , %d.\n",num1,num2);

    return 0;
}

*/
///Write a number that print a floating double and floating number
/*
#include<stdio.h>
int main()
{
    float num1=10.5;
    double num2=10.5222222224;
    char ch='a';
    printf("num1= %.1f \n",num1);
    printf("num2= %lf\n",num2);
    printf("chareacter= %c",ch);
    return 0;

}
*/
///Write a program that takes a integer number and print that number
/*
#include<stdio.h>
int main()
{
    int num;
    printf("Please enter a number = ");
    scanf("%d",&num);
    printf("You have pressed : %d",num);
    return 0;

}
*/
///SCANF
/*
#include<stdio.h>
int main()
{

    int num1,num2;
    printf("Enter two integer : ");
    scanf ("%d %d",&num1,&num2);
    printf("Numbers are : %d , %d",num1,num2);

    return 0;

}
*/
///test-size of
/*
#include<stdio.h>
int main()
{
    int i;
    float f;
    double d;
    char c;

    printf("Size of int =%d\n",sizeof(i));
    printf("Size of float =%d\n",sizeof(f));
    printf("Size of double =%d\n",sizeof(d));
    printf("Size of char =%d",sizeof(c));
    return 0;


}
*/
///ASCII demo -1
/*
#include<stdio.h>
int main()
{
    int n;
    printf("Enter any Number  : ");
    scanf("%d",&n);
    printf("The Ascii character is : %c",n);
    return 0;

}
*/
///ASCII demo -2
/*
#include<stdio.h>
int main()
{
    char ch;
    printf("Enter any character ");
    scanf("%c",&ch);
    printf("The ASCII value is = %d \n",ch);
}
*/
///Lower case to upper case
/*
#include<stdio.h>
int main()
{
    char lower;
    printf("Enter any lower case latter : ");
    scanf("%c",&lower);//a=97
    printf("The uppercase latter : %c",lower-32);//A=65
    return 0;
}
*/
///Upper case to lower case
/*
#include<stdio.h>
int main()
{
    char upper;
    printf("Enter any upper case latter : ");
    scanf("%c",&upper);//A=65
    printf("The lowercase latter : %c",upper+32);//a=97
    return 0;
}
*/
///lower case to upper case with library function
/*
#include<stdio.h>
int main()
{
    char lower,upper;
    printf("Enter any lower case latter : ");
    scanf("%c",&lower);

    upper = toupper(lower);//toupper = library function
    printf("Uppercase letter = %c",upper);
    return 0;


}
*/
///upper case to lower case with library function
/*
#include<stdio.h>
int main()
{
    char lower,upper;
    printf("Enter any upper case latter : ");
    scanf("%c",&upper);

    lower = tolower(upper);//tolower = library function
    printf("Upplower letter = %c",lower);
    return 0;

}
*/
///decimal to octal
/*
#include<stdio.h>
int main()
{

    int number;
    printf("Decimal number = ");
    scanf("%d",&number);
    printf("Octal number = %o",number);
    getch();
}
*/
/// decimal to hexa-decimal

/*
#include<stdio.h>
int main()
{

    int number;
    printf("Decimal number = ");
    scanf("%d",&number);
    printf("Hexa-decimal number = %x",number);
    getch();
}
*/
///Hexa decimal to decimal
/*
#include<stdio.h>
int main()
{

    int number;
    printf("Hexa-ecimal number = ");
    scanf("%x",&number);
    printf("Decimal number = %d",number);
    getch();
}
*/
///octal to hexa-decimal
/*
#include<stdio.h>
int main()
{

    int number;
    printf("Octal number = ");
    scanf("%o",&number);
    printf("Hexa-decimal number = %x",number);
    getch();
}
*/
/// hexa-decimal to octal
/*

#include<stdio.h>
int main()
{

    int number;
    printf("Hexa-decimal = ");
    scanf("%x",&number);
    printf("Octal number = %o",number);
    getch();
}
*/
///Write a program that takes two integer and display sum
/*
#include<stdio.h>
int main()
{

    int num1,num2,sum;
    printf("Enter two number : ");
    scanf("%d %d",&num1,&num2);
    sum = num1 +num2;
    printf("The sum is : %d",sum);
    return 0;

}
*/
///Write a program that takes two integer and display average
/*
#include<stdio.h>
int main()
{

    int num1,num2,sum;
    float avg;
    printf("Enter two number : ");
    scanf("%d %d",&num1,&num2);
    sum = num1 +num2;
    printf("The sum is : %d\n",sum);
    avg = (float) sum /2 ;//type casting
    printf("The average is = %.2f\n",avg);
    return 0;

}
*/
///Write a program that two integer and display sum,subtraction,multiplication,divitione,modulus
/*
#include<stdio.h>
int main()
{
    int num1,num2,result;
    printf("Enter two number : ");
    scanf("%d %d",&num1,&num2);

    result = num1 +num2 ;
    printf("Sum = % d \n",result);

    result = num1 -num2 ;
    printf("Sub = % d \n",result);
    result = num1 *num2 ;
    printf("Mul = % d \n",result);

    result = num1 /num2 ;
    printf("Div = % d \n",result);

    result = num1 % num2 ;
    printf("Reminder = % d \n",result);
    return 0;



}
*/
///Write a program that calculate the area of a triangle
/*
#include<stdio.h>
int main()
{
    float base,hight ,area;
    printf("Base =");
    scanf("%f",&base);

    printf("Hight =");
    scanf("%f",&hight);

    area = 1/2.0 * base * hight ;
    printf("Area = %.2f\n",area);
    return 0;




    return 0;


}

*/
///Average of three integer number
/*
#include<stdio.h>
int main()
{
    int num1,num2,num3,sum;
    float avg;
    printf("Enter three number : ");

    scanf("%d %d %d",&num1,&num2,&num3);

    sum = num1 + num2 + num3;

    avg =  sum / 3.0 ;
    printf("The sum of three number : %d\n",sum);
    printf("Average = %.2f \n",avg);
    return 0;

}
*/
///Area of a triangle from base & height
/*
#include<stdio.h>
int main()
{
    float base,height,area;
    printf("Enter base :");
    scanf("%f",&base);
    printf("Enter height :");
    scanf("%f",&height);
    area = 0.5*base *height ;

    printf("Area  = %.2f \n",area);
    return 0;

}
*/
///Given 3 length of a triangle.find tha area of the triangel
//area = sqrt(s*(s-a)*(s-b)*(s-c))
//s=(a+b+c)/2
/*
#include<stdio.h>
//#include<math.h>
int main()
{
    double a,b,c,s,area;
    printf("Enter 3 value ");
    scanf("%lf %lf %lf",&a,&b,&c);
    s = (a+b+c)/2;

    area = sqrt(s*(s-a)*(s-b)*(a-c));
    printf("Area = %.2lf",area);
    return 0;
}
*/
///Area of a rectangle
/*
#include<stdio.h>
int main()
{
    float length, width , area;
    printf("Enter length = ");
    scanf("%f",&length);
    printf("Enter width = ");
    scanf("%f",& width);
    area = length *  width ;
    printf("Area of rectangle  = %.2f ",area);
    return 0;

}
*/
///area of a circle
/*
#include<stdio.h>
//include<math.h>
int main()
{
    float area,radius;
    printf("Enter redius :");
    scanf("%f",&radius);

    area = 3.1416 * radius * radius ;//area= M_PI* radius *radius ;

    printf("Area = %.2f",area);
    return 0;


}
*/
///C_ -> F
/*

#include<stdio.h>
int main()
{
    float C,F;
    printf("Enter centigrade :");
    scanf("%f",&C);
    F=(C*1.8)+32;
    printf("F = %.2f",F);
    return 0;

}
*/

///F -> C
/*
#include<stdio.h>
int main()
{
    float C,F;
    printf("Enter Farn :");
    scanf("%f",&F);
    C=(F-32)/1.8;
    printf("C = %.2f",C);
    return 0;
}
*/
///Swap 1
/*

#include<stdio.h>
int main()
{

    int num1=10;
    int num2=5;

    int temp;
    temp =num1;//  temp =  10
    num1 =num2;//  num1 =  5
    num2 =temp; // num2 = 10
    printf("Num1 = %d\n",num1);
    printf("Num2 = %d",num2);
}

*/
///Swap without temp variable
/*

#include<stdio.h>
int main()
{

    int num1=10;
    int num2=5;

    num1 = num1 -num2;//5
    num2 = num1 +num2;//10
    num1 = num2 -num1;//5


    printf("Num1 = %d\n",num1);
    printf("Num2 = %d",num2);
}
*/
///ax^2+bx+c = 0 eq solve program
/*
#include<stdio.h>
int main()
{

    double a,b,c,d,x1,x2;
    printf("Enter the value of a ,b,c : ");
    scanf("%lf %lf %lf",&a,&b,&c);
    d=sqrt(b*b-4*a*c);
    x1=(-b+d)/(2*a);
    x2=(-b-d)/(2*a);
    printf("X1 = %lf\n",x1);
    printf("X2 = %lf",x2);

}
*/
///Absolute value
/*
#include<stdio.h>
int main()
{
    int result =abs(-25);
    printf("%d",result);
}
*/
//
///sqrt root
/*
#include<stdio.h>
int main()
{
    double result = sqrt(25);
    printf("%.2lf",result);
}
*/
///sqrt root from user
/*
#include<stdio.h>
int main()
{
    int x;
    printf("Enter any number :");
    scanf("%d",&x);
    double result = sqrt(x);
    printf("%.2lf",result);


}
*/
///power
/*
#include<stdio.h>
int main()
{
    int x,y;
    printf("Enter base number :");
    scanf("%d",&x);
    printf("Enter power value :");
    scanf("%d",&y);
    double result = pow(x,y);
    printf("%.2lf",result);
}
*/
///math1
/*
#include<stdio.h>
int main()
{
    int x = 10.5 ;
    double result = log(x);
    printf("log(%lf)",result);

}
*/
///exp
/*
#include<stdio.h>
int main()
{
    int x = 1 ;
    double result = exp(x);
    printf("exp(%lf)",result);

}
*/
///round value
/*
#include<stdio.h>
int main()
{
    double x = 10.55 ;
    double result = round(x);
    printf("round(%lf) = %.2lf \n",x,result);

}
*/
//
///trunc
/*

#include<stdio.h>
int main()
{
    double x = 10.555555555555 ;
    double result = trunc(x);
    printf("round(%lf) = %lf \n",x,result);

}
*/
///ceil
/*

#include<stdio.h>
int main()
{
    double x = 10.35 ;
    double result = ceil(x);
    printf("round(%lf) = %.2lf \n",x,result);

}
*/
///floor
/*

#include<stdio.h>
int main()
{
    double x = -10.55 ;
    double result = floor(x);
    printf("round(%lf) = %.2lf \n",x,result);

}
*/
///Assignment operator
/*
#include<stdio.h>
int main()
{
    int a=5;
    a+=2;//a=a+2
    printf("%d\n",a);

    a*=2;//a=a*2
    printf("%d\n",a);

    return 0;
}
*/
///unary plus
/*
#include<stdio.h>
int main()
{
    int x= 10;
    int result = -x;
    printf("%d",result);

    return 0;
}
*/
///postfix increment
/*
#include<stdio.h>
int main()
{

    int x= 10;
    int y=x++;//y=10
    printf("x=%d \n",x);
    printf("y=%d \n",y);
    return 0;
}
*/
///prefix increment
/*
#include<stdio.h>
int main()
{

    int x= 10;
    int y=++x;//y=11
    printf("x=%d \n",x);
    int z=++x;
    printf("z= %d \n",x);
    printf("y=%d \n",y);
    return 0;
}
*/

///postfix decrement
/*
#include<stdio.h>
int main()
{
    int x=10;
    int y =x --;//y=10
    printf("x = %d\n",x);
    printf("y = %d ",y);
    return 0;
}
*/
///prefix decrement
/*
#include<stdio.h>
int main()
{
   int x= 5;
   int y = --x;
   printf("x = %d \n",x);
   printf("y = %d ",y);

}
*/
///clear concept in increment and decrement
/*
#include<stdio.h>
int main()
{
    int x=10;

    printf("%d\n",x++);//x=10
    printf("%d\n",x);//x=11
    printf("%d\n",++x);//x=12
    printf("%d\n",x);//x=12
    printf("%d\n",x--);//x=12
    printf("%d\n",--x);//x=10

}
*/
///Relational operator
//control statement -conditional control statement , loop control statement
///conditional control statement ->   if -else ,switch
/*
>
<
>=
<=
==
!=
*/
/*
#include<stdio.h>
int main()
{


    int num;
    printf("Enter an integer :");
    scanf("%d",&num);

    if(num%2==0)
    printf("Even\n");
    else      //if(num%2!=0)
    printf("odd\n");

    return 0;

}

*/
///long statement rules -> use {}
/*
#include<stdio.h>
int main()
{
    int time=15;
    if(time==10)
        {
           printf("Good morning.\n");
           printf("Md.Saydur Rahman Rifat\n");

        }
     else
          {
             printf("Sorry it's not morning.\n");
             printf("Md.Saydur Rahman Rifat\n");
          }

}
*/
///large number
/*

#include<stdio.h>
int main()

{
    int num1,num2;
    printf("Enter fast number :");
    scanf("%d",&num1);

    printf("Enter second number :");
    scanf("%d",&num2);

    if(num1>num2)
    {
        printf("Large number is = %d\n",num1);

    }
   else if(num1<num2)
     {
         printf("Large number is = %d\n",num2);

     }
     else
        printf("Numbers are equal");

}
*/
///marks
/*

#include<stdio.h>
int main()
{

    float marks;
          printf("Enter your marks :");
    scanf("%f",&marks);

    if(marks>100)
        printf("Wrong input");
    else if(marks>=80)
        printf("A+");

    else if(marks>=70)
        printf("A");

    else if(marks>=60)
        printf("A-");

    else if(marks>=50)
        printf("B");

    else if(marks>=40)
        printf("C");

    else if(marks>=33)
        printf("D");
    else
        printf("Fail");
}
*/
///even ,odd determination
/*
#include<stdio.h>
int main()
{

    int num;
    printf("Enter any number :");
    scanf("%d",&num);
    if(num%2==0)
        printf("Even");
    else
        printf("Odd");
    return 0;
}
*/
///positive or negative number
/*
#include<stdio.h>
int main()
{

    int  num;
    printf("Enter any number :");
    scanf("%d",&num);

    if(num>0)
        printf("Positive");
    else if(num<0)
        printf("Negative");
    else
        printf("Zero");
}
*/
///large number

/*
#include<stdio.h>
int main()

{
    int num1,num2;
    printf("Enter two number :");
    scanf("%d %d",&num1,&num2);

    if(num1>num2)
    {
        printf("Large number is = %d\n",num1);

    }
   else if(num1<num2)
     {
         printf("Large number is = %d\n",num2);

     }
     else
        printf("Numbers are equal");

}
*/
///Enter a letter check vowel and consonant  (Relational operator )
/*
#include<stdio.h>
int main()
{
    char ch;
    printf("Enter a letter :");
    scanf("%c",&ch);

    if(ch=='a')
        printf("Vowel");
    else if(ch=='e')
        printf("Vowel");
    else if(ch=='o')
        printf("Vowel");
    else if(ch=='u')
        printf("Vowel");
    else if(ch=='i-')
        printf("Vowel");

   else
       printf("Consonant");

}
*/
///Enter a letter check vowel and consonant (logical operator)
/*
#include<stdio.h>
int main()

{

    char ch;
    printf("Enter a latter :");
    scanf("%c",&ch);

    if(ch=='a'|| ch =='e'||ch=='i'||ch=='o'||ch=='u')
        printf("VOWEL");
    else
        printf("Consonanat");

}
*/
///write a programs that read 3 numbers and display maximum

/*
#include<stdio.h>
int main()

{

    int num1,num2,num3;
    printf("Enter three number :");
    scanf("%d %d %d",&num1,&num2,&num3);

    if(num1>num2 && num1>num3)
        printf("Large number is = %d\n",num1);

    else if(num2>num1 && num2>num3)
        printf("Large number is = %d\n",num2);

     else if(num3>num1 && num3>num2)
            printf("Large number is = %d\n",num3);
     else
        printf("The numbers are equal");
     return 0;

}
*/
///write a programs that read 3 numbers and display minimum

/*
#include<stdio.h>
int main()

{

    int num1,num2,num3;
    printf("Enter three number :");
    scanf("%d %d %d",&num1,&num2,&num3);

    if(num1<num2 && num1<num3)
        printf("Large number is = %d\n",num1);

    else if(num2<num1 && num2<num3)
        printf("Large number is = %d\n",num2);

     else if(num3<num1 && num3<num2)
            printf("Large number is = %d\n",num3);
     else
        printf("The numbers are equal");
     return 0;

}
*/

///large number
/*

#include<stdio.h>
int main()
{
    int num1,num2,num3;
    printf("Enter 3 number :");
    scanf("%d %d %d",&num1,&num2,&num3);
    if(num1>num2 && num1>num3)
        printf("Large number is = %d\n",num1);

    else if(num2>num1 && num2>num3)
        printf("Large number is = %d\n",num2);

     else if(num3>num1 && num3>num2)
            printf("Large number is = %d\n",num3);
     else
        printf("The numbers are equal");
     return 0;


}

*/
///leap-year
/*

#include<stdio.h>
int main()
{

    int year;
    printf("Enter any year :");
    scanf("%d",&year);

    if(year%400==0)
        printf("Leap-year");
    else if(year%4==0 && year%100!=0)
        printf("Leap-year");
     else
        printf("Not leap year ");
     getch();
}
*/
///small and capital letter
/*

#include<stdio.h>
int main()
{
    char ch;
    printf("Enter any latter :");
    scanf("%c",&ch);

    if(ch>='A' && ch<='Z' )
        printf("Capital letter");
    else if(ch>='a' && ch<='z')
       printf("Small letter");

    else
        printf("Not a letter");
    getch();


}
*/
/// vowel and consonant
/*
#include<stdio.h>
int main()

{

    char ch;
    printf("Enter a latter :");
    scanf("%c",&ch);

    if(ch=='a'|| ch =='e'||ch=='i'||ch=='o'||ch=='u'||ch=='A'|| ch =='E'||ch=='I'||ch=='O'||ch=='U')
        printf("VOWEL");
    else
        printf("Consonant");

}
*/
///pass and fail
/*
#include<stdio.h>
int main()
{

    int result;
    printf("Enter mark :");
    scanf("%d",&result);
    if(result>=33)
        printf("Pass");
    else
        printf("Fail");
}
*/
///Letter grade
/*
#include<stdio.h>
int main()
{
    int mark;
    printf("Enter your Mark :");
    scanf("%d",& mark);

   if(mark>100 || mark<0)

       printf("Invalid marks");

    else if(mark>=80 && mark<=100)
          printf("A+");
    else if(mark>=70 && mark<=79)
        printf("A");
    else if(mark>=65 && mark<=69)
        printf("A-");
    else if(mark>=61 && mark<=64)
        printf("B");
    else if(mark>=50 && mark<=69)
        printf("C");
    else if(mark>=40 && mark<=49)
        printf("D");
    else if(mark>=33 && mark<=39)
        printf("E");
    else
             printf("Fail");

}
*/
///local variable
/*
#include<stdio.h>
int main()
{
    int a=10;
    printf("Inside the main function a= %d \n",a);
    //display(); // error
}
/*void display ()
{
    ptintf("Inside the display function a= %d \n",a);
}
*/
///global variable
/*
#include<stdio.h>
int a=10;
int main()
{

    printf("Inside the main function a= %d \n",a);
    display();
}
void display ()
{
    printf("Inside the display function a= %d \n",a);
}
*/

///Switch keyword -> switch ,case , break, default

///Write a program that read a digit and display it's digit
/*
#include<stdio.h>
int main()
{
    int digit;
    printf("Enter a digit :");
    scanf("%d",&digit);

    switch(digit)
    {
    case 0:
        printf("Zero\n");
        break;
    case 1:
        printf("One\n");
        break;
    case 2:
        printf("Two\n");
        break;
    case 3:
        printf("Three\n");
        break;
    case 4:
        printf("Four\n");
        break;
    case 5:
        printf("Five\n");
        break;
    case 6:
        printf("Six\n");
    case 7:
        printf("Seven\n");
        break;
    case 8:
        printf("Eight\n");
        break;
    case 9:
        printf("Nine\n");
        break;
    default:
        printf("Not a valid digit");
    }
}
*/
/// vowel -consonant using switch case
/*
#include<stdio.h>
int main()
{
    char ch;
    printf("Enter any letter : ");
    scanf("%c",&ch);

    switch(ch)
    {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'A':
    case 'E':
    case 'I':
    case 'O':
    case 'U':

        printf("vowel\n");
        break;
    default:
        printf("Consonant\n");
    }

}
*/
///menu base temperature conversion
/*
#include<stdio.h>
int main()
{
    printf("Temperature conversion menu \n");
    printf("1.Fahrenheit to Celsius\n");
    printf("2.Celsius to Fahrenheit \n");

    int choice;
    float temp,convetedtemp;

    printf("Enter your Choice :");
    scanf("%d",&choice);

    switch(choice)
    {
    case 1:
        {
            printf("Enter the Fahrenheit temperature :");
            scanf("%f",&temp);
            convetedtemp=(temp-32)/1.8;
            printf("The temperature in Celsius is : %f\n",convetedtemp);
            break;
        }
    case 2 :
        {
            printf("Enter the Celsius temperature :");
            scanf("%f",&temp);
            convetedtemp=(1.8*temp)+32;
            printf("The temperature in Fahrenheit is : %f\n",convetedtemp);
            break;
        }
    default:
        printf("Not a correct option");
    }


}
*/
///Calculator
/*
#include<stdio.h>
int main()
{
    double num1,num2;
    char oper;

    printf("Enter an operator(+,-,*,/) :");
    scanf("%c",&oper);
    printf("Enter 2 Numbers :");
    scanf("%lf %lf",&num1,&num2);

    switch(oper)
    {
    case '+':
    {
        printf("%.2lf + %.2lf = %.2lf ",num1,num2,num1+num2);
        break;
    }

    case '-':
    {
        printf("%.2lf - %.2lf = %.2lf ",num1,num2,num1-num2);
        break;
    }

    case '*':
    {
        printf("%.2lf * %.2lf = %.2lf ",num1,num2,num1*num2);
        break;
    }

    case '/':
    {
        printf("%.2lf / %.2lf = %.2lf ",num1,num2,num1/num2);
        break;
    }
    default:
        printf("Not a valid operator\n");

    }
}
*/
///Conditional operator or Ternary operator
/*

#include<stdio.h>
int main()
{

    int num1,num2,large;
    printf("Enter two numbers : ");
    scanf("%d %d",&num1,&num2);

    large = num1>num2 ? num1 : num2;//condition ? expression 1 :expression 2 ;
    printf("Large Number = %d\n",large);
    return 0;

}
*/
///Bitwise OR
/*

#include<stdio.h>
int main()
{

    int a=32;
    int b=12;
    int c;

    c=a&b;
    printf("a&b =%d\n",c);
    c=a | b;
    printf("a|b =%d\n",c);
    c=a ^ b;
    printf("a^b =%d\n",c);
}
*/

///LOOP-for,while loop,do while
///for loop
/*
#include<stdio.h>
int main()
{
    int i;
    for(i=1; i<=5; i++)               //for(initialization ; condition ; counterUpdate )
                                      //{
    {
        printf("C programming .\n");
        printf("%d\n",i);  //statement
                                      // }
    }
}
*/
///for loop
/*
#include<stdio.h>
int main()
{

    int i=1;
    for(i=1;i<=10;i++)
    {
        printf("%d\n",i);
    }
}
*/
///While loop
/*
#include<stdio.h>
int main()
{

    int i=1;               //initialization
    while(i<=10)           //condition
    {
        printf("%d\n",i);
        i++;               //increment
    }
}
*/
///while loop

/*
#include<stdio.h>
int main()
{

    int i=1;             //initialization
    do
    {
        printf("%d\n",i);
        i++;             //increment
    }
    while(i<=10);        //condition;
}
*/
///Print even number
/*
#include<stdio.h>
int main()
{

    int i;
    for(i=2;i<=100;i=i+2)
    {
        printf("%d\n",i);
    }
}
*/
///Break & continue statement*
/*
#include<stdio.h>
int main()
{

    int i;
    for(i=1;i<20;i++)
    {
        if(i%3==0)
            continue;//bypass
        printf("%d\n",i);

        if(i==10)
            break;//loop terminate
    }
}
/*
1
2
4
5
7
8
10
*/
/// Goto statement*
/*
#include<stdio.h>
int main()
{
    int i=1;
    print:                      //level:
        printf("%d\t",i);
        i++;
        if(i<5)
            goto print;         // goto level           //level is calling here
        return 0;
}

*/

///Time table

/* Enter any number :3
   num X i = num*1
   3 X 1 =3
   3 X 2 =6

   ........
   3 X 10=30

*/
/*
#include<stdio.h>
int main()
{
    while(1){                            ///use while(1) if you terminate the program again
    int num,i;
    printf("Enter any number : ");
    scanf("%d",&num);
    for(i=1;i<=10;i++)
    {
        printf("%d X %d = %d\n",num,i,num*i);
    }
    }

}
*/

///factorial
/*

#include<stdio.h>
int main()
{
    int n,i,fact=1;
    printf("Enter any positive number :");
    scanf("%d",&n);
    for(i=1;i<=n;i++)
    {
        fact=fact *i;//fact=120
    }
    printf("%d",fact);
    getch();
}
*/
///Prime number
/*
#include<stdio.h>
int main()
{
    int i,num,count=0;
    printf("Enter integer a number :");
    scanf("%d",&num);
    for(i=2;i<num;i++)
    {
        if(num%i==0)
            count++;
            break;
    }
    if(count==0)
        printf("Prime");
    else
        printf("Not Prime");

}
*/

///GCD and LCD
/*
#include<stdio.h>
int main()
{

    int num1,num2,n1,n2,rem,gcd,lcm;
    printf("Enter two number :");
    scanf("%d %d",&num1,&num2);

    n1 = num1;
    n2 = num2;
    while(n2!=0)
    {
        rem=n1%n2;
        n1=n2;
        n2=rem;

    }
    gcd=n1;
    lcm=(num1*num2)/gcd;
    printf("GCD = %d\n",gcd);
    printf("LAM = %d\n",lcm);
    getch();

}
*/
///Sum of digits
/*
#include<stdio.h>
int main()
{

    int num,temp,r,sum=0;
    printf("Enter any number : ");
    scanf("%d",&num);

    temp=num;
    while(temp!=0)
     {
         r = temp%10;
         sum=sum+r;
         temp=temp/10;
     }
     printf("Sum of digit  : %d \n",sum);

}
*/
///Reverse of a digit
/*
#include<stdio.h>
int main()
{

    int num,temp,r,sum=0;
    printf("Enter any number : ");
    scanf("%d",&num);

    temp=num;
    while(temp!=0)
     {
         r = temp%10;
         sum=sum*10+r;
         temp=temp/10;
     }
     printf("Reverse of Number : %d \n",sum);

}
*/
///Palindrome

/*
#include<stdio.h>
int main()
{

    int num,temp,r,sum=0;
    printf("Enter any number : ");
    scanf("%d",&num);

    temp=num;
    while(temp!=0)
     {
         r = temp%10;
         sum=sum*10+r;
         temp=temp/10;
     }
     if(num==sum)
     {
         printf("Palindrome Number");
     }
     else
        printf("Not Palindrome");


}
*/
///Armstrong number >1^3+5^3+3^  =   153
/*
#include<stdio.h>
int main()
{

    int num,temp,r,sum=0;
    printf("Enter any number : ");
    scanf("%d",&num);

    temp=num;
    while(temp!=0)
     {
         r = temp%10;
         sum=sum+r*r*r;
         temp=temp/10;
     }
     if(num==sum)
     {
         printf("Armstrong Number");
     }
     else
        printf("Not Armstrong");


}
*/

///User given Armstrong
/*
#include<stdio.h>
int main()
{
    int initial,finalnum,temp,r,sum=0,i;
    printf("Initial number :");
    scanf("%d",&initial);
    printf("Final number :");
    scanf("%d",&finalnum);

    for(i=initial;i<=finalnum;i++)
    {
       temp=i;
    while(temp!=0)
     {
         r = temp%10;
         sum=sum+r*r*r;
         temp=temp/10;
     }
     if(sum==i)
     {
          printf("%d\n",i);
     }

     sum=0;

     }

}
*/
///Counting digit
/*
#include<stdio.h>
int main()
{

    int num ,count=0;
    printf("Enter any integer number : ");
    scanf("%d",&num);

    while(num!=0)
    {
        num =num/10;
        ++ count;
    }
    printf("Total number of digit = %d\n",count);
}
*/
///Strong number
/*
#include<stdio.h>
int main()
{

    int num,sum=0,rem,temp,fact,i;
    printf("Enter an integer :");
    scanf("%d",&num);

    temp = num;

    while(temp!=0)
     {
         rem= temp % 10;
         fact=1;
         for(i=1;i<=rem;i++)
         {
             fact=fact*i;
         }

         sum=sum+fact;
         temp=temp/10;

     }
     if(sum==num)
        printf("%d is a strong number .",num);
     else
        printf("%d is not a Strong number.",num);
}
*/
///                Series

///1+2+3+4+......................+n
/*
#include<stdio.h>
int main()
{

    int n,sum=0,i;
    printf("Enter the last number of the series :");
    scanf("%d",&n);
    printf("1 + 2 + 3 +...............+ %d",n);
    for(i=1;i<=n;i=i+1)
    {
        sum=sum + i;
    }
    printf("= %d\n",sum);
    getch();
}
*/
///Parallel Series 1+3+5+..........+N
/*
#include<stdio.h>
int main()
{
    int n,sum=0,a=2;
    printf("Enter the last number : ");
    scanf("%d",&n);

    printf("2+4+6+.................+%d",n);
    while(a<=n)
    {
        sum=sum+a;
        a=a+2;

    }
    printf("=%d\n",sum);



    getch();
}
*/

///1*3*4 + 2*5*5 + 3*7*8 +...........+  n1 +  n2
/*
#include<stdio.h>
int main()
{
    int n1,n2,n3,sum=0,a=1,b=3,c=4;
    printf("Enter n1 , n2 and n3: ");
    scanf("%d %d %d",&n1,&n2,&n3);
    printf("1.3.4 +2.5.5 +3.7.8 +.......+ %d.%d.%d",n1,n2,n3);


    while(a<=n1 && b<=n2)
    {
        sum=sum + a*b*c ;
        a=a+1;
        b=b+2;
        c=c+2;

    }
    printf("= %d\n",sum);

    getch();

}
*/

///1 2 3 ........ n
/*

#include<stdio.h>
int main()
{
    int n,i;
    printf("Enter n = ");
    scanf("%d",&n);

    for(i=1;i<=n;i++)
    {
        printf("%d ",i);
    }
}
*/
///1+2+3+..............+n

/*
#include<stdio.h>
int main()
{
    int n,i,sum=0;
    printf("Enter n=");
    scanf("%d",&n);

    for(i=1;i<=n;i++)
    {
        printf("%d ",i);
        sum=sum + i;
    }
    printf("\nSum = %d\n",sum);
}
*/
///1+3+5+..............+n(sum of odd number)
/*
#include<stdio.h>
int main()
{
    int n,i,sum=0;
    printf("Enter n=");
    scanf("%d",&n);

    for(i=1;i<=n;i=i+2)
    {
        printf("%d ",i);
        sum=sum + i;
    }
    printf("\nSum = %d\n",sum);
}
*/

/*

#include<stdio.h>
int main()
{
    int n,i,sum=0;
    printf("Enter n=");
    scanf("%d",&n);

    for(i=2;i<=n;i=i+2)
    {
        printf("%d ",i);
        sum=sum + i;
    }
    printf("\nSum = %d\n",sum);
}
*/

///1+4+7+..............+n(sum of even number)
/*
#include<stdio.h>
int main()
{
    int n,i,sum=0;
    printf("Enter n=");
    scanf("%d",&n);

    for(i=1;i<=n;i=i+3)
    {
        printf("%d ",i);
        sum=sum + i;
    }
    printf("\nSum = %d\n",sum);
}
*/

///1.5+2.5+3.5+4.5+.........+N
/*
#include<stdio.h>
int main()
{
    float i,sum=0,n;
    printf("Enter n = ");
    scanf("%f",&n);

    for(i=1.5;i<=n;i++)
    {
        sum = sum + i;

    }
    printf("Sum = %.2f\n ",sum);


}
*/

///1^2+2^2+3^3+........+n^2
/*
#include<stdio.h>
int main()
{

    int i,n,sum=0;
    printf("Enter n = ");
    scanf("%d",&n);

    for(i=1;i<=n;i++)
    {
        sum=sum+i*i;

    }
    printf("Sum = %d ",sum);

}
*/

///1/1+1/2+1/3+........+1/n
/*
#include<stdio.h>
int main()
{
    double sum=0,n,i;
    printf("Enter n = ");
    scanf("%lf",&n);


    for(i=1;i<=n;i++)
    {
        sum=sum +(1/i);
        if(i==1)
            printf("\n1 + ");
        else if(i==n)
            printf("(1/%.1lf )",i);
        else
            printf("(1/%.1lf )+", i);

    }
    printf("\nSum = %.2lf\n",sum);

}
*/

/// 1*2*3*.......*n
/*
#include<stdio.h>
int main()
{


    int i,n,result = 1;
    printf("Enter n = ");
    scanf("%d",&n);

    for(i=1;i<=n;i++)
    {
        result=result * i ;

    }
    printf("Result = %d\n",result);

}
*/
/// 1^3*3^3*5^3*.......*n^3
/*
#include<stdio.h>
int main()
{


    int i,n,result = 1;
    printf("Enter n = ");
    scanf("%d",&n);

    for(i=1;i<=n;i=i+2)
    {
        result=result * i *i *i;

    }
    printf("Result = %d\n",result);

}
*/
///sum=1-2+3-4+5-6+.....
///sum =(1+3+5+....)-(2+4+6+.....)

/*
#include<stdio.h>
int main()
{
    int n,i,even=0,odd=0;
    printf("Enter n=");
    scanf("%d",&n);

    for(i=1;i<=n;i++)
      {
          if(i%2==0)
            even=even+i;
          else
            odd=odd+i;

      }
      printf("Sum = %d\n",odd-even);

}
*/
///Fibonacci
/*
#include<stdio.h>
int main()
{

    int first =0,second=1,count=0,fibo,n;
    printf("Enter range :");
    scanf("%d",&n);
    while(count<n)
    {
        if(count<=1)
            fibo=count;
        else
        {
            fibo=first+second;
            first=second;
            second=fibo;
        }
        printf("%d ",fibo);
        count++;
    }
}
*/
///               Pattern type-1

/// Pattern-1
/*

#include<stdio.h>
int main()
{
    int n,row,col;

    printf("Entre N =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
           printf("%d ",col);
        }
        printf("\n");
    }
}
*/

/// Pattern -2
/*
#include<stdio.h>
int main()
{

    int n,row,col;

     printf("Enter N :");
     scanf("%d",&n);

     for(row=1;row<=n;row++)
     {
         for(col=1;col<=row;col++)
         {
             printf("%d ",row);

         }
         printf("\n");
     }
}
*/
///pattern -3

/*
#include<stdio.h>
int main()
{

    int n,row,col;

     printf("Enter N :");
     scanf("%d",&n);

     for(row=1;row<=n;row++)
     {
         for(col=1;col<=row;col++)
         {
             printf("%d ",col%2);

         }
         printf("\n");
     }
}
*/
///pattern-4
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",row%2);

        }
        printf("\n");
    }

}

*/

///pattern-5
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%c ",col+64);

        }
        printf("\n");
    }



}

*/
///pattern-6
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%c ",row+64);

        }
        printf("\n");
    }



}
*/

///pattern-7
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf(" * ");

        }
        printf("\n");
    }



}
*/
///pattern-8
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf(" # ");

        }
        printf("\n");
    }



}
*/
///               Pattern type-2


///pattern-1
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=n;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",col);
        }
        printf("\n");
    }
}
*/

///pattern-2
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=n;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",row);
        }
        printf("\n");
    }
}
*/
///pattern-3
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=n;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",col%2);
        }
        printf("\n");
    }
}
*/
///pattern-4
/*

#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=n;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("* ");
        }
        printf("\n");
    }
}
*/

///     pattern type-3

///pattern 1
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",col);
        }
        printf("\n");
    }
     for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",col);
        }
        printf("\n");
    }

}
*/

///pattern 2
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter n =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",row);

        }
        printf("\n");

    }
    for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d ",row);

        }
        printf("\n");
    }
}
*/
///pattern 3
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%c ",col+64);
        }
        printf("\n");
    }
     for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("%c ",col+64);
        }
        printf("\n");
    }

}
*/
///pattern 4
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter n =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("* ");
        }
        printf("\n");
    }
     for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=row;col++)
        {
            printf("* ");
        }
        printf("\n");
    }

}
*/
///     pattern type-4
///pattern-1
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);
    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            //print space
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            //print number
            printf("%d",col);
        }
        printf("\n");
    }
}
*/
///pattern -2
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);
    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            //print space
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            //print number
            printf("%c",col+64);
        }
        printf("\n");
    }
}
*/
///pattern -3
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);
    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            //print space
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            //print number
            printf("*");
        }
        printf("\n");
    }
}
*/

///     pattern type-5
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);
    for(row=n;row>=1;row--)
    {
        for(col=1;col<=n-row;col++)
        {
            //print space
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            //print number
            printf("%d",col);
        }
        printf("\n");
    }
}

*/
///     pattern type-6
/*
#include<stdio.h>
int main()
{

    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        printf("\n");

    }
        for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");
        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        printf("\n");

    }

}
*/

///      pattern type-7
/*

#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n;col++)
        {
            printf("%d ",col);
        }
            printf("\n");

    }

}
*/
///      pattern type-8
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter N =");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=row;col++)
        {
            printf("%d  ",col*row);
        }
            printf("\n");

    }

}
*/

///      pattern type-9

///pattern -1
/*

#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter  n=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        //number of space
        for(col=1;col<=n-row;col++)
        {
            printf(" ");
        }
        //number of digit
        for(col=1;col<=2*row-1;col++)
        {
            printf("%d",col);
        }
        printf("\n");
    }

}
*/
///pattern-2
/*

#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=1; row<=n; row++)
    {
        //number of space
        for(col=1; col<=n-row; col++)
        {
            printf(" ");
        }
        //number of digit
        for(col=1; col<=2*row-1; col++)
        {
            printf("*");
        }
        printf("\n");
    }

}
*/
///      pattern type-10
/*
#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=n; row>=1; row--)
    {
        //number of space
        for(col=1; col<=n-row; col++)
        {
            printf(" ");
        }
        //number of digit
        for(col=1; col<=2*row-1; col++)
        {
            printf("*");
        }
        printf("\n");
    }

}

*/
///      pattern type-11
/*
#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=1; row<=n; row++)
    {
        //number of space
        for(col=1; col<=n-row; col++)
        {
            printf(" ");
        }
        //number of digit
        for(col=1; col<=2*row-1; col++)
        {
            printf("*");
        }
        printf("\n");
    }
     for(row=n-1; row>=1; row--)
    {
        //number of space
        for(col=1; col<=n-row; col++)
        {
            printf(" ");
        }
        //number of digit
        for(col=1; col<=2*row-1; col++)
        {
            printf("*");
        }
        printf("\n");
    }

}
*/

///      pattern type-12
/*
#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=1; row<=n; row++)
    {
        //number of space
        for(col=1; col<=n-row;col++)
        {
            printf(" ");
        }
        //number of star
        for(col=1;col<=row; col++)
        {
            printf("* ");
        }
        printf("\n");
    }

}
*/
///      pattern type-13
/*
#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=n; row>=1; row--)
    {
        //number of space
        for(col=1; col<=n-row;col++)
        {
            printf(" ");
        }
        //number of star
        for(col=1;col<=row; col++)
        {
            printf("* ");
        }
        printf("\n");
    }

}
*/
///      pattern type-14
/*
#include<stdio.h>
int main()
{
    int  n,row,col;
    printf("Enter  n=");

    scanf("%d",&n);

    for(row=1; row<=n; row++)
    {
        //number of space
        for(col=1; col<=n-row;col++)
        {
            printf(" ");
        }
        //number of star
        for(col=1;col<=row; col++)
        {
            printf("* ");
        }
        printf("\n");
    }
    for(row=n-1; row>=1; row--)
    {
        //number of space
        for(col=1; col<=n-row;col++)
        {
            printf(" ");
        }
        //number of star
        for(col=1;col<=row; col++)
        {
            printf("* ");
        }
        printf("\n");
    }

}
*/
///Rectangle star pattern
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n;col++)
        {
            if(row==1 || row == n || col ==1 || col==n)
            {
                printf("*");

            }
            else
                printf(" ");
        }
        printf("\n");
    }
}
*/

///Triangle of pattern
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n;col++)
        {
            if( col ==1 ||row == n ||  col==row)
            {
                printf("*");

            }
            else
                printf(" ");
        }
        printf("\n");
    }
}
*/
///X shape  pattern
/*
#include<stdio.h>
int main()
{
    int n,row,col;
    printf("Enter n=");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n;col++)
        {
            if(row==col || row+col==n+1)
            {
                printf("*");

            }
            else
                printf(" ");
        }
        printf("\n");
    }
}

*/
///Floyeds Triangle
/*
#include<stdio.h>
int main()
{

    int row ,col, n,count=0;
    printf("Enter number = ");
    scanf("%d",&n);


    for(row=1;row<=n;row++)
    {
        //number of columns
        for(col=1;col<=row;col++)
        {
            printf("%d",++count);

        }
        printf("\n");
    }
}
*/

/// Pattern
/*
#include<stdio.h>
int main()
{

    int row ,col, n;
    printf("Enter number = ");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");

        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        for(col=row-1;col>=1;col--)
        {
            printf("%d",col);
        }
        printf("\n");
    }

}
*/

///Pattern
/*
#include<stdio.h>
int main()
{

    int row ,col, n;
    printf("Enter number = ");
    scanf("%d",&n);

    for(row=n;row>=1;row--)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");

        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        for(col=row-1;col>=1;col--)
        {
            printf("%d",col);
        }
        printf("\n");
    }

}

*/
/// Pattern
/*
#include<stdio.h>
int main()
{

    int row ,col, n;
    printf("Enter number = ");
    scanf("%d",&n);

    for(row=1;row<=n;row++)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");

        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        for(col=row-1;col>=1;col--)
        {
            printf("%d",col);
        }
        printf("\n");
    }
    for(row=n-1;row>=1;row--)
    {
        for(col=1;col<=n-row;col++)
        {
            printf(" ");

        }
        for(col=1;col<=row;col++)
        {
            printf("%d",col);
        }
        for(col=row-1;col>=1;col--)
        {
            printf("%d",col);
        }
        printf("\n");
    }


}
*/

///                                     ARRRAY

/// Array declaration and initialization and sum of array

//data_type arrat_name [array_size]
/*

#include<stdio.h>
int main()
  {
      int num[5];
      num[0]=10;
      num[1]=20;
      num[2]=30;
      num[3]=32;
      num[4]=42;

     //or int num[]={10,20,30,32,42}


      printf("%d\n",num[0]);
      printf("%d\n",num[1]);
      printf("%d\n",num[2]);
      printf("%d\n",num[3]);
      printf("%d\n",num[4]);
      int sum= num[0]+num[1]+num[2]+num[3]+num[4];
      printf("The sum is %d\n",sum);


}
*/
/// Array input from loop
/*
#include<stdio.h>
int  main()
{
    int a[100],sum=0,i,n;

    printf("How many numbers :");
    scanf("%d",&n);

    printf("Enter numbers :");
     for(i=0;i<=n;i++)
     {
         scanf("%d",&a[i]);
     }



    for(i=0;i<n;i++)
    {
        sum=sum + a[i];
    }
    printf("The sum is : %d\n",sum);
    printf("The average is :%.2f\n",(float)sum/n);

}

*/

/// Write a program that a number and display maximum
/*
#include<stdio.h>
int main()
{
    int num[100],n,i;
    printf("How many number = ");
    scanf("%d",&n);
    printf("Enter number =");
    for(i=0;i<n;i++)
    {
        scanf("%d",&num[i]);
    }
    int max = num[0];
    for(i=1;i<n;i++)
    {
        if(max<num[i])
            max=num[i];
    }
    printf("Maximum = %d\n",max);
}
*/
/// Write a program that a number and display minimum
/*
#include<stdio.h>
int main()
{
    int num[100],i,n,min;

    printf("How many number :");
    scanf("%d",&n);
    printf("Enter number = ");
    for(i=0;i<n;i++)
    {
        scanf("%d",&num[i]);
    }
    min = num[0];
    for(i=0;i<n;i++)
    {
        if(min>num[i])
            min=num[i];
    }
    printf("Minimum = %d",min);

}
*/

///Fibonacci series using array
/* first  = 0
   second = 1
   fibo   = first + second
   first  = second
   second = fibo
*/
/*
#include<stdio.h>
int main()
{
    int first,second,fibo,n,i,a[100];
    printf("Enter the last number of Fibonacci=");
    scanf("%d",&n);
    a[0]=0;
    a[1]=1;


    for(i=2;i<n;i++)
    {
        a[i]=a[i-1]+a[i-2];

    }
    //printing the Fibonacci series
    printf("\n");
    for(i=0;i<n;i++)
    {
        printf("%d",a[i]);
    }

}
*/

///               Linear search
/*
#include<stdio.h>
int main()
{
    int num[]= {10,2,30,15,28,5};
    int value ;
    printf("Enter the number you want to search : ");
    scanf("%d",&value);
    int position = -1;
    for(int i=0; i<6; i++)
    {
        if(value==num[i])
        {
            position=i+1;
            break;
        }

    }

    if(position== -1)
    {
        printf("Not found");

    }
    else
        printf("The position of %d is = %d ",value,position);


}
*/


/// how to copy a array value
/*
#include<stdio.h>
int main()
{
    int array1[50],array2[50],i,n;
    printf("How many number  : ");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {
        scanf("%d",&array1[i]);
    }
    printf("Array 1 =");
    for(i=0;i<5;i++)
    {
        printf("%d ",array1[i]);
    }
    //copying all the elements from array 1 to array 2
    for(i=0;i<5;i++)
    {

        array2[i]= array1[i];

    }
    printf("\nArray 2 =");
    for(i=0;i<5;i++)
    {
        printf("%d ",array2[i]);
    }
}
*/

///                  2D array

/*
#include<stdio.h>
int main()
{
    int i,j,A[3][4]={{5,6,7,7},{9,10,11,12},{13,14,15,16}};
    for(i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            printf("%d ",A[i][j]);
        }
         printf("\n");

    }




}
*/

/// user input in 2D array
/*
#include<stdio.h>
int main()
{
     int i,j,A[3][4];


     for(i=0;i<3; i++)
     {
         for(j=0;j<4; j++)
         {
             printf("A[%d][%d] = ",i,j);
             scanf("%d",&A[i][j]);
         }
         printf("\n\a");

     }
      for(i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            printf("%d ",A[i][j]);
        }
         printf("\n");

    }

}
*/


/// Matrix
/*
#include<stdio.h>
int main()
{
     int i,j,A[3][4],B[3][4];

    //scanning A matrix
    printf("Enter elements for A matrix:\n");
     for(i=0;i<3; i++)
     {
         for(j=0;j<4; j++)
         {
             printf("A[%d][%d] = ",i,j);
             scanf("%d",&A[i][j]);
         }
         printf("\n\a");

     }
    //printing A matrix
    printf("A =");
      for(i=0;i<3;i++)
    {
        printf("\t");
        for(j=0;j<4;j++)
        {
            printf("%d ",A[i][j]);
        }
         printf("\n");

    }
    //scanning  B matrix
    printf("\n\nEnter elements for B matrix:\n");
     for(i=0;i<3; i++)
     {
         for(j=0;j<4; j++)
         {
             printf("A[%d][%d] = ",i,j);
             scanf("%d",&B[i][j]);
         }
         printf("\n\a");

     }
    //printing B matrix
    printf("A =");
      for(i=0;i<3;i++)
    {
        printf("\t");
        for(j=0;j<4;j++)
        {
            printf("%d ",B[i][j]);
        }
         printf("\n");

    }

}
*/


/// Matrix addition and subtraction
/*
#include<stdio.h>
int main()
{
    int numberofRows,numberofcols;
    int i,j,A[10][10],B[10][10],C[10][10];

    printf("Enter the numbers of rows and cols :\n");
    scanf("%d %d",&numberofRows,&numberofcols);

    //input for a matrix
    printf("Enter the number for A matrix = \n");
    printf("A = \n");
    for(i=0; i<numberofRows; i++)
    {

        for(j=0; j<numberofcols; j++)
        {
            printf("A[%d][%d] = ",i,j);
            scanf("%d",&A[i][j]);

        }
        printf("\n");

    }

    //input for b matrix

   printf("Enter the value of B matrix = \n");
    printf("B = \n");
    for(i=0; i<numberofRows; i++)
    {


        for(j=0; j<numberofcols; j++)
        {
            printf("B[%d][%d] = ",i,j);

            scanf("%d",&B[i][j]);
        }

        printf("\n");

    }
    //printing for a matrix


    printf("A = ");
    for(i=0; i<numberofRows; i++)
    {
        printf("\t");
        for(j=0; j<numberofcols; j++)
        {
            printf("%d ",A[i][j]);

        }
        printf("\n");

    }

   //print the number of b matrix
     printf("B = ");
    for(i=0; i<numberofRows; i++)

    {
        printf("\t");
        for(j=0; j<numberofcols; j++)
        {
            printf("%d ",B[i][j]);
        }
        printf("\n");
    }



   //adding the matrix

   for(i=0;i<numberofRows;i++)

   {
       for(j=0;j<numberofcols;j++)
       {
           C[i][j]= A[i][j]+ B[i][j];

       }

   }

   printf("A+B = ");

       for(i=0; i<numberofRows; i++)
        //print the number of c matrix
    {

        for(j=0; j<numberofcols; j++)
        {
            printf("%d ",C[i][j]);
        }
        printf("\n");
        printf("      ");

    }
    printf("\n");
    //subtraction  the matrix

   for(i=0;i<numberofRows;i++)

   {
       for(j=0;j<numberofcols;j++)
       {
           C[i][j]= A[i][j] - B[i][j];

       }

   }

   printf("A-B = ");

       for(i=0; i<numberofRows; i++)
        //print the number of c matrix
    {

        for(j=0; j<numberofcols; j++)
        {
            printf("%d ",C[i][j]);
        }
        printf("\n");
        printf("      ");
    }
  return 0;


}
*/


///             Matrix multiplication



/*
#include<stdio.h>
int main()
{
    int first[10][10],second[10][10],result[10][10],r1,r2,c1,c2,i,j,k,sum=0;
    printf("Enter row and column for the 1st matrix : ");
    scanf("%d %d",&r1,&c1);

    printf("Enter row and column for the 2nd matrix : ");
    scanf("%d %d",&r2,&c2);

    while(c1!=r2)
    {
        printf("Error !! Column of first matrix not equal to row of second \n");

        printf("Enter row and column for the 1st matrix : ");
        scanf("%d %d",&r1,&c1);

        printf("Enter row and column for the 2nd matrix : ");
        scanf("%d %d",&r2,&c2);

    }

    //taking input for 1st matrix
    printf("Enter element for 1st matrix = \n");
    for(i=0; i<r1; i++)
    {

        for(j=0; j<c1; j++)
        {

            printf("First[%d][%d] = ",i,j);
            scanf("%d",&first[i][j]);
        }
        printf("\n");
    }

    //taking input for 2nd matrix
    printf("\nEnter element for 2nd matrix = \n");
    for(i=0; i<r2; i++)
    {

        for(j=0; j<c2; j++)
        {
            printf("Second[%d][%d] = ",i,j);
            scanf("%d",&second[i][j]);
        }
        printf("\n");
    }

    //multiplying matrix
    for(i=0; i<r1;i++)
    {
        for(j=0; j<c2; j++)
        {
            for(k=0; k<c1; k++)
            {

                sum=sum + first[i][k] * second[k][j];

            }
            result[i][j] = sum ;
            sum=0;
        }
    }

    //printing  for 1st matrix
    printf("\n\nFirst matrix :\n");
    for(i=0; i<r1; i++)
    {

        for(j=0; j<c1; j++)
        {
            printf("%d ",first[i][j]);


        }
        printf("\n");


    }

    //printing  for 2nd matrix
    printf("\n\nSecond matrix :\n");
    for(i=0; i<r2; i++)
    {

        for(j=0; j<c2; j++)
        {
            printf("%d ",second[i][j]);

        }
        printf("\n");

    }


    //printing  result matrix
    printf("\n\nResult matrix :\n");
    for(i=0; i<r1; i++)
    {

        for(j=0; j<c2; j++)
        {
            printf("%d ",result[i][j]);

        }
        printf("\n");

    }


}
*/


///              Transpose matrix
/*


#include<stdio.h>
int main()
{
    int A[10][10], transpose[10][10], i,j,row,col;
    printf("Enter the number of row and colums for the matrix :");
    scanf("%d %d",&row,&col);
    //getting element for the matrix
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            printf("A[%d][%d] = ",i,j);
            scanf("%d",&A[i][j]);
        }
        printf("\n");
    }

     //printing element for the matrix
     printf("Entered matrix \n");
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            printf("%d ",A[i][j]);
        }
        printf("\n");
    }

    //transpose loop


    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            transpose[j][i]=A[i][j];


        }
    }
    //printing transpose matrix
    printf("Transpose matrix = \n");

    for(i=0;i<col;i++)
    {
        for(j=0;j<row;j++)
        {
            printf("%d ",transpose[i][j]);


        }
        printf("\n");

    }

}
*/


///             Diagonal Matrix
/*
#include<stdio.h>
int main()
{
    int A[3][3],i,j,sum=0;
    //getting the element for the matrix
    printf("\nEnter the element for the matrix .\n");
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            printf("A[%d][%d] = ",i,j);
            scanf("%d",&A[i][j]);

        }
    }
    printf("Entered Matrix = \n");
     for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {

            printf("%d ",A[i][j]);

        }
        printf("\n");
    }
    //sum of diagonal element

    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            if(i==j)
            {
                printf("%d ",A[i][j]);

                 sum=sum+A[i][j];
            }

        }

    }
    printf("Sum of diagonal element is  = %d ",sum);



}

*/

/// upper sum and lower sum element
/*
#include<stdio.h>
int main()
{
    int A[3][3],i,j,uppersum=0,lowersum=0;
    //getting the element for the matrix
    printf("\nEnter the element for the matrix .\n");
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            printf("A[%d][%d] = ",i,j);
            scanf("%d",&A[i][j]);

        }
    }
    printf("Entered Matrix = \n");
     for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {

            printf("%d ",A[i][j]);

        }
        printf("\n");
    }
    //upper sum and lower sum element

    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            if(i<j)
            {
                uppersum= uppersum+A[i][j];

            }
            if(i>j)
            {
                lowersum=lowersum+A[i][j];
            }

        }

    }
    printf("Upper sum triangles is  = %d \n",uppersum);
    printf("Lower sum triangles is  = %d ",lowersum);



}
*/

///               STRING

/*
#include<stdio.h>
int main()
{
    char s1[6];
    s1[0]= 'R';
    s1[1]= 'i';
    s1[2]= 'f';
    s1[3]= 'a';
    s1[4]= 't';
    s1[6]='\0'; //null character
    printf("s1 = %s \n",s1);
}
*/
/*
#include<stdio.h>
int main()
{
    // char s1[6]={'R','i','f','a','t'};
    //extra index for null Character
    // char s1[]="Saydur Rahman";
    // "\"-> line breaking character
    char s1[]="Saydur Rahman\
                      rifat";


    printf("S1 = %s \n",s1);
}
*/

/// String input
/*
#include<stdio.h>
int main()
{
    char s1[30];
    printf("Enter your full name : ");
    gets(s1);//full sting function -> gets ();

    printf("Full name = %s\n ",s1);
}
*/


/// show character by character string
/*
#include<stdio.h>
int main()
{
    char s1[]="Saydur";
    int i=0;
    while(s1[i] !='\0')
    {
        printf("%c\n",s1[i]);
        i++;
    }
}
*/


///finding length of string using strlen() function
/*

#include<stdio.h>
int main()
{
    char s1[] = "Saydur Rahman Rifat";
    int len = strlen(s1);
    printf("Length = %d\n ",len);
}
*/

///finding length of string without strlen() function
/*
#include<stdio.h>
int main()
{
    char s1[]="Saydur Rahman Rifat";
    int i=0,len=0;
    while(s1[i]!='\0')
    {
        i++;
        len++;

    }
    printf("Length = %d\n",len);
}
*/

/// A string copy in another string using strcpy() funtion
/*
#include<stdio.h>
int main()
{
    char scurce[]="C programming";
    char target[20];
    strcpy(target,scurce);
    printf("Source string = %s\n",scurce);
    printf("Target string = %s\n",target);
}
*/

///Concat string  using strcat() function
/*
#include<stdio.h>
int main()
{
    char str1[]="My name is ";
    //char str2[]="Saydur Rahman";
    strcat(str1,"Saydur Rahman");
    printf("Str1 = %s \n",str1);
}
*/

///Concat string  without strcat() function
/*
#include<stdio.h>
int main()
{
    char str1[50]="My name is ";
    char str2[]="Saydur Rahman";
    int i=0,len=0,j=0;
    while(str1[i]!='\0')
    {
      i++;
      len++;
    }
    while(str2[j]!='\0')
    {
        str1[len+j]=str2[j];
        j++;
    }
    printf("Str1 = %s \n",str1);
}
*/


///String compare Strcmp()
/*
#include<stdio.h>
int main()
{
    char str1[]="Saydur Rahman";
    char str2[]="Saydur Rahman";


    int d = strcmp(str1,str2);
    if(d==0)
    {
        printf("Strings are equal");

    }
    else
        printf("Stings are not equal");
}
*/

///Strings reverse using strrev()
/*
#include<stdio.h>
int main()
{
   char str[]="Saydur Rahman";
    printf("Str = %s\n",str);


   strrev(str);

   printf("Str = %s\n",str);



}
*/
///Strings reverse without strrev() function
/*
#include<stdio.h>
int main()
{
   char str1[]="Saydur Rahman";
   char str2[30];

  int i=0,len=0,j;
  while(str1[i]!='\0')
  {
      i++;
      len++;
  }
  for(j=0,i=len-1;i>=0;i--,j++)
  {
      str2[j]=str1[i];
  }
  str2[j]='\0';
  printf("str1 = %s\n",str1);
  printf("Str2 = %s\n",str2);



}
*/
///String palindrome
/*
#include<stdio.h>
int main()
{
    char str1[]="MAM";
    char str2[30];

    int i=0,len=0,j;
    while(str1[i]!='\0')
    {
        i++;
        len++;
    }
    for(j=0,i=len-1; i>=0; i--,j++)
    {
        str2[j]=str1[i];
    }
    str2[j]='\0';
    printf("str1 = %s\n",str1);
    printf("Str2 = %s\n",str2);

    int d= strcmp(str1,str2);
    if(d==0)
        printf("String is palindrome.");
    else
        printf("String is not palindrome.");

    getch();
}
*/

///String swapping
/*
#include<stdio.h>
int main()
{
    char str1[15]= "Sakib";
    char str2[15]= "Tamim";
    printf("Before swapping:\n");
    printf("Str1 = %s\n",str1);
    printf("Str2 = %s\n",str2);

    char temp[15];
    strcpy(temp,str1);
    strcpy(str1,str2);
    strcpy(str2,temp);

    printf("\n\nAfter swapping:\n");
    printf("Str1 = %s\n",str1);
    printf("Str2 = %s\n",str2);
}
*/


///Using strupr()   and strlwr()
/*
#include<stdio.h>
int main()
{
    char str1[]= "Saydur Rahman";
    char str2[]= "SAYdur RAhmaN";
    strupr(str1);
    strlwr(str2);
    printf("Str1 = %s\n",str1);
    printf("Str2 = %s\n",str2);

}
*/


///Finding number of vowel ,consonant ,words,digits and others from string
/*
#include<stdio.h>
int main()
{
    char str[100],ch;
    int i,vowel,consonant,word,digit,other;

    printf("Enter a string : ");
    gets(str);

    i=vowel=consonant=word=digit=other=0;
    while((ch=str[i])!='\0')
    {

        if(ch=='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u'||
                ch=='A'||ch=='E'||ch=='I'||ch=='O'||ch=='U')
            vowel++;
        else if((ch>='a' && ch<='z') || (ch>='A' && ch<='Z'))
            consonant++;

        else if(ch>='0' && ch>='9')
            digit++;
        else if(ch==' ')
            word++;
        else
            other++;
        i++;

    }
    word++;



   printf("Number of vowel     =%d\n",vowel);
   printf("Number of Consonant =%d\n",consonant);
   printf("Number of Words     =%d\n",word);
   printf("Number of Digits    =%d\n",digit);
   printf("Number of Others    =%d\n",other);

}
*/

///Number of capital-small letters,digits
/*
#include<stdio.h>
int main()
{
  char str[50];

  int i,capital,small,digit;
  i=capital=small=digit=0;


  printf("Enter a string : ");
  gets(str);

  while(str[i]!='\0')
  {
      if(str[i]>=65 && str[i]<90)
        capital++;
      else if(str[i]>=97 && str[i]<122)
        small++;
      else if(str[i]>=48 && str[i]<57)
       digit++;

      i++;
  }
  printf("Number of capital latter = %d\n",capital);
  printf("Number of small latter = %d\n",small);
  printf("Number of Digit latter = %d\n",digit);



}
*/

///                       FUNCTION
/*
#include<stdio.h>
//Return_type Function_name(arguguments)

int sum(int a,int b)
{
   return a+b;
}


int main()
{
    int num1,num2;
    printf("Enter two numbers :");
    scanf("%d %d",&num1,&num2);
    //int sum = num1 + num2;
    int result=sum(num1,num2);
    printf("The sum is : %d\n",result );//sum(num1,num2)
}
*/

///Squire with a function
/*
#include<stdio.h>
int squire(int a)
{
    return a*a;
}
int main()
{
    int num;
    printf("Enter any integer number :");
    scanf("%d",&num);
    int result = squire(num) ;
    printf("Squire is %d\n ",result);
}
*/


///Area of a Triangle
/*
#include<stdio.h>
double triangelarea (double b,double h);
int main()
{
    double base,height;
    printf("Enter Base = ");
    scanf("%lf",&base);
     printf("Enter hight = ");
    scanf("%lf",&height);

    double area= triangelarea(base,height);

    printf("Area = %.2lf\n",area);
}
 double triangelarea (double b,double h)
 {
     return 0.5*b*h;
 }

*/


/// X to the power Y using user-defined function

/*
#include<stdio.h>

void calculatePower(double base,double exp)
{
    double result=1,i;
    for(i=1;i<=exp;i++)
    {
        result =result * base;
    }
    printf("%.2lf\n",result);
}
int main()
{
    double base,exp;
    printf("Enter base = ");
    scanf("%lf",&base);
    printf("Enter Exponent  = ");
    scanf("%lf",&exp);
    calculatePower(base,exp);
}
*/


/// X to the power y using Library function
//base^exponent
//2^3 = 2*2*2 = 8
//2^4 = 2*2*2 = 16
//3^4 = 3*3*3*3 = 81
/*
#include<stdio.h>
int main()
{
  double base,exp,result;

  printf("Enter base =");
  scanf("%lf",&base);
  printf("Enter Exponent = ");
  scanf("%lf",&exp);

  result = pow(base,exp);
  printf("%.2lf",result);


}
*/
/// X to the power y not using Library function
/*
#include<stdio.h>

int main()
{
  double base,exp,result=1,i;

  printf("Enter base =");
  scanf("%lf",&base);
  printf("Enter Exponent = ");
  scanf("%lf",&exp);

  for(i=1;i<=exp;i++)
  {
      result= result*base;//result=2*2=4*2=8
  }
  printf("%.1lf\n",result);

}
*/

///Passing array to a function
/*
#include<stdio.h>

void display(int x[])
{
    for(int i=0;i<5;i++)
    {
        printf("%d\n",x[i]);
    }
}

int main()
{
    int num[]={12,23,42,12,34};
    display(num);

}
*/

///finding maximum value from an array using function
/*
#include<stdio.h>
int maximum(int x[])
{
    int max = x[0];
    for(int i=1;i<5;i++)
    {
        if(max<x[i])
            max=x[i];
    }

    return max;
}
int main()
{
    int num[]={12,23,50,12,34};
    int maximumValue = maximum(num);
    printf("Maximum = %d\n ",maximumValue);
}
*/

///passing string to a function
/*
#include<stdio.h>
void display(char s1[])
{
    int i=0;
    while(s1[i]!='\0')
    {
        printf("%c\n",s1[i]);
        i++;
    }
}

int main()
{
    char str[]="Saydur";
    display(str);
}
*/

///Factorial using Recursion
/*

#include<stdio.h>
int fact(int n);
int main()
{
    int result=fact(5);
    printf("Factorial of 5 = %d",result);

}
int fact(int n)
{
    if(n==1)
        return 1;
    else
        return n*fact(n-1);
}

*/

///          STRUCTURE
/*
#include<stdio.h>
//global structure
struct Person
{
    int age;
    float salary ;
};

int main()
{

    struct Person person1,person2;
    printf("Person 1 : \n");
    person1.age=27;
    person1.salary=12312.50;
    printf("Age = %d\n",person1.age);
    printf("Salary  = %.2f\n",person1.salary);

    printf("Person 2 : \n");
    person1.age=25;
    person2.salary=11342.50;
    printf("Age = %d\n",person2.age);
    printf("Salary  = %.2f\n",person2.salary);
    getch();
}
*/

///     Local  structure
/*
#include<stdio.h>

int main()
{
    //local structure
    struct Person
  {
       int age;
      float salary ;
  };

    struct Person person1,person2;//local variable
    printf("Person 1 : \n");
    person1.age=27;
    person1.salary=12312.50;
    printf("Age = %d\n",person1.age);
    printf("Salary  = %.2f\n",person1.salary);

    printf("Person 2 : \n");
    person1.age=25;
    person2.salary=11342.50;
    printf("Age = %d\n",person2.age);
    printf("Salary  = %.2f\n",person2.salary);
    getch();

}

*/

/// Input structure element
/*
#include<stdio.h>
//global structure
struct Person
{
    int age;
    float salary ;
};

int main()
{

    struct Person person1,person2;
    printf("Enter information for person 1 :\n");
    printf("Enter age : ");
    scanf("%d",&person1.age);

    printf("Enter Salary :");
    scanf("%f",&person1.salary);

    printf("Enter information for person 2 :\n");
    printf("Enter age : ");
    scanf("%d",&person2.age);

    printf("Enter Salary :");
    scanf("%f",&person2.salary);


    printf("Person 1 : \n");
    printf("Age = %d\n",person1.age);
    printf("Salary  = %.2f\n",person1.salary);

    printf("Person 2 : \n");
    printf("Age = %d\n",person2.age);
    printf("Salary  = %.2f\n",person2.salary);
    getch();
}
*/

///Initialization structure variable
/*
#include<stdio.h>
//global structure
struct Person
{
    int age;
    float salary ;
};

int main()
{

    struct Person person1={27,2512312};
    struct Person person2,person3;

    //element wise assignment
    person2.age=25;
    person2.salary=27750.12;

    //structure variable assignment
    person3=person2;

    printf("Person 1 : \n");
    printf("Age = %d\n",person1.age);
    printf("Salary  = %.2f\n",person1.salary);

    printf("Person 2 : \n");
    printf("Age = %d\n",person2.age);
    printf("Salary  = %.2f\n",person2.salary);

    printf("Person 3 : \n");
    printf("Age = %d\n",person3.age);
    printf("Salary  = %.2f\n",person3.salary);

}
*/

/// Structure comparison
/*
#include<stdio.h>
//global structure
struct Person
{
    int age;
    float salary ;
};

int main()
{

    struct Person person1={27,2512312};
    struct Person person2,person3;

    //element wise assignment
    person2.age=25;
    person2.salary=27750.12;

    //structure variable assignment
    person3=person2;

    if(person2.age==person3.age && person2.salary==person3.salary)
    printf("Person 2 equal person 3");
    else
        printf("person2 not equal to person3");

    getch();


}
*/
///Array of structure
/*
#include<stdio.h>
//global structure
struct Person
{
    int age;
    float salary ;
};

int main()
{
   struct Person person[4];
   int i;
   for(i=0;i<4;i++)
   {
       printf("Enter  information for Person %d = \n",i+1);
       printf("Enter age : ");
       scanf("%d",&person[i].age);
       printf("Enter salary : ");
       scanf("%f",&person[i].salary);
   }
   for(i=0;i<4;i++)
   {
       printf("Information for Person %d = \n",i+1);
       printf("Age : %d\n",person[i].age);
       printf("Salary : %.2f\n",person[i].salary);

   }




   getch();

}
*/

///Array within structure
/*
#include<stdio.h>
//global structure
struct Person
{
    char name[50];
    int age;
    float salary ;
};

int main()
{
   struct Person person[4];
   int i;
   for(i=0;i<4;i++)
   {
       printf("\nEnter  information for Person %d = \n",i+1);
       printf("Enter name :");
       fflush(stdin);
       gets(person[i].name);
       printf("Enter age : ");
       scanf("%d",&person[i].age);
       printf("Enter salary : ");
       scanf("%f",&person[i].salary);
   }
   for(i=0;i<4;i++)
   {
       printf("\nInformation for Person %d = \n",i+1);

       printf("Name : %s\n",person[i].name);
       printf("Age : %d\n",person[i].age);
       printf("Salary : %.2f\n",person[i].salary);

   }

   getch();

}
*/

///Passing structure variable to function
/*
#include<stdio.h>
struct Person
{
    char name[20];
    int age;
    float salary;

};
void display(struct Person p)
{
    printf("\nName : %s\n",p.name);
    printf("Age : %d\n",p.age);
    printf("Salary : %.2f\n",p.salary);
}
int main()
{
    struct Person person1,person2;

    strcpy(person1.name,"Saydur Rahman");
    person1.age=21;
    person1.salary=20000.00;
    display(person1);


    strcpy(person2.name,"Sakib al hasan");
    person2.age=33;
    person2.salary=2000000000.00;
    display(person2);

    getch();
}
*/


///            union
/*
#include<stdio.h>
union test
{
    int x,y;
};
int main()
{
    union test t1;

    t1.x=10;
    printf("t1.x = %d\n",t1.x);
    printf("t2.y = %d\n",t1.y);

    t1.x=20;
    printf("t1.x = %d\n",t1.x);
    printf("t2.y = %d\n",t1.y);


    getch();
}
*/

/// Size of union and structure
/*
#include<stdio.h>
union test1
{
    int x,y;
};
union test2
{
    char ch;
    int x;
};
union test3
{
    char name[20];
    double d;
};

struct  test4
{
    char ch;
    int x;
    int y;

};
int main()
{

     union test1 t1;
     union test2 t2;
     union test3 t3;
     struct test4 t4;

     printf("sizeof(test1) = %d\n",sizeof(t1));
     printf("sizeof(test2) = %d\n",sizeof(t2));
     printf("sizeof(test3) = %d\n",sizeof(t3));
     printf("sizeof(test4) = %d\n",sizeof(t4));



    getch();
}
*/

///         Enumeration
/*
#include<stdio.h>

enum days_of_week
{
    Sun,Mon,Tue,Wed,Thu,Fri,Sat
};
int main()
{

    enum days_of_week day1,day2;

    day1 = Tue;
    day2 = Wed;

    int dif = day2-day1;
    printf("Day1 = %d\n",dif);
    getch();
}
*/

///      typedef
/*

#include<stdio.h>

int main()
{
    typedef char LETTER;

    LETTER ch;
    ch='A';
    printf("Ch = %c\n",ch);
    LETTER ch1;
    ch1 = 'B';
    printf("Ch1 = %c\n",ch1);

    getch();
}
*/

///Custom Data type
/*
#include<stdio.h>
struct book
{
    char name[20];
    int price;
};
int main()
{
    typedef struct book BOOK;
    BOOK b = {"AMER jobon kotha",350};
    printf("BOOK Name = %s\n",b.name);
    printf("BOOK price = %d\n",b.price);
    getch();
}
*/


///         Pointer
/*
#include<stdio.h>
int main()
{
    int x=15;

    int *ptr;
    ptr = &x;

    printf("Value of x = %d\n",x);
    printf("Address of x = %d\n",&x);
    printf("Value of pointer = %d\n",ptr);
    printf("contest  of pointer = %d\n",*ptr);
    printf("Address of pointer = %d\n",&ptr);


    getch();
}
*/

///Pointer pointing different variable
/*
#include<stdio.h>
int main()
{
    int x=10,y=20,z=30;

    int *ptr;

    ptr = &x;
    printf("X = %d\n",*ptr);
    ptr = &y;
    printf("Y = %d\n",*ptr);
    ptr = &z;
    printf("Z = %d\n",*ptr);

}

*/

///Adding two numbers using pointer
/*
#include<stdio.h>
int main()
{
    int x=10,y=20,sum;

    int *ptr1,*ptr2;
    ptr1= &x;
    ptr2= &y;

    sum = *ptr1 + *ptr2;
    printf("Sum is = %d\n",sum);

}
*/

///Swapping two number using pointer
/*
#include<stdio.h>
int main()
{
    int x=10,y=20,temp;

    int *ptr1,*ptr2;
    ptr1 = &x;
    ptr2 = &y;

    //swapping
    temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;

    printf("X = %d\n",x);
    printf("y = %d\n",y);


}
*/

///Swapping two number using pointer and function
/*
#include<stdio.h>
void swapping(int *ptr1,int *ptr2)
{
    int temp;
    temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;

}
int main()
{
    int x=10,y=20;

    printf("Before swapping : x=%d , y=%d\n",x,y);
    swapping(&x,&y);
    printf("After swapping : x=%d  , y=%d\n",x,y);

    getch();
}
*/

///Access array element using pointer
/*
#include<stdio.h>
int main()
{
    int a[5] ={10,20,30,40,50};
    int *ptr,i;
    ptr = &a[0];

    for(i=0;i<5;i++)
    {
        printf("%d\n",*ptr);
        ptr++;
    }
    getch();
}
*/

///             FILE
/*
#include<stdio.h>
int main()
{

    FILE*file;
    file =  fopen("test.txt","w");
    if(file==NULL)
    {
        printf("File doesn't exist");
    }

    else
    {
         printf("File is opened");
         fclose(file);
    }



    getch();
}
*/
///writing to a file using fputc()
/*
#include<stdio.h>
int main()
{
    FILE*file;
    char name[20]="\nAnisul Islam";
    int length = strlen(name);
    int i;

    file = fopen("test.txt","a");
    if(file==NULL)
    {
        printf("File doesn't exist");
    }
    else
    {
        printf("File is opened");
        for(i=0;i<length;i++)
        {
            fputc(name[i],file);//fputs(format specifiers,variable);
        }
        printf("\nFile is written successfully");
        fclose(file);
    }



}
*/
///  file using  fputs()
/*
#include<stdio.h>
int main()
{
    FILE *file;
    char name[50];

    file = fopen("test.txt","a");

    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is opened");
        printf("\nEnter your university name :");
        gets(name);

        fputs(name,file);  //fputs(format specifiers,variable);
        fputs("\n",file);
        printf("File is written successfully.");
        fclose(file);
    }

}
*/

/// File using fprintf()
/*
#include<stdio.h>
int main()
{
    FILE *file;
    char name[50];
    int age;

    file = fopen("test.txt","a");

    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is opened");
        printf("\nEnter your name :");
        gets(name);
        printf("Enter your age : ");
        scanf("%d",&age);

        //fprintf(filepointer,format specifiers,variable);
        fprintf(file,"Name = %s ,Age = %d\n",name,age);


        printf("File is written successfully.");
        fclose(file);
    }

}
*/

///Reading a file using fgetc()
/*
#include<stdio.h>
int main()
{
    FILE *file;
    char ch;


    file = fopen("test.txt","r");

    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is opened\n");
        while(!feof(file))
        {
            ch = fgetc(file);
            printf("%c",ch);

        }


        fclose(file);
    }
}

*/
///Reading a file using fgets()
/*
#include<stdio.h>
int main()
{
    FILE *file;
    char ch[40];


    file = fopen("test.txt","r");

    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is opened\n");
        //fgets("variable,size,filepointer");
        while(!feof(file))
        {
             fgets(ch,100,file);
             printf("%s\n",ch);

        }

        fclose(file);
    }
}
*/


///reading file using fsacnf()
/*
#include<stdio.h>
int main()
{
    FILE *file;
    char fname[40];
    char lname[30];
    int age;

    file = fopen("test.txt","r");

    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is opened\n");
        //fscanf(filepointer,formatespecifier,varivale)
        fscanf(file,"%s\t%s %d",&fname,&lname,&age);
        printf("%s\t%s %d\n",fname,lname,age);

        fclose(file);
    }
}
*/
///          Storing Student details
/*
#include<stdio.h>
int main()
{

    FILE *file;
    char name[20];
    int age,phoneNumber,num;
    file  = fopen("student.txt","a");
    if(file==NULL)
    {
        printf("File does not exist");
    }
    else
    {
        printf("File is open\n");
        printf("Enter number of student : ");
        scanf("%d",&num);

        for(int i=1;i<=num;i++)
        {

            printf("Enter students Name : ");
            getchar();
            gets(name);

            printf("Enter students age : ");
            scanf("%d",&age);

            printf("Enter students Phone number :");
            scanf("%d",&phoneNumber);

            fprintf(file,"\n%s\t\t%d\t%d\n",name,age,phoneNumber);
        }

        fclose(file);
    }


    getch();

}
*/
/*
///Perprocessor define

#include<stdio.h>
#define name "Saydur Rahman"
#define sr printf("\nSR Rifat");

int main()
{
    printf("%s\n"name);
    sr
    getch();
}
*/
