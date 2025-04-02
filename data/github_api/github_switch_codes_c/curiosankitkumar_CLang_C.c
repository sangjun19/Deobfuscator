#include <stdio.h>
/*int main(){
    int age;
    printf("enter age :");
    scanf("%d", &age);

    if (age >= 18){
        printf("adult \n");
    }
    else if (age >13 && age <18){
        printf("teenager \n");
    }
    else{
        printf("child");

    }
    return ;
}
*/
 //ternary operators

 /*int main (){
     int age ;
     printf("enter age:");
     scanf("%d",&age);

     age >=18 ? printf("adult \n") : printf("not adult");
     return 0;
 }
*/
 /*int main(){
     int day; //m-mon; t-tue; w- wed
     printf("enter day(1-7):");
     scanf("%s", &day);

     switch (day) {
         case 'm' : printf("moday \n");
                  break;
         case 't' : printf("tuesday \n");
                  break;
         case 'w' : printf("wednesday \n");
                  break;
         case 'T' : printf("thursday \n");
                  break;
         case 'f' : printf("friday \n");
                  break;
         case 's' : printf("saturday \n");
                  break;
         case 'S' : printf("sunday \n");
                  break;
         default :printf("not a valid day! \n");


     }
 }
*/
/*int main (){
    int marks;
    printf("enter number(0-100):");
    scanf("%d", &marks);

 /*   if(marks >= 0 && marks <= 30){
        printf("FAIL \n");
    } else if (marks >30 && marks <= 100) {
        printf("pass \n");
    } else{
        printf("wrong marks");
    }
    */
    /*
    //ternary opertaors solve
    marks <= 30? printf("FAIL \n") : printf("PASS \n");
    return 0;
}
*/

/*int main(){
    int marks;
    printf("enter number(0-100): ");
    scanf("%d", &marks);

    if (marks <30) {
        printf("C \n");
    }
    else if (marks >= 30 && marks <70) {
        printf("B \n");
    }
    else if (marks >= 70 && marks <90){
        printf("A \n");
    }
    else {
        printf("A+ \n");
    }
    return 0;
}
*/

/*int main (){
    int x =2 ;

    if (x=4) {
        printf("x is equal to 1 \n");
        printf("%d \n", x);
    } else {
        printf("x is not equal to 1");
    }
    return 0;
}
*/
/*int main (){
    char ch;
    printf("enter character : ");
    scanf("%c", & ch);

    if (ch >= 'A' && ch <= 'Z'){
        printf("upper case \n");
    }
    else if(ch >= 'a' && ch <= 'z'){
        printf("lower case \n");
    }
    else{
        printf("not english letter \n");
    }
    return 0;
}
*/

// Q-- Write a C program to find maximum between two numbers.

/*int main(){
    int num1 , num2;

    printf("Enter a numbers :");
    scanf("%d%d", &num1 ,&num2);

    if (num1 > num2){
        printf("%d is maximum" , num1);
    }
    if(num2 > num1){
        printf("%d is minimum", num2);
    }
    if(num1 == num2){
        printf("is equal to");
    }
    return 0;
}
*/
//Write a C program to find maximum between three numbers.

/*int main(){
    int num1, num2, num3, max;
    printf("enter three numbers :");
    scanf("%d%d%d" ,&num1, &num2 ,&num3);

    if (num1 > num2){
        if (num1 > num3){
            max = num1;
        }
        else{
            max = num3;
        }
    }
    else {
        if(num2 > num3){
            max = num2;
        }
        else {
            max = num3;
        }
    }
    printf("Maximum among all three numbers = %d", max);
    return 0 ;
}
*/
// Q--Write a C program to check whether a number is negative, positive or zero.

/*int main (){
    int num;
    printf("Enter any numbers : ");
    scanf("%d", &num);

    if (num < 0){
        printf("Number is NEGATIVE : ");

    }
    if (num > 0){
            printf("Number is POSITIVE :");

    }
    if (num == 0){
        printf("Number is ZERO : ");
    }
    return 0 ;

}
*/

//Q--Write a C program to check whether a number is divisible by 5 and 11 or not.

/*int main (){
    int num ;
    printf("Enter any number : ");
    scanf("%d"  , &num);

    if ((num % 5 == 0) && (num % 11 == 0)){
            printf("num is divisible by 5 and 11 ");

    }
    else {
        printf("Number is not divisible by 5 and 11 ");

    }
    return 0;
}
*/
// Q--Write a C program to check whether a number is even or odd.

/*int main (){
    int n ;
    printf("enter any numbers ");
    scanf("%d" , &n);

    if (n % 2 ==0){
        printf("n is even");
    }
    else {
        printf("n is odd");
    }
}
*/
//Q-- Write a C program to check whether a number is even or odd.

/*int main(){
    int year;
    printf("Enter any year ");
    scanf("%d", &year);

    if ((year % 4 == 0) && (year % 100 != 0) || (year % 400 == 0))
    {

        printf("LEEP YEAR");
    }
    else{
        printf("COMMON YEAR");
    }
        return 0;
}
*/
// Q-- Write a C program to check whether a character is alphabet or not.

/*int main () {
    char ch ;
    printf("Enter any ch ");
    scanf("%c" , & ch);

    if ((ch >= 'a' && ch <= 'z') ||(ch >= 'A' && ch <= 'Z')) {
        printf("Character is an ALPHABET.");
    }
    else {
        printf("Character is NOT ALPHABET");
    }
    return 0;
}
*/

// Q--  Write a C program to input any alphabet and check whether it is vowel or consonant.

/*int main (){
    char ch;
    printf("Enter any character ");
    scanf("%c" , &ch);

    if((ch == 'A') || (ch == 'E') || (ch == 'I') || (ch == 'O') || (ch == 'U') ||
       (ch == 'a') || (ch == 'e') || (ch == 'i') || (ch == 'o') || (ch == 'u')) {
       printf("'%d' is VOWEL .",ch);

       }
       else if (ch >= '0' && ch <= '10'){
            printf("'%c' is a digit .",ch );
       }
       else {
        printf("'%c' is special charcter.", ch);
       }

       return 0;
}

*/

// Q-- Write a C program to check whether a character is uppercase or lowercase alphabet.

/*int main (){
    int ch;
    printf("Enter any charcter ");
    scanf("%c", &ch);

    if (ch >= 'A' && ch <= 'Z'){
        printf("'%c' is upper case. ", ch);
    }
    else if (ch >= 'a' && ch <= 'z'){
        printf("'%c' is lower case. ", ch);
    }
    else {
        printf("'%c' none of these. ", ch);
    }
    return 0;
}
*/

// Q--Write a C program to input week number and print week day.

/*int main (){
    int week;
    printf("Enter any week (1-7): ");
    scanf("%d" , & week );

    if (week == 1){
        printf("Monday");
    }
    else if (week == 2){
        printf("Tuesday");
    }
    else if (week == 3){
        printf("Wednesday");
    }
    else if (week == 4){
        printf("Thrusday");
    }
    else if (week == 5){
        printf("Friday");
    }
    else if (week == 6) {
        printf("Saturaday");
    }
    else if (week == 7) {
        printf("saunday");
    }
    else {
        printf("Nome of  these");
    }


    return 0;
}
*/
// SECOND METHOD.

/*
int main()
{

    const char * WEEKS[] = { "Monday", "Tuesday", "Wednesday",
                            "Thursday", "Friday", "Saturday",
                            "Sunday"};
    int week;


    printf("Enter week number (1-7): ");
    scanf("%d", &week);

    if(week > 0 && week < 8)
    {

        printf("%s", WEEKS[week-1]);
    }
    else
    {
        printf("Invalid input! Please enter week number between 1-7.");
    }

    return 0;
}
*/

// Q--Write a C program to input month number and print number of days in that month.

/*int main (){

    int month ;
     printf("Enter any  month (1-12):");
     scanf("%d" , & month);


    if (month == 1){
        printf("31 days");
    }
    else if (month == 2){
        printf("30 days");
    }
    else if (month  == 3){
        printf("31 days");
    }
    else if (month == 4){
        printf("30 days");
    }
    else if (month == 5){
        printf("31 days");
    }
    else if (month == 6) {
        printf("30 days");
    }
    else if (month == 7) {
        printf("31 days");
    }
     if (month == 8){
        printf("31 days");
    }
    else if (month == 9){
        printf("30 days");
    }
    else if (month  == 10){
        printf("31 days");
    }
    else if (month == 11){
        printf("30 days");
    }
    else if (month == 12){
        printf("31 days");
    }

    else {
        printf("None of  these");


    return 0 ;
   }
}
*/

// SECOND METHOD

/*int main()
{
    int month;


    printf("Enter month number (1-12): ");
    scanf("%d", &month);



    if(month==1 || month==3 || month==5 || month==7 || month==8 || month==10 || month==12)
    {
        printf("31 days");
    }
    else if(month==4 || month==6 || month==9 || month==11)
    {

        printf("30 days");
    }
    else if(month==2)
    {
        printf("28 or 29 days");
    }
    else
    {
        printf("Invalid input! Please enter month number between (1-12).");
    }

    return 0;
}
*/

// Q-- Write a C program to count total number of notes in given amount.

/*int main (){
    int amount;
    int note500, note100, note50, note20, note10, note5, note2, note1;
    printf("Enter amount: ");
    scanf("%d" , &amount);

    if(amount >= 500)
    {
        note500 = amount/500;
        amount -= note500 * 500;
    }
    if(amount >= 100)
    {
        note100 = amount/100;
        amount -= note100 * 100;
    }
    if(amount >= 50)
    {
        note50 = amount/50;
        amount -= note50 * 50;
    }
    if(amount >= 20)
    {
        note20 = amount/20;
        amount -= note20 * 20;
    }
    if(amount >= 10)
    {
        note10 = amount/10;
        amount -= note10 * 10;
    }
    if(amount >= 5)
    {
        note5 = amount/5;
        amount -= note5 * 5;
    }
    if(amount >= 2)
    {
        note2 = amount /2;
        amount -= note2 * 2;
    }
    if(amount >= 1)
    {
        note1 = amount;
    }

        printf("Total number of notes = /n " );
        printf("500 = %d\n", note500);
        printf("100 = %d\n", note100);
        printf("50 = %d\n", note50);
        printf("20 = %d\n", note20);
        printf("10 = %d\n", note10);
        printf("5 = %d\n", note5);
        printf("2 = %d\n", note2);
        printf("1 = %d\n", note1);

        return 0;
    }
*/

// Q-- Write a C program to input angles of a triangle and check whether triangle is valid or not.

/*
int main () {
    int triangle ;
    printf(" Enter any number ");
    scanf("%d" , &triangle);

    if (triangle == 180)
    {
        printf("triangle is Valid" );
    }
    else if (triangle != 180)
    {
        printf("triangle is not valid ");
    }
    return 0;
}
*/
 // MAIN METHOD
/*
int main (){
    int angle1, angle2, angle3, sum;
    printf("Enter three angle of a triangle: \n");
    scanf("%d%d%d" ,&angle1, &angle2, &angle3);

    sum = angle1 + angle2 + angle3;

    if (sum == 180 && angle1 >= 0 && angle2 >= 0 && angle3 >= 0)
    {
        printf("Triangle is valid");
    }
    else
    {
        printf("Triangle i not valid");
    }
    return 0;
}
*/

// Q--Write a C program to input all sides of a triangle and check whether triangle is valid or not.

/*int main()
{
    int sde1, sde2, sde3;
    printf("enter 1st sides: ");
    scanf("%d",&sde1);

    printf("enter 2nd sides: ");
    scanf("%d",&sde2);

    printf("enter 3rd sides: ");
    scanf("%d",&sde3);
    if((sde1+sde2)>sde3 && (sde2+sde3)>sde1 && (sde1+sde3)>sde2){
        printf("valid triangle");
    }
    else{
        printf("invalid triangle");
    }
    return 0;
}
*/

// Q-- Write a C program to check whether the triangle is equilateral, isosceles or scalene triangle.

/*int main () 
{
    int side1, side2, side3;
    printf("enter 1st sides: ");
    scanf("%d", &side1);

    printf("enter 2nd sides: ");
    scanf("%d", &side2);

    printf("enter 3rd sides: ");
    scanf("%d", &side3);

    if ((side1 == side2) && (side2 == side3)) 
    {
        printf("The triangle is equilateral.");
    }
    else if ((side1 == side2) || (side1 == side3) || (side2 == side3)) 
    {
        printf("The triangle is isosceles");
    }
    else 
    {
        printf("The triangle is scalene");
    }
*/

// Q-- Write a C program to find all roots of a quadratic equation.


/*#include <stdio.h>
#include <math.h>

int main()
{
    int a, b, c;
    printf("a: ");
    scanf("%d", &a);

    printf("b: ");
    scanf("%d", &b);

    printf("c: ");
    scanf("%d", &c);

    float disc = pow(b, 2) - 4 * a * c;

    if (disc == 0)
    {
        printf("both roots are real and equal: %f and %f", (float)(-b) / (2 * a), (float)(-b) / (2 * a));
    }
    else if (disc > 0)
    {
        printf("both roots are real and different: %f and %f", (-b + sqrt(disc)) / (2 * a), (-b - sqrt(disc)) / (2 * a));
    }
    else
    {
        float realPart = (float)(-b) / (2 * a);
        float imaginaryPart = sqrt(-disc) / (2 * a);
        printf("both roots are complex: %f + %fi and %f - %fi", realPart, imaginaryPart, realPart, imaginaryPart);
    }

    return 0;
}
*/

// Q--Write a program to check whether a given character is Alphabet or not using if else statement 

// Note: Check for both upper and lower case characters
/*
#include<stdio.h>

 int main (){
    char ch;
    printf("Enter a character:");
    scanf("%c" , &ch);

    if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'))
    {
        printf("%c is an alphabet." , ch );
    }
    else{
        printf("%c is an not alphabet.", ch);
    }
    return 0;
}
*/

// Q-- Write a C program to calculate profit or loss.
/*
#include<stdio.h>
#include<math.h>

int main ()
{
    int cp , sp , amt ;
    printf("Enter cost price : ");
    scanf("%d", &cp);

    printf("Enter selling price : ");
    scanf("%d", &sp);


    if (sp > cp)   {
        amt = cp - sp ;
        printf("profit = %d" , amt);

    }
    else if (cp > sp)
    {
        amt = sp - cp ; 
        printf("loss = %d" , amt);
    }
    else {
        printf("no profit no loss");
    }
}
*/

//Q-- Write a C program to input marks of five subjects Physics, Chemistry, Biology, Mathematics and Computer.
// Calculate percentage and grade according to following:

// Percentage >= 90% : Grade A
// Percentage >= 80% : Grade B
// Percentage >= 70% : Grade C
// Percentage >= 60% : Grade D
// Percentage >= 40% : Grade E
// Percentage < 40% : Grade F

/*
#include<stdio.h>
#include<math.h>

int main (){
    float Phy;
    float Che;
    float Bio;
    float Mth;
    float com;

    float per=0;
    printf("enter five subject marks: ");
    scanf("%f \n", &Phy);
    scanf("%f \n", &Che);
    scanf ("%f \n", &Bio);
    scanf("%f \n", &Mth);
    scanf("%f \n", &com);
    printf("%f %f %f %f %f %f\n",per,Phy,Che,Bio,Mth,com);
    per = (Phy + Che + Bio + Mth + com) / 5.0;

    printf("percentage = %f\n", per);
    
    if (per >= 90){
        printf("Garde A");
    }
    else if (per >= 80){
        printf("Grade B");
    }
    else if (per >= 70){
        printf("Grade C");
    }
    else if (per >= 60){
        printf("Grade D");
    }
    else if (per >= 40){
        printf("Grade E");
    }
    
    else {
        printf("Grade A++");
    }
    
    return 0;

}
*/

// Q-- Write a C program to input basic salary of an employee and calculate its Gross salary according to following:
// Basic Salary <= 10000 : HRA = 20%, DA = 80%
// Basic Salary <= 20000 : HRA = 25%, DA = 90%
// Basic Salary > 20000 : HRA = 30%, DA = 95%

/*

#include<stdio.h>

int main (){
    float gross, basics, hra, da;
    printf("Enter a basics salary of a employee: ");
    scanf("%f", &basics);

    if (basics <= 10000)
    {
        da = basics * 0.8;
        hra = basics * 0.2;
    }
    else if (basics <= 20000)
    {
        da = basics * 0.9;
        hra = basics * 0.25;
    }
    else 
    {
        da = basics * 0.95;
        hra = basics * 0.3;

        gross = basics + hra + da;
        printf("GROSS SALARY OF EMPLOYEE = %.2f",gross);
    }
    return 0 ;
}
*/
// Q--Write a C program to input electricity unit charges and calculate total electricity bill according to the given condition:
// For first 50 units Rs. 0.50/unit
// For next 100 units Rs. 0.75/unit
// For next 100 units Rs. 1.20/unit
// For unit above 250 Rs. 1.50/unit
// An additional surcharge of 20% is added to the bill

#include<stdio.h>

int main () 
{
    int unit ;
    float amt, total_amt , sur_charge;

    printf("Enter total units consumed: ");
    scanf("%d", &unit);

    if (unit <= 50) 
    {
        amt = unit * 0.50;
    }
    else if (unit <= 150);
    {
        amt = 25 + ((unit-50) * 0.75);   
    }
    else if(unit <= 250)
    {
        amt = 100 + ((unit-150) * 1.20);
    }
    else
    {
        amt = 220 + ((unit-250) * 1.50);
    }

    /*
     * Calculate total electricity bill
     * after adding surcharge
     */
    sur_charge = amt * 0.20;
    total_amt  = amt + sur_charge;

    printf("Electricity Bill = Rs. %.2f", total_amt);

    return 0;
}




















