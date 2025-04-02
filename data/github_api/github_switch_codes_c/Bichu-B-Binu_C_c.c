/* simple program*/
// #include <stdio.h>

// int main() {
//     // Write C code here
//     int a;
//     scanf("%d",&a);
//     printf("you have entered %d",a);
//     return 0;
// }
/*sum of a number*/

// #include <stdio.h>

// int main(void) {
//     // Write C code here
//  int num1,num2,sum;
//  printf("Enter two number:");
//  scanf("%d%d",&num1,&num2);
//  sum=num1+num2;
//  printf("Result:%d",sum);
//     return 0;
// }

/*Average of a number*/
// #include <stdio.h>

// int main(void) {
//     // Write C code here
// float num1,num2,num3,average;
// printf("Enter three number:");
// scanf("%f%f%f",&num1,&num2,&num3);
// average=(num1+num2+num3)/3;
// printf("result:%f",average);
//     return 0;
// }

/*Swapping*/

// #include <stdio.h>

// int main(void) {
//     // Write C code here
// int a=10,b=20,temp;
// temp=a;
// a=b;
// b=temp;
// printf("a:%d b:%d",a,b);


//     return 0;
// }


// #include <stdio.h>

// int main() {
//     // Write C code here
//     char a;
//     printf("Enter a char");
//     scanf("%c",&a);
//     printf("approved %c",a);

//     return 0;
// }

// #include <stdio.h>

// int main() {
    
//     int num1;
//     float num2, sum;
//     printf("Enter first number:");
//     scanf("%d",&num1);
//     printf("Enter second number:");
//     scanf("%f",&num2);
//     sum=num1+num2;
//     printf("result %f",sum);

//     return 0;
// }


// #include <stdio.h>

// int main() {
//     int P;
//     float R,n,SI;
//     printf("Enter the principal amount:");
//     scanf("%d",&P);
//     printf("Enter the intrest rate:");
//     scanf("%f",&R);
//     printf("Enter the number of years:");
//     scanf("%f",&n);
//     SI=(P*R*n)/100;
//     printf("Simple intrest is: %f",SI);
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int num;
//    printf("Enter a number:");
//    scanf("%d",&num);
//    if(num<0){
//        printf("The number is negative");
//    }else{
//        printf("The number is Positive");
//    }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int num1,num2;
//    printf("Enter first number:");
//    scanf("%d",&num1);
//    printf("Enter second number:");
//    scanf("%d",&num2);
//    if(num1<num2){
//        printf("Second number gratert than First number and the number is %d",num2);
//    }else{
//        printf("First number grater than second  number and the number is %d",num1);
//    }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int num1,num2,num3;
//    printf("Enter first number:");
//    scanf("%d",&num1);
//    printf("Enter second number:");
//    scanf("%d",&num2);
//    printf("Enter third number:");
//    scanf("%d",&num3);
//    if(num1<num2){
//        if(num2<num3){
//              printf("The gratest number is %d",num3);
//        }else{
//              printf("The gratest number is %d",num2);
//        }
//    }else{
//        if(num1<num3){
//              printf("The gratest number is %d",num3);
//        }else{
//              printf("The gratest number is %d",num1);
//        }
//    }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int num1,num2,choice,result;
//    printf("Enter first number:");
//    scanf("%d",&num1);
//    printf("Enter second number:");
//    scanf("%d",&num2);
//     printf("1 for addition \n2 for substration\n3 for multiplication \n4 for division \nEnter your choice:");
//     scanf("%d",&choice);
//     if(choice==1){
//         result=num1+num2;
       
//     }else if(choice==2){
//         result=num1-num2;
      
//     }else if(choice==3){
//         result=num1*num2;
        
//     }else if(choice==4){
//         result=num1/num2;
     
//     }else {
//         printf("Your not a human");
//     }
//        printf("Your result is %d",result);
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int food;
//    printf("\nItems \n1.Porotta \n2.Briyani \n3.Frid Rice \n4.Mandhi \nSelect a Item: ");
//    scanf("%d",&food);
//     // scanf("%d",&choice);
//    switch(food){
//        case 1:
//        printf("You are selected item is Porotta");
//        break;
//        case 2:
//        printf("You are selected item is Briyani");
//        break;
//        case 3:
//        printf("You are selected item is Frid Rice");
//        break;
//        printf("You are selected item is Mandhi");
//        break;
//        default:
//        printf("This item not avilable");
//    }
      
//     return 0;
// }


// Write a program to check whether a student has passed or failed in a subject after he or she enters their mark (pass mark for a subject is 50 out of 100).
// Program should accept an input from the user and output a message as “Passed” or “Failed”

// #include <stdio.h>

// int main() {
//    float mark;
//    printf("Enter the mark of student:");
//    scanf("%f",&mark);
//    if(mark>=50){
//       printf("Passed");
//    }else{
//        printf("Failed");
//    }
   
//     return 0;
// }


// Write a program to show the grade obtained by a student after he/she enters their total mark percentage.


// #include <stdio.h>

// int main() {
//     float totalMark;
//     printf("Enter the student mark:");
//     scanf("%f",&totalMark);
//     if(totalMark>=90){
//         printf("A Grade\n");
//     }else if(totalMark>=80){
//         printf("B Grade\n");
//     }else if(70<=totalMark){
//         printf("C Grade\n");
//     }else if(60<=totalMark){
//         printf("D Grade\n");
//     }else if(50<=totalMark){
//         printf("E Grade\n");
//     }else if(totalMark<=50){
//         printf("Failed");
//     }else{
//         printf("Your entered wroung data");
//     }
    
//     return 0;
// }




// Using the ‘switch case’ write a program to accept an input number from the user and output the day as follows. 


// #include <stdio.h>

// int main() {
//   int day;
//   printf("Pick one 1-7 days:");
//   scanf("%d",&day);
  
//   switch(day){
//       case 1:
//       printf("Sunday");
//       break;
//       case 2:
//       printf("Monday");
//       break;
//       case 3:
//       printf("Tuesday");
//       break;
//       case 4:
//       printf("Wednesday");
//       break;
//       case 5:
//       printf("Thursday");
//       break;
//       case 6:
//       printf("Friday");
//       break;
//       case 7:
//       printf("Saturday");
//       break;
//       default:
//       printf("Invalid Entry");
//   }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//    int i;
//    for(i=1;i<=100;i++){
//        printf("%d\n",i);
//    }

//     return 0;
// }

// #include <stdio.h>

// int main() {
//    int i,num,sum;
//    printf("Enter your number:");
//    scanf("%d",&num);
//    sum=0;
//    for(i=0;i<=num;i++){
//        sum=sum+i;
//    }
//        printf("Sum %d\n",sum);

//     return 0;
// }


// #include <stdio.h>

// int main() {
//     int num,i,flag=0;
//     printf("Enter a number:");
//     scanf("%d",&num);
    
//     for(i=2;i<num/2;i++){
//         if(num%i==0){
//             flag=1;
//             break;
//         }
      
//     }
//      if(flag==0){
//             printf("It is a prime number");
//         }else{
//             printf("It is not a prime number");
//         }
 
//     return 0;
// }


// #include <stdio.h>

// int main() {
//     int i,num,j;
   
//     printf("How many stars do you want enter a number:");
//     scanf("%d",&num);
//     for(i=1;i<=num;i++){
//         for(j=0;j<i;j++){
//             printf("*");
//         }
//         printf("\n");
//     }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//   int i,num,j;
//   printf("How many stars you want in reverse order:");
//   scanf("%d",&num);
//   for(i=1;i<=num;i++){
//       for(j=num;j>=i;j--){
//           printf("* ");
//       }
//       printf("\n");
//   }

//     return 0;
// }


// Write a program to print the multiplication table of given number
// Accept an input from the user and display its multiplication table
// Eg: 
// Output: Enter a number
// Input: 5
// Output: 
// 1 x 5 = 5
// 2 x 5 = 10
// 3 x 5 = 15
// 4 x 5 = 20
// 5 x 5 = 25
// 6 x 5 = 30
// 7 x 5 = 35
// 8 x 5 = 40
// 9 x 5 = 45
// 10 x 5 = 50

// #include <stdio.h>

// int main() {
//   int i,num;
//   printf("Enter your multiplication number:");
//   scanf("%d",&num);
//   for(i=1;i<=10;i++){
   
//      printf("%d*%d=%d\n",i,num,i*num);
//   }
  
//     return 0;
// }

// Write a program to find the sum of all the odd numbers for a given limit
// Program should accept an input as limit from the user and display the sum of all the odd numbers within that limit
// For example if the input limit is 10 then the result is 1+3+5+7+9 = 25
// Output: Enter a limit
// Input: 10
// Output: Sum of odd numbers = 25 


// #include <stdio.h>

// int main() {
//   int i,num,result=0;
//   printf("Enter a limit:");
//   scanf("%d",&num);
  
//   for(i=0;i<num;i++){
//       if(i%2!=0){
//           result=result+i;

//       }
//   }
//             printf("sum of odd number:%d\n",result);
//     return 0;
// }


// Write a program to print the following pattern (hint: use nested loop)
// 1
// 1 2
// 1 2 3
// 1 2 3 4
// 1 2 3 4 5

// #include <stdio.h>

// int main() {
//   int i,j,num;
//   printf("Enter a number:");
//   scanf("%d",&num);
//   for(i=1;i<=num;i++){
//       for(j=1;j<=i;j++){
//           printf("%d ",j);
//       }
//       printf("\n");
//   }
//     return 0;
// }


// #include <stdio.h>

// int main() {
//   int a[1000];
//   int i,limit;
  
  
//   printf("Enter a limit:");
//   scanf("%d",&limit);
  
//   printf("Enter values:\n");
//   for(i=0;i<limit;i++){
//       scanf("%d",&a[i]);
//   }
  
  
//   printf("Entered values:\n");
//   for(i=0;i<limit;i++){
//       printf("%d\n",a[i]);
//   }
  
  
//     return 0;
// }

// #include <stdio.h>

// int main() {
//   int a[1000];
//   int i,limit,sum=0;
  
//   printf("Enter a limit:");
//   scanf("%d",&limit);
  
//   printf("Enter values:\n");
//   for(i=0;i<limit;i++){
//       scanf("%d",&a[i]);
//   }
  
//   for(i=0;i<limit;i++){
//       sum=sum+a[i];
//   }
  
//   printf("Sum:%d",sum);
  
//     return 0;
// }


// #include <stdio.h>

// int main() {
//   int arr[1000];
//   int i,search,limit,flag;
//   printf("Enter a limit:");
//   scanf("%d",&limit);
  
//   printf("Enter values:\n");
//    for(i=0;i<limit;i++){
//       scanf("%d",&arr[i]);
//   }
//     printf("Enter your search Number:");
//   scanf("%d",&search);
  
 

//   for(i=0;i<limit;i++){
//       if(search==arr[i]){
//       flag=1;
//           break;
//       }
//   }
//    if(flag==1){
//           printf("Your position is :%d",i+1);
//       }else{
//           printf("Position not found");
//       }
//     return 0;
// }

// output
// Enter a limit:5
// Enter values:
// 12
// 46
// 82
// 46
// 73
// Enter your search Number:82
// Your position is :3


//Selection Sorting
// #include <stdio.h>

// int main() {
//     int arr[1000],i,j,limit,temp;
    
//     printf("Enter your limit:");
//     scanf("%d",&limit);
    
//     printf("Enter your Values:\n");
//     for(i=0;i<limit;i++){
//         scanf("%d",&arr[i]);
//     }
    
    
//     for(i=0;i<limit-1;i++){
//         for(j=i+1;j<limit;j++){
//             if(arr[i]>arr[j]){
//                 temp=arr[i];
//                 arr[i]=arr[j];
//                 arr[j]=temp;
//             }
//         }
        
//     }
//     printf("Sorted\n");
//     for(i=0;i<limit;i++){
//         printf("%d\n",arr[i]);
//     }
  
    
//     return 0;
// }
// Write a program to interchange the values of two arrays
// Program should accept an array from the user, swap the values of two arrays and display it on the console
// Eg: Output: Enter the size of arrays
// Input: 5
// Output: Enter the values of Array 1
// Input: 10, 20, 30, 40, 50
// Output: Enter the values of Array 2
// Input: 15, 25, 35, 45, 55
// Output: Arrays after swapping: 
// Array1: 15, 25, 35, 45, 55
// Array2: 10, 20, 30, 40, 50


// #include <stdio.h>

// int main() {
//     int i,j,limit,arr[1000],temp;
    
    
//     printf("Enter the limit:\n");
//     scanf("%d",&limit);
    
//     printf("Enter the Array 1 values:\n");
//     for(i=0;i<limit;i++){
//         scanf("%d",&arr[i]);
//     }
    
//      printf("Enter the Array 2 values:\n");
//     for(j=0;j<limit;j++){
//         scanf("%d",&arr[j]);
//     }
//             temp=arr[i];
//             arr[i]=arr[j];
//             arr[j]=temp;

//         printf("Array 1\n");
//         for(i=0;i<limit;i++){
//             printf("%d\n",arr[i]);
//         }
        
//        printf("Array 2\n");
//         for(j=0;j<limit;j++){
//             printf("%d\n",arr[j]);
//         }
  
    
    
//     return 0;
// }


// #include <stdio.h>

// int main() {
//     char name[100];
//     printf("Enter your name:");
//     scanf("%s",name);
//     printf("Your name is:%s",name);

//     return 0;
// }


#include <stdio.h>

int main() {
    int a[3][3],i,j;
    
    printf("Enter the metrix numbers:\n");
    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            scanf("%d",&a[i][j]);
        }
    }

    printf("Entered number is:\n");
    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            printf("%d\t",a[i][j]);
        }
        printf("\n");
    }
    return 0;
}