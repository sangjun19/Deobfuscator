#include<stdio.h>
void readarrays(int*,int,int);
void displayarrays(int*,int,int);
void sort(int*,int,int,int);
void binarysearch(int*,int,int,int,int);
void linearsearch(int*,int,int);
void mean(int*,int,int);
void mideum(int*,int,int);
int  lar(int*,int,int);
void mode(int*,int,int,int);
void selar(int*,int,int);
void sm(int*,int,int);
void sesm(int*,int,int);
void matrix(int*,int,int);
void marging(int*,int,int,int);
int main(void)
{
     int n,i,j,key,skey,large,m;
     printf("Enter number of arrays:\t");
     scanf("%d",&n);
     int A[n];
     int option;
     do
     {
         printf("\n************Main Menu*************");
         printf("\n 1. Read the arrays");
         printf("\n 2. Print the arrays");
         printf("\n 3. Sort the arrays");
         printf("\n 4. Binary search the arrays");
         printf("\n 5. Linear search the arrays");
         printf("\n 6. mean number in the arrays");
         printf("\n 7. mideum number in the arrays");
         printf("\n 8. Largest number in the arrays");
         printf("\n 9. Mode number in the arrays");
         printf("\n 10. Second largest number in the arrays");
         printf("\n 11. Smallest number in the array");
         printf("\n 12, Second Smallest number in the array");
         printf("\n 13. Merging arrays");
         printf("\nEnter the option:\t");
         scanf("%d",&option);
         switch(option)
         {
              case 1 : readarrays(A,n,0);
                        break;
              case 2 : printf("The element in arrays is :\t");
                        displayarrays(A,n,0);break;
              case 3 :  sort(A,n,i,j);break;
              case 4 :  printf("\nEnter the element to search :\t");
                          scanf("%d",&key);
                            binarysearch(A,n,0,n-1,key);
                            break;
              case 5 :  printf("\nEnter the element to search :\t");
                          scanf("%d",&skey);
                          linearsearch(A,n,skey);
                          break;
              case 6:  mean(A,n,i);break;
              case 7:  mideum(A,n,i);break;
              case 8:  printf("The largest number is  %d",lar(A,n,i));break;
              case 9:  mode(A,n,i,j);break;
              case 10: selar(A,n,i);break;
              case 11: sm(A,n,i);break;
              case 12: sesm(A,n,i);break;
              case 13: marging(A,i,j,n);break;
         }
     }while(option<=13);
}
void readarrays(int A[],int n,int i)
{
    if(i<n)
    {
       printf("A[%d]=",i);
       scanf("%d",&A[i]);
       readarrays(A,n,i+1);
    }
}
void displayarrays(int A[],int n,int i)
{
    if(i<n)
    {
        printf("%d\t",A[i]);
        displayarrays(A,n,i+1);
    }
}
void sort(int A[],int n,int i,int j)
{
   int temp;
   printf("\nThe sorted element is :\t");
    for(j=0 ; j < n-1 ; j++){
        for(i=0;i<n;i++)
          {
               if(A[i]>A[i+1])
               {
                    temp=A[i];
                    A[i]=A[i+1];
                    A[i+1]=temp;
               }
          }
    }
     displayarrays(A,n,0);
}
void binarysearch(int A[],int n,int i,int j,int key)
{
     int mid=(i+j)/2;
     if(key==A[mid])
     {
         printf("\nFound element that you search at %d.",mid);
     }
     else if(key>A[mid])
     {
         binarysearch(A,n,mid+1,j,key);
     }
     else{
         binarysearch(A,n,i,mid-1,key);
     }

}
void linearsearch(int A[],int n,int skey)
{
     if(A[n-1]==skey)
     {
         printf("The element found at %d .",n-1);
         linearsearch(A,n-1,skey);

     }
}
void mean(int A[],int n,int i)
{
     int a=0;
     if(i<n)
     {
          a+=A[i];
     }
     printf("The mean number is %d:\t",(float)a/n);
}
void mideum(int A[],int n, int i)
{
     if(n%2==0)
     {
         printf("The mideum is :\t %d",(float)(A[n/2]+A[n/2+1])/2);
     }
     else
     {
         printf("The mideum is :\t %d",(float)A[n/2]);
     }
}
int lar(int A[],int n,int i)
{
     int large= A[0];
     for(i=0;i<n;i++)
     {
          if(A[i]>large)
          {
             large=A[i];
          }
     }
     return large;
}
void mode(int A[],int n,int i,int j)
{
    int B[10];
    int z = lar(A,n,i);
    for(i=0;i<=z;i++)
    {
        B[i]=0;
    }
    for(j=0;j<n;j++)
    {
        ++B[A[j]];
    }
    int mode;
    int h=B[0];
    for(i=0;i<=z;i++)
    {
        if(B[i]>h)
        {
           h=B[i];
           mode=i;
        }
    }
    for(i=0;i<=z;i++)
    {
        printf("\n %d count %d",i,B[i]);
    }
    printf("\nThe max number is %d",mode);
    printf("\n The mode number is %d\n",h);
    for(i=0;i<=z;i++)
    {
         if(B[i]!=0)
         {
             printf(" %d \t",i);
         }
    }
}
void selar(int A[],int n,int i)
{
    int large=A[0];
    for(i=0;i<n;i++)
    {
          if(A[i]>large)
          {
              large=A[i];
          }
    }
    int second_large=A[1];
    for(i=0;i<n;i++)
    {
          if(A[i]!=large && A[i]>second_large)
          {
               second_large= A[i];
          }
    }
    printf("The second largest number is %d",second_large);
}
void sm(int A[],int n,int i)
{
   int small = A[0];
   for(i=0;i<n;i++)
   {
        if(A[i]<small)
        {
            small = A[i];
        }
   }
   printf("The smallest number is %d",small);
}
void sesm(int A[],int n,int i)
{
   int small = A[0];
   for(i=0;i<n;i++)
   {
        if(A[i]<small)
        {
            small = A[i];
        }
   }
   int sesmall= A[1];
   for(i=0;i<n;i++)
   {
         if(A[i]!=small && A[i]<sesmall)
         {
              sesmall=A[i];
         }
   }
   printf("The second smallest number is %d",sesmall);
}
void marging(int A[],int i,int j,int n)
{
   displayarrays(A,n,i);
   int n1;
   printf("\nEnter the number of elements in arrays 2:\t");
   scanf("%d",&n1);
   int B[n1];
   readarrays(B,n1,0);
   displayarrays(B,n1,0);
   int m;
   m = n + n1;
   int A1[m];
   i=0;
   for(j=0;j<n;j++)
   {
        A1[i]=A[j];
        i++;
   }
   for(j=0;j<n1;j++)
   {
        A1[i]=B[j];
        i++;
   }
   printf("\nThe marging arrays is ");
   displayarrays(A1,m,0);
}
