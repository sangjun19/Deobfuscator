#include<stdio.h>
int main(){
    int a;
    float w,h,b,h2,r;
    float ans2,ans3;
    
    printf("1. Rectangle\n2. Triangle\n3. Circle\nEnter your choice: ");
    scanf("%d",&a);
    if(a<=3 && a>=1){
    switch (a)
    {
    case 1 :
        printf("Enter the width: ");
        scanf("%f",&w);
        printf("Enter the height: ");
        scanf("%f",&h);
        if(w<0 || h<0 ){
            printf("The area of the rectangle is: Error");
        }else {printf("The area of the rectangle is: %.2f",w*h);}

        break;
    case 2:
        printf("Enter the base: ");
        scanf("%f",&b);
        printf("Enter the height: ");
        scanf("%f",&h2);
        ans2=b*h2*0.5;
        if(b<0 || h2<0){
            printf("The area of the triangle is: Error");
        }else {printf("The area of the triangle is: %.2f",ans2);}
        break;
    case 3:
        printf("Enter the radius: ");
        scanf("%f",&r);
        ans3=3.14 * r *r ;
        if( r<0){
            printf("The area of the circle is: Error");
        }else {printf("The area of the circle is: %.2f",ans3);}
        break;
    }
    }else if(a>=4){
        printf("Invalid choice");
    }
    return 0;
}

