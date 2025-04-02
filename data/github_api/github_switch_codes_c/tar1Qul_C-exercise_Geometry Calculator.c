#include <stdio.h>
int main(){
    char shape;
    printf("Shape(Circle/Square/Rectangle):");
    scanf("%c",&shape);

    switch(shape){
        case'C':
        printf("Enter Radius:");
        int radius;
        double Carea;
        scanf("%lf",&radius);
        Carea=3.1416*radius*radius;
        printf("Area=%.2f",Carea);break;

        case 'S':
        printf("Enter lenght:");
        double lenght,areas;
        scanf("%lf",&lenght);
        areas=lenght*lenght;
        printf("Area=%lf",areas);break;

        case 'R':
        printf("Enter height and width(10 5):");
        double h,w,arear;
        scanf("%lf %lf",&h,&w);
        arear=h*w;
        printf("Area=%lf",arear);break;

        default:printf("Invalid");
        }
  return 0;
}
