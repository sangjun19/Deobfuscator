//
//  main.c
//  Question 4
//
//  Created by Aidan Law on 17/06/2014.
//  Copyright (c) 2014 Aidan Law. All rights reserved.
//

#include <stdio.h>

int main(void)
{
    int num;
    
    printf("Guess a whole number between -6 and 6: ");
    scanf("%d", &num);
    
    printf("The statement on the next line is found using if and else statements\n");
    
    if (num==0)
    {
        printf("Correct\n");
    }
    
    else if (num==1 || num==-1)
    {
        printf("Hot\n");
    }
    
    else if (num==-3 || num==-2 || num==2 || num==3)
    {
        printf("Warm\n");
    }
    
    else if (num==-5 || num==-4 || num==4 || num==5)
    {
        printf("Cold\n");
    }
    
    else
    {
        printf("Not a valid input\n");
    }
    
    
    // Can also be done with case statements
    
    printf("The statement on the next line is found using switch case statements\n");
    
    switch (num)
    {
        case 0:
            printf("Correct\n");
            break;
        case -1: case 1:
            printf("Hot\n");
            break;
        case -3: case -2: case 2: case 3:
            printf("Warm\n");
            break;
        case -5: case -4: case 4: case 5:
            printf("Cold\n");
            break;
        default:
            printf("Not a vlaid input\n");
            break;
            
            
    }
    
    
    return 0;
}

