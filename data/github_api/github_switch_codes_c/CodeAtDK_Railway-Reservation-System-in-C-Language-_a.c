#include <stdio.h>

void AJMER_SHATABDI(){

    FILE *filePointer; 

    filePointer = fopen("Untitled-1.txt","a");
    fprintf(filePointer,"                              Ticket Detail\n");
    fprintf(filePointer,"Departure                                                  Arrival\n");
    fprintf(filePointer,"  16:10                                                    21:40\n");
    fprintf(filePointer,"PNR                                             Train no./Name                      Class\n");
    fclose(filePointer);

    



    printf("12015  AJMER SHATABDI\n");
    printf("Departure             Arrival\n");
    printf("  16:10                21:40\n");
    printf("  CC      EC\n");
    printf("  AVL     AVL\n");
    printf("  30      30\n");
    printf("Rs.700   Rs.800\n");


}

int AJMER_SHATABDI_Booking(){

    int class,fare;

    FILE *filePointer; 
    filePointer = fopen("Untitled.txt","a");

    printf("Select class1\n");
    scanf("%d",&class);

    switch (class){
        
        case 1:

            printf("you have selected CC\n");
            fare = 700;
            fprintf(filePointer,"%d                                           12986   DEE_JP_DOUBLE_DECKER                  (AC) CC \n");
            return fare;
            break;

        case 2:

            printf("you have selected EC\n");
            fare = 800;
            fprintf(filePointer,"%d                                           12986   DEE_JP_DOUBLE_DECKER                  (AC) EC \n");
            return fare;
            break;


        /*case 1:

            printf("you have selected 1A\n");
            fare = 5342;
            fprintf(filePointer,"%d                                            092347  CSMT Rajdhani                  (AC) 1A \n");
            return fare;
            break;
            */


        default:

            printf("Your selected option is not avl\n");

    }
    fclose(filePointer);

}




 


void DEE_JP_DOUBLE_DECKER(){

    FILE *filePointer; 

    filePointer = fopen("Untitled-1.txt","a");
    fprintf(filePointer,"                              Ticket Detail\n");
    fprintf(filePointer,"Departure                                                  Arrival\n");
    fprintf(filePointer,"  17:00                                                    22:00\n");
    fprintf(filePointer,"PNR                                             Train no./Name                      Class\n");
    fclose(filePointer);

    



    printf("12985   DEE_JP_DOUBLE_DECKER  \n");
    printf("Departure             Arrival\n");
    printf("  16:10                21:40\n");
    printf("  CC      EC\n");
    printf("  AVL     AVL\n");
    printf("  30      30\n");
    printf("Rs.700   Rs.800\n");


}





int DEE_JP_DOUBLE_DECKER_Booking(){

    int class,fare;

    FILE *filePointer; 
    filePointer = fopen("Untitled.txt","a");

    printf("Select class1\n");
    scanf("%d",&class);

    switch (class){
        
        case 1:

            printf("you have selected CC\n");
            fare = 700;
            fprintf(filePointer,"%d                                           12986   DEE_JP_DOUBLE_DECKER                  (AC) CC \n");
            return fare;
            break;

        case 2:

            printf("you have selected EC\n");
            fare = 800;
            fprintf(filePointer,"%d                                           12986   DEE_JP_DOUBLE_DECKER                  (AC) EC \n");
            return fare;
            break;


        /*case 1:

            printf("you have selected 1A\n");
            fare = 5342;
            fprintf(filePointer,"%d                                            092347  CSMT Rajdhani                  (AC) 1A \n");
            return fare;
            break;
            */


        default:

            printf("Your selected option is not avl\n");

    }
    fclose(filePointer);

}


