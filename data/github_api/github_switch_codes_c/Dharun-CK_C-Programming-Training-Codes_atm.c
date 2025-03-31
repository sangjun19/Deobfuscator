#include<stdio.h>
int main()
{
    long int accno= 732121205016,eaccno ,bal = 120000,nbal, opin=7321,  epin , npin , rpin, o,amm;
    printf("****************************************************************************");
    printf("\n");
    printf("\nA*T*M     TRANSACTIONS ");
    printf("\n****************************************************************************");
    printf("\nEnter the Account Number:");
    scanf("%li",&eaccno);
    if(eaccno!=accno)
    {
        printf("***Please Check the Account Number***");
        
    }
    else
    {
        
        printf("Enter the PIN:");
        scanf("%li",&epin);
        if(epin!=opin)
        {
            printf("Invalid Pin!");
            
        }
        else
        {
        printf(" The account Number IS %li\n",eaccno);
        printf("The Balance is %li\n",bal);
        printf("\n");
        printf("\n1 - WITHDRAW\n2- DEPOSIT\n3- BALANCE ENQUIRY\n4- PIN CHANGE");
        printf("\nEnter THE Opinion:");
        scanf("%li",&o);
        
        switch(o){
            case 1:
            if(amm<bal)
            {
            printf("Enter the Amount:");
            scanf("%li",&amm);
            nbal = bal-amm;
            printf("THE NEW BALANCE is %li",nbal);
            }
            else 
            {
                printf("AMOUNT EXCEEDED!");
            }
            break;
           
            case 2:
            printf("Enter the Amount:");
            scanf("%li",&amm);
            nbal = bal+amm;
            printf("THE NEW BALANCE is %li",nbal);
            break;
           
            case 3:
            printf("\nVerifying...........");
            printf("\nAccount Number :%li", accno);
            printf("\nBalance : %li",bal);
            printf("\nThe Balance is Verified! properly");
            break;
            
            
            case 4:
            printf("Enter The pin: ");
            scanf("%li",&epin);
            if(epin==opin)
            {
            printf("Enter THe Old pin : ");
            scanf("%li",&epin);
            if(opin == epin)
            {
                printf("Enter The New Pin :");
                scanf("%li",&npin);
                printf("Re-enter The Pin:");
                scanf("%li",&rpin);
                if(npin==rpin)
                printf("Your New pin is %li",npin);
                else
                printf("MISSMATCHED");
                
            }
          else
            {
                printf("Incorrect Pin!");
            }
            }
        }
            
        }
        
        
    }
return 0;
}