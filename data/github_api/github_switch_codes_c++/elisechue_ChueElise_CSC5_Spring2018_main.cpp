/* 
 * File:   main.cpp
 * Author: Elise Chue
 * Created on March 13, 2018, 10:25 AM
 * Purpose:  all kinds of sums using switch operator
 */

//System Libraries Here
#include <iostream>

using namespace std;

//User Libraries Here

//Global Constants Only, No Global Variables
//Math, Physics, Science, Conversions, 2-D Array Columns

//Function Prototypes 

//Program Execution Begins Here

int main(int argc, char** argv) {
   //Declare Variables
    short sumPos, sumNeg, sumTot, x;
    
    //Input or initialize values
    sumPos=sumNeg=sumTot=0; //need to do this for probs like this so that
                            //the garbage memory of a PC doesnt override it
    
    //Calculate the answer
    cout<<"This program sums 10 negative or positive integers"<<endl;
    cout<<"Utilize numbers between -100 and 100"<<endl;
    
    cout<<"Input First Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Second Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Third Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }  
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
        
    cout<<"Input Fourth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    } 
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Fifth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }    
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Sixth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }    
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
        
    cout<<"Input Seventh Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    } 
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Eighth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }  
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
            
    cout<<"Input Ninth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    }
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    cout<<"Input Tenth Integer ->"<<endl;
    cin>>x;
    switch(x>0){
        case true:sumPos+=x;break;
        default: sumNeg+=x;    
    } 
    //sumPos+=(x>0?x:0);
    //sumNeg+=(x<0?x:0);
    sumTot+=x;
    
    //Display Outputs
    cout<<"The positive sum = "<<sumPos<<endl;
    cout<<"The negative sum = "<<sumNeg<<endl;
    cout<<"The total sum    = "<<sumTot<<endl;
    //Exit Program!
    
    return 0;
}

