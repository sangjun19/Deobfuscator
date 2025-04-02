#include<iostream>
using namespace std;
int main(){

    char button;
    cout<<"Input a character : ";
    cin>>button;
    switch(button){
        case 'a':
         cout<<"yeah"<<endl;
         break;
        case 'b':
         cout<<"beah"<<endl;
         break;
        default:
         cout<<"nothing"<<endl;
         break;

    }
    return 0;
}