#include<iostream>
#include<string>

using namespace std;

class Employee{

private :
    int empId;
    string empName;
    string empAdress;
    int result=1;
public :
    virtual double calculateSalary(){
        return 0;
    };
    void inputDetails(){
        cout<<"\nEnter Employee Details\n";
        cout<<"Name : ";
        cin>>empName;
        cout<<"ID : ";
        cin>>empId;
        getchar();
        cout<<"\nAdress :";
        getline(cin,empAdress);
    }
    void displayDetails(){
        cout<<"Name : "<<empName<<"\n";
        cout<<"ID : "<<empId<<"\n";
        cout<<"Adress : "<<empAdress<<"\n";
    }
};

class PermanentEmployee : public Employee{
private :
    double basicSalary;
    double hra;
    double da;
public:
    double calculateSalary(){
    double netSalary;
    netSalary=basicSalary+hra+da-800;
    return netSalary;
    } 
    void inputPE_Details(){
        cout<<"\nEnter Employee Details\n";
        cout<<"Basic Salary : ";
        cin>>basicSalary;
        cout<<"HRA : ";
        cin>>hra;
        cout<<"DA :";
        cin>>da;
    }
    void displayPE_Details(){
        cout<<"Basic Salary : "<<basicSalary<<"\n";
        cout<<"HRA : "<<hra<<"\n";
        cout<<"DA : "<<da<<"\n";
    }
};

class TemporaryEmployee : public Employee{
private:
    int noOfDay;
    double wagePerDay;
public:
    double calculateSalary(){
        double netSalary;
        netSalary=noOfDay*wagePerDay;
        return netSalary;
    }
    void inputTE_Details(){
        cout<<"No of days worked : ";
        cin>>noOfDay;
        cout<<"wage Per Day : ";
        cin>>wagePerDay;
    }
    void inputPE_Details(){
        cout<<"No of days worked : "<<noOfDay<<"\n";
        cout<<"DA : "<<wagePerDay<<"\n";
    }
};

int main(){
    PermanentEmployee PE;
    TemporaryEmployee TE;
    int choice;
    do{
        cout<<"Select from the options\n";
        cout<<"1. Salary of Permanent Employee\n";
        cout<<"2. Salary of Temporary Employee\n";
        cout<<"3. EXIT\n";
        cout<<"\nEnter your choice : ";
        cin>>choice; 

        switch (choice)
        {
        case 1:
            PE.inputDetails();
            PE.inputPE_Details();
            cout<<"Your Net Salary is "<<PE.calculateSalary()<<"\n;";
            system("pause");
            break;
        case 2:
            TE.inputDetails();
            TE.inputTE_Details();
            cout<<"Your Net Salary is "<<TE.calculateSalary()<<"\n;";
            system("pause");
            break;
        case 3:
            break;
        default:
            cout<<"\nINVALID INPUT\n";
            system("pause");
            break;
        } 
    }
    while( choice==1 || choice==2 );

    return 0;
}