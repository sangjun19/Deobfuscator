// Repository: joxmode/C-calculator
// File: calculatorC++.cpp

#include <iostream>
using namespace std;

int main() {
    int choice; 
    int num1, num2;

    cout << "Choose a mathematical operation:\n";
    cout << "1 for plus\n";
    cout << "2 for minus\n";
    cout << "3 to multiply\n";
    cout << "4 to divide\n";

    cin >> choice; // Read user choice

    if (choice >= 1 && choice <= 4) { 
        cout << "Enter first number: ";
        cin >> num1;

        cout << "Enter second number: ";
        cin >> num2;

        switch (choice) {
        case 1: // Addition
            cout << "Answer = " << (num1 + num2) << endl;
            break;
        case 2: // Subtraction
            cout << "Answer = " << (num1 - num2) << endl;
            break;
        case 3: // Multiplication
            cout << "Answer = " << (num1 * num2) << endl;
            break;
        case 4: // Division
            if (num2 != 0) {
                cout << "Answer = " << (num1 / num2) << endl;
            }
            else {
                cout << "Error: Division by zero is not allowed." << endl;
            }
            break;
        default:
            cout << "Invalid choice." << endl;
            break;
        }
    }
    else {
        cout << "Invalid choice. Please select a number between 1 and 4." << endl;
    }

    return 0;
}
