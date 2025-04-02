#include <iostream>
#include <string>

using namespace std;

int main() {
    string fullName;
    string course;
    int score;

    // Ask user for input
    cout << "Enter student's full name: ";
    cin >> fullName;
    
    cout << "Enter course name: ";
    cin.ignore();  // Ignore newline character left in input buffer
    getline(cin, course);

    cout << "Enter score (0-100): ";
    cin >> score;

    // Validate input
    if (score <= 0 || score >= 100) {
        cout << "Invalid score. Please enter a score between 0 and 100.\n";
        return 0;
    }

    // Determine grade using switch-case
    switch(score / 10) {   // Convert score to grade
        case 10:
        case 9:
        case 8:
            cout << fullName << ", " << course << ": Grade A (Excellent)\n";
            break;
        case 7:
            cout << fullName << ", " << course << ": Grade B (Very Good)\n";
            break;
        case 6:
            cout << fullName << ", " << course << ": Grade C (Good)\n";
            break;
        case 5:
            cout << fullName << ", " << course << ": Grade D (Fair)\n";
            break;
        default:
            cout << fullName << ", " << course << ": Grade F (Fail)\n";
            break;
    }

    return 0;
}
