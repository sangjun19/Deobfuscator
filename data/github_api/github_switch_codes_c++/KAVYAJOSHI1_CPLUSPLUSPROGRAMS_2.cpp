#include <iostream>
using namespace std;

template <typename T>
class Stack {
private:
    T* arr; // Array to hold stack elements
    int top; // Index of the top element in the stack
    int capacity; // Maximum size of the stack

public:
    // Constructor to initialize stack with given capacity
    Stack(int size) : capacity(size), top(-1) {
        arr = new T[capacity];
    }

    // Push an element onto the stack
    void push(T x) {
        if (top == capacity - 1) {
            cout << "Stack Overflow" << endl;
            return;
        }
        arr[++top] = x;
    }

    // Pop an element from the stack
    void pop() {
        if (top == -1) {
            cout << "Stack Underflow" << endl;
            return;
        }
        cout << "ELEMENT POPPED IS: " << arr[top--] << endl;
    }

    // Display all elements in the stack
    void display() {
        if (top == -1) {
            cout << "Stack is Empty" << endl;
            return;
        }
        cout << "THE STACK IS:" << endl;
        for (int i = 0; i <= top; i++) {
            cout << arr[i] << endl;
        }
    }

    // Destructor to release allocated memory
    ~Stack() {
        delete[] arr;
    }
};

int main() {
    int n;
    cout << "ENTER THE SIZE: ";
    cin >> n;

    Stack<int> stack(n); // Create a stack with capacity n for int

    while (true) {
        int choice;
        cout << "ENTER 1 TO PUSH:" << endl;
        cout << "ENTER 2 TO POP:" << endl;
        cout << "ENTER 3 TO DISPLAY:" << endl;
        cout << "ENTER 4 TO EXIT:" << endl;
        cin >> choice;

        switch (choice) {
            case 1: {
                int x;
                cout << "ENTER THE ELEMENT TO PUSH: ";
                cin >> x;
                stack.push(x);
                break;
            }
            case 2: {
                stack.pop();
                break;
            }
            case 3: {
                stack.display();
                break;
            }
            case 4: {
                cout << "Exiting..." << endl;
                return 0;
            }
            default: {
                cout << "INVALID INPUT" << endl;
                break;
            }
        }
    }

    return 0;
}
