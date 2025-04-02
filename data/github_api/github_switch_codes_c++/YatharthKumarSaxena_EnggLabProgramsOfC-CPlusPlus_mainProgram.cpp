#include "functionsofDLL.cpp"
template <class Type>
void operateFunctionOnInterface(DoublyLinkedList<Type>& DLL){
    while(true){
        cout<<"What function you want to operate \n";
        cout<<"\n1. Insert at head\n";
        cout<<"2. Insert at tail\n";
        cout<<"3. Insert after Nth Node\n";
        cout<<"4. Insert before Nth Node\n";
        cout<<"5. Delete at head\n";
        cout<<"6. Delete at tail\n";
        cout<<"7. Delete Nth Node\n";
        cout<<"8. Delete Node by value\n";
        cout<<"9. Display Doubly Linked List\n";
        cout<<"10. Display Doubly Linked List in reverse order\n";
        cout<<"11. Display Size of the Doubly Linked List\n";
        cout<<"12. Make Doubly Linked List empty\n";
        cout<<"13. Check your DLL is empty or not\n";
        cout<<"14. Exit\n";
        int n;
        cout<<"\nPlease enter your choice :- ";
        cin>>n;
        cout<<endl;
        switch (n) {
            case 1: {
                Type val;
                cout << "Enter value to insert at head: ";
                cin >> val;
                DLL.insertAtHead(val);
                cout<<endl;
                break;
            }
            case 2: {
                Type val;
                cout << "Enter value to insert at tail: ";
                cin >> val;
                DLL.insertAtTail(val);
                cout<<endl;
                break;
            }
            case 3: {
                Type val;
                int loc;
                cout << "Enter value to insert after Nth node: ";
                cin >> val;
                cout << "Enter the position (N): ";
                cin >> loc;
                DLL.insertAfterNthNode(val, loc);
                cout<<endl;
                break;
            }
            case 4: {
                Type val;
                int loc;
                cout << "Enter value to insert before Nth node: ";
                cin >> val;
                cout << "Enter the position (N): ";
                cin >> loc;
                DLL.insertBeforeNthNode(val, loc);
                cout<<endl;
                break;
            }
            case 5: {
                Type val = DLL.deleteAtHead();
                cout << "Deleted value from head: " << val << endl;
                cout<<endl;
                break;
            }
            case 6: {
                Type val = DLL.deleteAtTail();
                cout << "Deleted value from tail: " << val << endl;
                cout<<endl;
                break;
            }
            case 7: {
                int loc;
                cout << "Enter the position (N): ";
                cin >> loc;
                Type val = DLL.deleteNthNode(loc);
                cout << "Deleted value at position " << loc << ": " << val << endl;
                cout<<endl;
                break;
            }
            case 8: {
                Type val;
                cout << "Enter value to delete: ";
                cin >> val;
                DLL.deleteNodeByValue(val);
                cout<<endl;
                break;
            }
            case 9: {
                cout<<"Doubly Linked List looks as given below :- \n";
                DLL.display();
                cout<<endl;
                break;
            }
            case 10: {
                cout<<"Doubly Linked List in reverse order looks as given below :- \n";
                DLL.displayReverse();
                cout<<endl;
                break;
            }
            case 11: {
                int listSize = DLL.size();
                cout << "Size of the list: " << listSize << endl;
                cout<<endl;
                break;
            }
            case 12: {
                DLL.makeListEmpty();
                cout << "The list has been emptied\n";
                cout<<endl;
                break;
            }
            case 13: {
                if (DLL.isEmpty()) {
                    cout << "The list is empty\n";
                } else {
                    cout << "The list is not empty\n";
                }
                cout<<endl;
                break;
            }
            case 14:
                cout << "\nExiting the operation menu.\n";
                return;
            default:
                cout << "Please enter a valid choice\n";
                cout<<endl;
                break;
        }
    }
}
void activeInterface(){
    cout<<"Please specify the type of your Doubly Linked list \n";
    cout<<"\n1. Int\n";
    cout<<"2. Char\n";
    cout<<"3. Float\n";
    cout<<"\nPlease enter your choice:- ";
    int type;
    cin>>type;
    cout<<endl;
    switch(type){
        case 1:{
            DoublyLinkedList<int>DLL;
            operateFunctionOnInterface(DLL);
            break;
        }
        case 2:{
            DoublyLinkedList<char>DLL;
            operateFunctionOnInterface(DLL);
            break;
        }
        case 3:{
            DoublyLinkedList<float>DLL;
            operateFunctionOnInterface(DLL);
            break;
        }
        default:
            cout<<"Invalid Choice\n";
            break;
    }
}
int main(){
    cout<<"Welcome to the Doubly Linked List Program\n";
    while(true){
        int n;
        cout<<"\n1. Want to create a Doubly Linked List\n";
        cout<<"2. For Exit\n";
        cout<<"\nPlease enter your choice :- ";
        cin>>n;
        cout<<endl;
        switch (n)
        {
        case 1:{
            activeInterface();
            break;
        }
        case 2:
            cout<<"Exiting the main program\n";
            cout<<endl;
            exit(0);
            break;
        
        default:
            break;
        }
    }
    return 0;
}