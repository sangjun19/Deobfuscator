// Repository: prabhanjanbhat/DS-lab
// File: 1.c

#include <stdio.h>

// Structure to represent a bank account
struct BankAccount {
    int accountNumber;
    float balance;
};

// Function to create a new account
void createAccount(struct BankAccount *account, int accountNumber) {
    account->accountNumber = accountNumber;
    account->balance = 0.0;
    printf("Account created successfully. Account Number: %d\n", accountNumber);
}

// Function to deposit money into the account
void deposit(struct BankAccount *account, float amount) {
    account->balance += amount;
    printf("Deposit successful. New balance: %.2f\n", account->balance);
}

// Function to withdraw money from the account
void withdraw(struct BankAccount *account, float amount) {
    if (amount > account->balance) {
        printf("Insufficient funds. Withdrawal failed.\n");
    } else {
        account->balance -= amount;
        printf("Withdrawal successful. New balance: %.2f\n", account->balance);
    }
}

// Function to inquire about the account balance
void balanceInquiry(struct BankAccount *account) {
    printf("Account Balance: %.2f\n", account->balance);
}

int main() {
    struct BankAccount account;
    int accountNumber;
    int choice;
    float amount;

    // Account creation
    printf("Enter Account Number: ");
    scanf("%d", &accountNumber);
    createAccount(&account, accountNumber);

    // Menu for operations
    do {
        printf("\n1. Deposit\n");
        printf("2. Withdraw\n");
        printf("3. Balance Inquiry\n");
        printf("4. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Enter deposit amount: ");
                scanf("%f", &amount);
                deposit(&account, amount);
                break;
            case 2:
                printf("Enter withdrawal amount: ");
                scanf("%f", &amount);
                withdraw(&account, amount);
                break;
            case 3:
                balanceInquiry(&account);
                break;
            case 4:
                printf("Exiting the program. Goodbye!\n");
                break;
            default:
                printf("Invalid choice. Please enter a valid option.\n");
        }

    } while (choice != 4);

    return 0;
}
