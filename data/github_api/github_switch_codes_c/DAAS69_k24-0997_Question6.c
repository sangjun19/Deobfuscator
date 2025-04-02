#include <stdio.h>

typedef struct travel_packages {
    char name[9];
    char destination[9];
    int duration;
    long cost;
    int seats;
} package;

void display(package arr[]) {
    for (int i=0;i<3;i++) {
        printf("Package number %d:\n", i + 1);
        printf("Name: %s\n", arr[i].name);
        printf("Destination: %s\n", arr[i].destination);
        printf("Duration: %d days\n", arr[i].duration);
        printf("Cost: %ld\n", arr[i].cost);
        printf("Available seats: %d\n\n", arr[i].seats);
    }
}

void book_package(package arr[], int size) {
    int package_choice, seats_to_book;
    printf("Available packages:\n");
    display(arr);
    printf("Enter the package number you want to book: ");
    scanf("%d", &package_choice);

    if (package_choice < 1 || package_choice > size){
        printf("Invalid package choice! Please try again.\n");
        book_package(arr, 3);
    }

    printf("Enter the number of seats you want to book: ");
    scanf("%d", &seats_to_book);

    if (arr[package_choice-1].seats >= seats_to_book){
        arr[package_choice-1].seats -= seats_to_book;
        printf("Successfully booked %d seat(s) for package '%s'!\n\n", seats_to_book, arr[package_choice - 1].name);
    } 
    else{
        printf("Sorry, only %d seat(s) are available for this package.\n\n", arr[package_choice - 1].seats);
    }
}

int main() {
    package arr[3] = {
        {"package1", "USA", 10, 600000, 21},
        {"package2", "Russia", 12, 400000, 32},
        {"package3", "China", 14, 200000, 5}
    };
    int choice;

    while (1) {
        printf("Welcome to Fast Travels!\n");
        printf("1. Display available packages\n");
        printf("2. Book a package\n");
        printf("3. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                display(arr);
                break;
            case 2:
                book_package(arr, 3);
                break;
            case 3:
                printf("exiting :)");
                return 0;
            default:
                printf("Invalid choice! Please try again.");
        }
    }
}
