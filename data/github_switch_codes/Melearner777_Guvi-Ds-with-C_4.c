// Repository: Melearner777/Guvi-Ds-with-C
// File: 4.c

/*Complex Menu Navigation using Switch Statement
Problem Statement:
Write a C program that simulates a menu-driven application for a library management system using the switch statement. The menu should offer the following options:

Add a book
Display books
Search for a book by title
Exit
The twist is that each option should lead to a submenu with additional choices. For example, the "Add a book" option should allow the user to enter the title, author, and year of publication. The "Display books" option should display all books in a specific format. The "Search for a book by title" option should allow for a partial title search.

Description:

The program should use the switch statement to handle the main menu and each submenu.
Implement functions for each submenu operation.
Use dynamic memory allocation to store book details.
Use a struct to represent each book.

Input Format:

The user inputs the menu choice followed by submenu choices as required.
The input ends when the user selects the "Exit" option.

Output Format:

The output varies based on the selected menu and submenu options, including confirmation messages and book details.

The below are the output for corresponding operation:

Add a book:

Book added successfully.

Display books:

Case1:

No books in the library.

Case2:

Books in library:
Title: < Title >, Author: < Author >, Year: < Year >

Seach for a book by title:

Case1:

No book found with the given title.

Case2:

Books found:
Title: < Title >, Author: < Author >, Year: < Year >

Exit

Exiting...

Private Testcase Input 1:
1
C Programming
Dennis Ritchie
1978
4*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TITLE 100
#define MAX_AUTHOR 100

struct Book {
    char title[MAX_TITLE];
    char author[MAX_AUTHOR];
    int year;
};

struct Library {
    struct Book* books;
    int count;
    int capacity;
};

void initLibrary(struct Library* lib) {
    lib->books = NULL;
    lib->count = 0;
    lib->capacity = 0;
}

void addBook(struct Library* lib) {
    if (lib->count == lib->capacity) {
        int newCapacity = (lib->capacity == 0) ? 1 : lib->capacity * 2;
        struct Book* newBooks = realloc(lib->books, newCapacity * sizeof(struct Book));
        if (newBooks == NULL) {
            printf("Memory allocation failed.\n");
            return;
        }
        lib->books = newBooks;
        lib->capacity = newCapacity;
    }

    struct Book newBook;
    scanf(" %[^\n]", newBook.title);
    scanf(" %[^\n]", newBook.author);
    scanf("%d", &newBook.year);

    lib->books[lib->count++] = newBook;
    printf("Book added successfully.\n");
}

void displayBooks(struct Library* lib) {
    if (lib->count == 0) {
        printf("No books in the library.\n");
        return;
    }

    printf("Books in library:\n");
    for (int i = 0; i < lib->count; i++) {
        printf("Title: %s, Author: %s, Year: %d\n", 
               lib->books[i].title, lib->books[i].author, lib->books[i].year);
    }
}

void searchBook(struct Library* lib) {
    char searchTitle[MAX_TITLE];
    scanf(" %[^\n]", searchTitle);

    int found = 0;
    for (int i = 0; i < lib->count; i++) {
        if (strstr(lib->books[i].title, searchTitle) != NULL) {
            if (!found) {
                printf("Books found:\n");
                found = 1;
            }
            printf("Title: %s, Author: %s, Year: %d\n", 
                   lib->books[i].title, lib->books[i].author, lib->books[i].year);
        }
    }

    if (!found) {
        printf("No book found with the given title.\n");
    }
}

int main() {
    struct Library lib;
    initLibrary(&lib);

    int choice;
    do {
        scanf("%d", &choice);
        switch(choice) {
            case 1:
                addBook(&lib);
                break;
            case 2:
                displayBooks(&lib);
                break;
            case 3:
                searchBook(&lib);
                break;
            case 4:
                printf("Exiting...\n");
                break;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 4);

    free(lib.books);
    return 0;
}

