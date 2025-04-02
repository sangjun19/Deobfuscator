#include <stdio.h>
#include <string.h>

typedef struct books {
    char title[9];
    char author[9];
    int publication_year;
    int price;
} book;

void display(book arr[]) {
    for (int i=0;i<3;i++) {
        printf("Book %d:\n", i + 1);
        printf("Title: %s\n", arr[i].title);
        printf("Author: %s\n", arr[i].author);
        printf("Publication Year: %d\n", arr[i].publication_year);
        printf("Price: %d\n\n", arr[i].price);
    }
}
void search(book arr[], char new[]){
    int flag=0;
    for(int i=0;i<3;i++){
        if(strcmp(arr[i].title, new) == 0){
            printf("Book %d:\n", i + 1);
            printf("Title: %s\n", arr[i].title);
            printf("Author: %s\n", arr[i].author);
            printf("Publication Year: %d\n", arr[i].publication_year);
            printf("Price: %d\n\n", arr[i].price);
            flag=1;
        }
    }
    if(flag==0){
        printf("We're sorry but we dont have any book by that title");
    }
}
void list_book(book arr[],char by_author[]){
    int flag=0;
    for(int i=0;i<3;i++){
        if(strcmp(arr[i].author, by_author) == 0){
            printf("Book %d:\n", i + 1);
            printf("Title: %s\n", arr[i].title);
            printf("Author: %s\n", arr[i].author);
            printf("Publication Year: %d\n", arr[i].publication_year);
            printf("Price: %d\n\n", arr[i].price);        
        }
    }
    if(flag==0){
        printf("We're sorry but we dont have any book by that author");
    }

}
int main() {
    char search_title[10],author_name[10];
    int choice;
    book arr[3] = {
        {"book1", "ahmed", 2019, 2000},
        {"book2", "saad", 2021, 2000},
        {"book3", "saad", 2000, 4000}
    };
    printf("welcome to the library\nList all books'1'\nSearch books by author name'2'\nSearch book by title '3'\n");
    scanf("%d",&choice);

    switch(choice){

        case 1:
            display(arr);
        break;

        case 2:
            printf("Please enter the author name: ");
            scanf("%s",author_name);
            list_book(arr,author_name);
        break;

        case 3:
            printf("Please enter the title of the book: ");
            scanf("%s",search_title);
            search(arr,search_title);
        break;

        default:
            printf("Invalid choice");    

    }

}
