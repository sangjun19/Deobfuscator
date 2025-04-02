#include <stdio.h>
#include <stdlib.h>

void doEncrypt(FILE *enc, FILE *sec)
{
    char key;
    while ((key = fgetc(enc)) != EOF)
    {
        fputc(key + 3, sec); // Encrypt and write character to sec(secured) file
    }
}

void doDecrypt(FILE *sec, FILE *dec)
{
    char key;
    while ((key = fgetc(sec)) != EOF)
    {
        fputc(key - 3, dec); // Decrypt from secured file and write characters to dec(decrypted) file
    }
}

void main(void)
{
    clrscr();

    FILE *dec = NULL;
    FILE *enc = NULL;
    FILE *sec = NULL;

    int token;

    printf("\nEnter 1 for Encrypting the contents in \"accounts.txt\" file");
    printf("\nEnter 2 for Decrypting the contents in \"sec.txt\" file");

    scanf("%d", &token);

    switch (token)
    {
    case 1:
        enc = fopen("acc.txt", "r");
        sec = fopen("sec.txt", "w");
        doEncrypt(enc, sec);
        printf("\nOperations Performed Successfully");
        break;
    case 2:
        sec = fopen("sec.txt", "r");
        dec = fopen("dec.txt", "w");
        doDecrypt(sec, dec);
        printf("\nOperations Performed Successfully");
        break;
    default:
        printf("\nINVALID INPUT");
        break;
    }

    fclose(enc);
    fclose(dec);
    fclose(sec);

    getch();
}
