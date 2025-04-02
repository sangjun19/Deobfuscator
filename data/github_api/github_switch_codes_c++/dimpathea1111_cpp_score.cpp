// // #include <iostream>
// // #include <iomanip>
// // using namespace std;

// // int main()
// // {
// //     system("clear");
// //     cout << left;
// //     cout << setw(40) << "Name";
// //     cout << setw(10) << "Gander";
// //     cout << setw(10) << "Age";
// //     cout << setw(10) << "Position";
// //     cout << setw(10) << "Salary" << endl;

// //     cout << left << "";
// //     cout << setw(40) << "dim patea";
// //     cout << setw(10) << "fomale";
// //     cout << setw(10) << "18";
// //     cout << setw(10) << " don't ";
// //     cout << setw(10) << "1000$" << endl;

// //     for (int i = 0; i < 80; i++)
// //     {
// //         cout << "-";
// //     }

// //     cout << "\n";
// //     cout << left << "";
// //     cout << setw(40) << "dim patea";
// //     cout << setw(10) << "fomale";
// //     cout << setw(10) << "18";
// //     cout << setw(10) << "don'";
// //     cout << setw(10) << "10000$" << endl;

// //     return 0;
// // }
// // Name: Chheang Sothearath Class:M3
// #include <iostream>
// #include <stdio.h>
// using namespace std;
// #define w cout
// int main()
// {
//     int i, j, k, p, m, n, col, row;
//     m = 2;
//     n = 3;
//     p = 5;
//     int code;
//     int A[2][3] = {{1, 2, 3}, {4, 5, 6}};
//     int B[3][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {20, 30, 40, 50, 60}};
//     int C[100][100], Sum[100][100], Mul[100][100];
//     w << "\n============================Practice 1 Matrix=================================";
//     do
//     {
//         w << "\n 1.Input Matrix from keyboard";
//         w << "\n 2.Output Matrix A and B";
//         w << "\n 3.Do Sub of Matrix A and B";
//         w << "\n 4.Do Mul of Matrix A and B";
//         w << "\n Press 5 to Countinue To Vector";
//         w << "\n Choose(1,2,3,4 and 5) from program to work";
//         w << "\n Please Choose Number:";
//         cin >> code;
//         switch (code)
//         {
//         case 1:
//         {
//             w << "\n Input Matrix from your Keyboard \n";
//             w << "\n Enter Row=";
//             cin >> row;
//             w << "\n Enter Coloumn=";
//             cin >> col;
//             for (int i = 0; i < col; i++)
//             {
//                 for (int j = 0; j < row; j++)
//                 {
//                     w << "\n Enter C[" << i << "][" << j << "]=";
//                     cin >> C[i][j];
//                 }
//             }
//             w << "\n Output Matrix C \n";
//             for (int i = 0; i < col; i++)
//             {
//                 for (int j = 0; j < row; j++)
//                 {
//                     w << C[i][j] << "\t";
//                 }
//                 w << endl;
//             }
//             break;
//         }
//         case 2:
//         {
//             w << "\n Output Matrix A \n";
//             for (i = 0; i < m; i++)
//             {
//                 for (j = 0; j < n; j++)
//                 {
//                     w << A[i][j] << "\t";
//                 }
//                 w << endl;
//             }
//             w << "\n Output Matrix B \n";
//             for (i = 0; i < n; i++)
//             {
//                 for (j = 0; j < p; j++)
//                 {
//                     w << B[i][j] << "\t";
//                 }
//                 w << endl;
//             }
//         }
//         break;
//         case 3:
//         {
//             w << "\n Sum(A+B)\n";
//             for (i = 0; i < m; i++)
//             {
//                 for (j = 0; j < n; j++)
//                 {
//                     Sum[i][j] = A[i][j] + B[i][j];
//                 }
//             }
//             for (int i = 0; i < m; i++)
//             {
//                 for (int j = 0; j < n; j++)
//                 {
//                     w << Sum[i][j] << "\t";
//                 }
//                 w << endl;
//             }
//             break;
//         }
//         case 4:
//         {
//             w << "\n Mul(A*B)\n";
//             for (i = 0; i < m; i++)
//             {
//                 for (j = 0; j < p; j++)
//                 {
//                     Mul[i][j] = 0;
//                     for (int k = 0; k < n; k++)
//                     {
//                         Mul[i][j] = Mul[i][j] + A[i][k] * B[k][j];
//                     }
//                 }
//             }
//             for (i = 0; i < 2; i++)
//             {
//                 for (j = 0; j < 5; j++)
//                 {
//                     w << Mul[i][j] << "\t";
//                 }
//                 w << endl;
//             }
//             break;
//         }
//         case 5:
//         {
//             w << "\n==============================You May Countinue=======================================";
//             break;
//         default:
//             w << "\n=================Please Pick Number That Are Mention====================";
//         }
//         }
//     } while (code != 5);
//     w << "\n========================Pratice 2 Vector==================================";
//     int a[100];
//     int op;
//     int l, v;
//     int b[7] = {23, 44, 99, 33, 11, 88, 76};
//     int y, found;
//     found = 0;
//     do
//     {
//         w << "\n 1.input Array from keyboard";
//         w << "\n 2.Output Array that has been Assign";
//         w << "\n 3.Search Item of Array";
//         w << "\n 4.Sort list of Array";
//         w << "\n 5.Update Value to Array";
//         w << "\n 6.Insert Each Element to Array";
//         w << "\n 7.Delete of Array";
//         w << "\n 8 Exit The Program";
//         w << "\n Please Choose Number(1,2,3,4,5,6,7 and 8):";
//         cin >> op;
//         switch (op)
//         {
//         case 1:
//         {
//             w << "\n Create List of Array";
//             w << "\n Enter y=";
//             cin >> y;
//             for (int l = 0; l < y; l++)
//             {
//                 w << "\n Enter a[" << l << "]=";
//                 cin >> a[l];
//             }
//             w << "\n Output list of Array on screen \n ";
//             for (int l = 0; l < y; l++)
//             {
//                 w << "\n a[" << l << "]" << a[l];
//             }
//             w << endl;
//             w << "\n a[" << y << "]= \t";
//             for (l = 0; l < y; l++)
//             {
//                 w << a[l] << "\t";
//             }
//             break;
//         }
//         case 2:
//         {
//             w << "\n Output Array that has been Assign";
//             for (l = 0; l < 7; l++)
//             {
//                 w << "\n b[" << l << "]=" << b[l];
//             }
//             break;
//         }
//         case 3:
//         {
//             w << "\n Search Item of Array That already assign";
//             int Item;
//             w << "\n Enter Item=";
//             cin >> Item;
//             for (l = 0; l < 7; l++)
//                 if (Item == b[l])
//                 {
//                     w << "\n Item array in search";
//                     w << "\n b[" << l << "]=" << b[l];
//                     found = 1;
//                 }
//             if (found == 0)
//             {
//                 w << "\n There no existing in this list";
//             }
//             break;
//         }
//         case 4:
//         {
//             w << "\n Sort List of Array";
//             int Temp;
//             for (l = 0; l < 7; l++)
//                 for (v = l + 1; v < 7; v++)
//                     if (b[l] > b[v])
//                     {
//                         Temp = b[l];
//                         b[l] = b[v];
//                         b[v] = Temp;
//                     }
//             w << "\nAfter sort:";
//             for (l = 0; l < 7; l++)
//                 w << b[l] << "\t";
//             break;
//         }
//         case 5:
//         {
//             w << "\n Update Value to Array";
//             int d, pos;
//             w << "\n Enter the position of Array=";
//             cin >> pos;
//             w << "\n Enter the new value of Array=";
//             cin >> d;
//             w << "\n Outpist the array on the screen";
//             for (l = 0; l < 7; l++)
//                 if (l == pos - 1)
//                     b[l] = d;
//             w << "\n Output list after update \n";
//             for (l = 0; l < 7; l++)
//                 w << b[l] << "\t";
//             break;
//         }
//         case 6:
//         {
//             w << "\n Insert Element to Array ";
//             int add, into;
//             w << "\n enter array to insert=";
//             cin >> add;
//             w << "\n put new array into=";
//             cin >> into;
//             for (l = 7; l >= into; l--)
//                 b[l] = b[l - 1];
//             b[l] = add;
//             w << "\n output array after insertion \n";
//             for (l = 0; l < 7; l++)
//                 w << b[l] << "\t\t";
//             break;
//         }
//         case 7:
//         {
//             w << "\n Delete each element of Array";
//             int ele, h;
//             int f;
//             f = 7;
//             w << "\n enter array to clear=";
//             cin >> ele;
//             for (l = 0; l < 7; l++)
//             {
//                 if (b[l] == ele)
//                 {
//                     for (h = l; h < f - 1; h++)
//                         b[h] = b[h + 1];
//                     found++;
//                     i--;
//                     f--;
//                 }
//             }
//             if (found == 0)
//             {
//                 w << "\n element not found!";
//             }
//             else
//             {
//                 w << "\n element deleted successfully";
//                 w << "\n Output list of Array after delete\n";
//                 for (h = 0; h < f - 1; h++)
//                 {
//                     w << b[h] << "\t";
//                 }
//             }
//         }
//         break;
//         case 8:
//         {
//             w << "\n====================Thank You==================================";
//             break;
//         default:
//             w << "\n Please that are mention";
//         }
//         }
//     } while (op != 8);
//     return (0);
// }

#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace std;
void vector()
{
    cout << "\n\t 1. Create list of vector";
    cout << "\n\t1.1 Assing value to array or";
    cout << "\n\t2. Output list of vector on screen ";
    cout << "\n\t3. Seach each of element on screen";
    cout << "\n\t4. Show list of vector after sort";
    cout << "\n\t5. Update new value to array";
    cout << "\n\t6. Insert each element in vector";
    cout << "\n\t7. Delete each element in vector";
}

void matrix()

{
    cout << "\n\\t1. Create list of matrix ";
    cout << "\n\tA. With assign value to array";
    cout << "\n\tB. With input form keyboard";
    cout << "\n\t2. Output list of matrix (A and B )";
    cout << "\n\t3. Sum matrix (Sum = A + B)";
    cout << "\nt\4, Multiple matrix (Mul = A * B)";
}

int main()
{
    matrix();

    int A[2][3] = {{11, 33, 44}, {10, 20, 30}};
    int B[3][5] = {{23, 12, 34, 67, 50}, {10, 50, 15, 54, 76}};
    int y, x, c, column, row, choose;
    int k = 3, p = 2, g = 5;
    int s[100][100], mul[100][100], sum[100][100];
    do
    {
        cout << "SELECT YOU CHOOOSE =========== : ";
        cin >> choose;
        switch (choose)
        {
        case 1:
        {
            cout << "\n Input Matrix from your Keyboard " << endl;
            cout << "\n Enter Coloumn =";
            cin >> column;
            cout << "\n Enter Row =";
            cin >> row;
            for (int y = 0; y < column; y++)
            {
                for (int x = 0; x < row; x++)
                {
                    cout << "\n Enter s[" << y << "][" << y << "]=" << endl;
                    cin >> s[y][x];
                }
            }
            cout << "\n Output Matrix  \n";
            for (int y = 0; y < column; y++)
            {
                for (int x = 0; x < row; x++)
                {
                    cout << s[y][x] << "\t";
                }
                cout << endl;
            }
            break;
        }
        case 2:
        {
            cout << "output maxtrix A :" << endl;

            for (y = 0; y < p; y++)
            {
                for (x = 0; x < k; x++)
                {
                    cout << A[y][x] << "t";
                }
                cout << endl;
            }

            cout << "output maxtrix B :" << endl;
            for (y = 0; y < p; y++)
            {
                for (x = 0; x < g; x++)
                {
                    cout << B[y][x] << "\t";
                }
            }
            break;
        }
        case 3:
        {
            cout << "Maxtrix of  Sum (A + B) :" << endl;
            for (y = 0; y < p; y++)
            {
                for (x = 0; x < k; x++)
                {
                    sum[y][x] = A[y][x] + B[y][x];
                }
            }
            for (y = 0; y < k; y++)
            {
                for (x = 0; x < g; x++)
                {
                    cout << sum[y][x] << "\t";
                }
                cout << endl;
            }
            break;
        }
        case 4:
        {
            cout << "Maxtrix of Mul (A * B) :" << endl;
            for (y = 0; y < p; y++)
            {
                for (x = 0; x < k; x++)
                {
                    mul[y][x] = 0;
                    for (int c = 0; c < k; c++)
                    {
                        mul[y][x] = mul[y][x] + A[y][x] * B[y][x];
                    }
                }
            }
            for (y = 0; y < p; y++)
            {
                for (x = 0; x < g; x++)
                {
                    cout << mul[y][x] << "\t";
                }
                cout << endl;
            }
        }
        default:;
        }
        cout << " The end !!!!";
        cout << "Thenk you";
        exit(0);
    } while (1);

    return 0;
}