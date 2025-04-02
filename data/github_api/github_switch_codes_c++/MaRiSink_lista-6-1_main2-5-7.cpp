#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace std;

int main()
{
    //5
   /*  srand (time(NULL));

    int i = 1 + rand() % 10;



     for (int a = 0; a < 10; a++)
    {
        for (int n = 0; n < 10; n++)
        {
            i = 1 + rand() % 10;

            if ( i == 1 || i == 2 || i ==3 || i == 4 )
            {
                cout << " M ";
            }
            else
            {
                if (  i == 5 || i == 6 || i ==7 || i == 8 || i == 9 || i == 10)
                {
                    cout << " . ";
                }

            }
        }
        cout << endl;
    }*/





    //6
   /* srand (time(NULL));

    int i = 1 + rand() % 10;
    float a1 = 0;
    float a2 = 0;
    float a3 = 0;
    float a4 = 0;
    float a5 = 0;
    float a6 = 0;
    float a7 = 0;
    float a8 = 0;
    float a9 = 0;
    float a10 = 0;

    float procent1 = 0;
    float procent2 = 0;
    float procent3 = 0;
    float procent4 = 0;
    float procent5 = 0;
    float procent6 = 0;
    float procent7 = 0;
    float procent8 = 0;
    float procent9 = 0;
    float procent10 = 0;





     for (int a = 0; a < 10000; a++)
    {
         i = 1 + rand() % 10;
         if ( i == 1 )
         {
             a1++;
         }
          if ( i == 2 )
         {
             a2++;
         }
          if ( i == 3 )
         {
             a3++;
         }
          if ( i == 4 )
         {
             a4++;
         }
          if ( i == 5 )
         {
             a5++;
         }
         if ( i == 6 )
         {
             a6++;
         }
          if ( i == 7 )
         {
             a7++;
         }
          if ( i == 8 )
         {
             a8++;
         }
          if ( i == 9 )
         {
             a9++;
         }
          if ( i == 10 )
         {
             a10++;
         }
    }

    procent1 = a1 / 100;
    procent2 = a2 / 100;
    procent3 = a3 / 100;
    procent4 = a4 / 100;
    procent5 = a5 / 100;
    procent6 = a6 / 100;
    procent7 = a7 / 100;
    procent8 = a8 / 100;
    procent9 = a9 / 100;
    procent10 = a10 / 100;


    cout << "Jedynek bylo " << a1 << endl;
    cout << procent1 << " % - zajmuje liczba jeden" << endl;
    cout << endl;

    cout << "Dwojek bylo " << a2 << endl;
    cout << procent2 << " % - zajmuje liczba dwa" << endl;
    cout << endl;

    cout << "Trojek bylo " << a3 << endl;
    cout << procent3 << " % - zajmuje liczba trzy" << endl;
    cout << endl;

    cout << "Czworek bylo " << a4 << endl;
    cout << procent4 << " % - zajmuje liczba cztery" << endl;
    cout << endl;

    cout << "Piatek bylo " << a5 << endl;
    cout << procent5 << " % - zajmuje liczba piec" << endl;
    cout << endl;

    cout << "Szostek bylo " << a6 << endl;
    cout << procent6 << " % - zajmuje liczba sesc" << endl;
    cout << endl;

    cout << "Siodemek bylo " << a7 << endl;
    cout << procent7 << " % - zajmuje liczba siedem" << endl;
    cout << endl;

    cout << "Osemek bylo " << a8 << endl;
    cout << procent8 << " % - zajmuje liczba osiem" << endl;
    cout << endl;

    cout << "Dzewientek bylo " << a9 << endl;
    cout << procent9 << " % - zajmuje liczba dziewiec" << endl;
    cout << endl;

    cout << "Dziesiatek bylo " << a10 << endl;
    cout << procent10 << " % - zajmuje liczba dziesienc" << endl;
    */


    //7
    srand (time(NULL));

    int i = 1 + rand() % 3;

    int iloscwygranych = 0;
    int iloscprzegranych = 0;

    for (int a = 0; a < 3; a++)
    {
        int i = 1 + rand() % 3;
        cout << "Co podasz?" << endl;
        cout << "1 - kamien" << endl;
        cout << "2 - noznicy" << endl;
        cout << "3 - papier" << endl;

        int odp;
        cin >> odp;

        switch (odp)
        {
        case 1:
            if ( i == 1 )
            {
                cout << " Tez kamien " << endl;
                cout << " remis " << endl;
            }
            if ( i == 2 )
            {
                cout << " Noznicy " << endl;
                cout << " Wygrales :) " << endl;
                iloscwygranych++;
            }
            if ( i == 3 )
            {
                cout << " Papier " << endl;
                cout << " Przegrales :( " << endl;
                iloscprzegranych++;
            }

            break;
        case 2:

            if ( i == 1 )
            {
                cout << " Papier " << endl;
                cout << " Wygrales :) " << endl;
                iloscwygranych++;
            }
            if ( i == 2 )
            {
                cout << " Tez noznicy " << endl;
                cout << " remis " << endl;
            }
            if ( i == 3 )
            {
                cout << " Kamien " << endl;
                cout << " Przegrales :( " << endl;
                iloscprzegranych++;
            }
            break;
        case 3:
            if ( i == 1 )
            {
                cout << " Kamien " << endl;
                cout << " Wygrales :) " << endl;
                iloscwygranych++;
            }
            if ( i == 2 )
            {
                cout << " Noznicy " << endl;
                cout << " Przegrales :( " << endl;
                iloscprzegranych++;
            }
            if ( i == 3 )
            {
                cout << " Tez papier " << endl;
                cout << " remis " << endl;
            }
            break;
        default:
            cout << "Nie ma takiej odp" << endl;
            break;

        }
    }
    if ( iloscprzegranych < iloscwygranych )
    {
        cout << endl;
        cout << "wygrales" << endl;
    }
    if ( iloscprzegranych > iloscwygranych )
    {
        cout << endl;
        cout << " przegrales " << endl;
    }
    if ( iloscprzegranych == iloscwygranych )
    {
        cout << endl;
        cout << "remis" << endl;
    }








}
