// Repository: artem-666478/LORDIK-C-
// File: DZ 2910.ccp

#include <iostream>
#include <cmath>
using namespace std;

/*int main() {
    setlocale(LC_ALL, "Russian");
    double R1, R2, R3;
    cout << "Введите значение R1: " << endl;
    cin >> R1;
    cout << "Введите значение R2:"<< endl;
    cin >> R2;
    cout << "Введите значение R3: "<<endl;
    cin >> R3;
    double R = 1 / (1 / R1 + 1 / R2 + 1 / R3);

    cout << "Общее сопротивление равно: " << R << endl;

    return 0;
}   */
/*int main() {
    setlocale(LC_ALL, "Russian");
    double x1, x2, y1, y2;

    cout << "Введите x1: ";
    cin >> x1;
    cout << "Введите y1: ";
    cin >> y1;

    cout << "Введите x2: ";
    cin >> x2;
    cout << "Введите y2: ";
    cin >> y2;

    double distance = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    cout << "Расстояние между точками (" << x1 << ", " << y1 << ") и (" << x2 << ", " << y2 << ") равно: " << distance << endl;

}*/
/*int main() {
    setlocale(LC_ALL, "Russian");
    int a, b, c;
    cout << "Введите a,b,c " << endl;
    cin >> a >> b >> c;
    if ((a < b) && (b < c)) {
        cout << "Неравенство выполняется." << endl;
    }
    else {
        cout << "Неравенство не выполняется." << endl;
    }
}*/
/*int main() {
    setlocale(LC_ALL, "Russian");
    double a = 2.5;
    double b = 1.125;
    double c = 3.324;

    if ((a > 1) &&(a < 3)){
        cout << "a принадлежит интервалу (1,3)"<<endl;
    }
    else {
        cout << "a  не принадлежит интервалу (1,3)" << endl;
    }
    if ((b > 1) &&(b < 3)) {
        cout << "b принадлежит интервалу (1,3)" << endl;
    }
    else {
        cout << "b  не принадлежит интервалу (1,3)" << endl;
    }
    if ((c> 1) &&(c < 3)) {
        cout << "c принадлежит интервалу (1,3)" << endl;
    }
    else {
        cout << "c не принадлежит интервалу (1,3)" << endl;
    }
}*/
int main() {
    setlocale(LC_ALL, "Russian");
        int a,b,d,f;
        cin >> a;
        b = a / 100;
        d = a % 100 / 10;
        f = a % 10;
        int с = 0;       
        while (a > 0) {
            a /= 10;
            с++;
        }

        if (с == 0) {
            cout << "Вы ввели ноль." << endl;
        }
        else {
            cout << "Количество цифр в числе: " << с << endl;
        }
        cout << "Сумма цифр в числе = " << b + d + f << endl;
        if (b > 0) {
            cout << "Первая цифра в числе : " << b << endl;
        }
        else {
            cout << "Первая цифра в числе : " << d << endl;

        }
        
        cout << "Последняя цифра в числе : " << f << endl;

       
    }

*/


#include <iostream>
using namespace std;

/*int main()
{
	setlocale(LC_ALL, "Russian");
	int x, y;
	cin >> x >> y;
	if ((y >= 2 * x - 1) && (y <= -2 * x + 1) && (y >= -2 * x - 1)) {
		cout << "Лежит";

	}
	else {
		cout << "Не лежит";
	}

}
#include <iostream>;
#include "math.h"
using namespace std;
/*int main()
{
	setlocale(LC_ALL, "RUS");

	double x, y;
	cout << "введите х: ";
	cin >> x;
	cout << "введите y: ";
	cin >> y;

	if ((y <= 0, 5 * x + 1) && (y >= -0, 5 * x - 1) or ((1 >= pow(y, 2) + pow(x, 2)) and x >= 0)) 
	{
		cout << "входит ";
	}
	else { cout << "не входит "; }





}int main()
{
    float y;
    float x;
    for (int i = 0; i < 5; i++) {
        switch (i)
        {
        case(0):
            y = -0.1;
            x = -2.5;
            break;
        case(1):
            y = 0.1;
            x = 1.5;
            break;
        case(2):
            y = 0.5;
            x = -0.5;
            break;
        case(3):
            y = 0.5;
            x = 1.5;
            break;
        case(4):
            y = 1.1;
            x = 0.5;
            break;
        }
    }
}
