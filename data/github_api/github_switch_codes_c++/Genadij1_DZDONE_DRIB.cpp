// Repository: Genadij1/DZDONE
// File: DZDONE/DRIB.cpp

#include "DRIB.h"

int main()
{
	Drib d1;
	int n1, n2, den1, den2;
	cout << "Enter the numerator1: ";
	cin >> n1;
	d1.setNumerator1(n1);
	cout << endl;
	cout << "Enter the denominator1: ";
	cin >> den1;
	d1.setDenominator1(den1);
	cout << endl;
	cout << "Enter the numerator2: ";
	cin >> n2;
	d1.setNumerator2(n2);
	cout << endl;
	cout << "Enter the denominator2: ";
	cin >> den2;
	d1.setDenominator2(den2);
	cout << endl;

	cout << "1. +\n2. -\n3. *\n4. /\n5. print\n6. exit" << endl;
	int action;
	START:
	cout << "Enter action number 1-6: ";
	cin >> action;

	switch (action)
	{
	case 1:
		d1.sum();
		goto START;
	case 2:
		d1.sub();
		goto START;
	case 3:
		d1.mult();
		goto START;
	case 4:
		d1.div();
		goto START;
	case 5:
		d1.print();
		goto START;
	case 6:
		cout << "Goodbye!";
		break;
	efault:
	cout << "Error! Try again!" << endl;
	goto START;
	}

	system("pause>nul");
	return 0;
}
