#define  _CRT_SECURE_NO_WARNINGS 1	
//#include<stdio.h>
//#include<string.h>
//#include<iostream>
//#include<algorithm>
//using namespace std;
//
//int main()
//{
//	string a;
//	getline(cin, a);//(cin, s)
//	int c = strlen(a);
//	printf("%d ", c);
//	int alp = 0, num = 0, blo = 0, oth = 0;
//	for (int i = 0; i <strlen(a); i++)
//	{
//		if ((a[i] >= 'a' && a[i] <= 'z') || (a[i] >= 'A' && a[i] <= 'Z'))
//			alp++;
//		else if (a[i] >= '0' && a[i] <= '9')
//			num++;
//		else if (a[i] == ' ')
//			blo++;
//		else
//			oth++;
//	}
//	printf("%d %d %d %d", alp, num, blo, oth);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int letter = 0, number = 0, blank = 0, others = 0, c;        //分别为字母、数字、空格、其他
//	while ((c = getchar()) != '\n') {
//		if (c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z')    //判断是否为字母
//			letter++;
//		else if (c >= '0' && c <= '9')                     //判断是都为数字
//			number++;
//		else if (c == ' ')                                 //判断是否为空格
//			blank++;
//		else                                              //其他
//			others++;
//	}
//	printf("%d %d %d %d\n", letter, number, blank, others);
//	return 0;
//}
//# include<stdio.h>
//# include<math.h>
//int main()
//{
//	double M;
//	scanf("%lf", &M);
//	double back, sum = M;
//	int i, N;
//	scanf("%d", &N);
//	back = M * pow(0.5, N);
//	for (i = 1; i < N; i++)
//		sum += M * pow(0.5, i - 1);
//	printf("%.2lf %.2lf", back, sum);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int x;
//	scanf("%d", &x);
//	if (x > 0)
//		printf("%d", 2 * x);
//	else if (x < 0)
//		printf("%d", 2 * x + 1);
//	else
//		printf("%d", 3 * x);
//	return 0;
//}

//#include<stdio.h>
//int main()
//{
//	int n;
//	scanf("%d", &n);
//	switch (n / 10)
//	{
//	case 10:
//	case 9:
//		printf("A");
//		break;
//	case 8:
//		printf("B");
//		break;
//	case 7:
//		printf("C");
//	case 6:
//		printf("D");
//		break;
//	default:
//		printf("E");
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int x;
//	scanf("%d", &x);
//	if ((x % 4 == 0 && x % 100 != 0) || x % 400 == 0)
//		printf("是闰年");
//	else
//		printf("不是闰年");
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int sum = 0;
//	int n, i;
//	int plu = 1;
//	scanf("%d", &n);
//	for (i = 1; i <= n; i++)
//	{
//		sum += i;
//		plu *= i;
//	}
//	printf("%d %d", sum, plu);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int n = 0;
//	scanf("%d", &n);
//	for (int i = 2; i < n; i++)
//	{
//		if (n % i == 0)
//		{
//			printf("不是素数");
//			break;
//		}
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int a[10];
//	int i, j;
//	int temp = 0;
//	for (i = 0; i < 10; i++)
//	{
//		scanf("%d", &a[i]);
//	}
//	for (i = 0; i < 9; i++)
//	{
//		for (j = 0; j < 9 - i; i++)
//		{
//			if (a[j] > a[j + 1])
//			{
//				temp = a[j];
//				a[j] = a[j + 1];
//				a[j + 1] = temp;
//			}
//		}
//	}
//	for (i = 0; i < 10; i++)
//	{
//		printf("%d ", a[i]);
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int a[2][3] = { {1,2,3},{4,5,6} };
//	int b[3][2], i, j;
//	printf("array a:\n");
//	for (i = 0; i <= 1; i++)
//	{
//		for (j = 0; j <= 2; j++)
//		{
//			printf("%d", a[i][j]);
//			b[j][i] = a[i][j];
//		}
//		printf("\n");
//	}
//	for (i = 0; i <= 2; i++)
//	{
//		for (j = 0; j <= 1; j++)
//			printf("%d", b[i][j]);
//		printf("\n");
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int a[2][3];
//	int b[3][2];
//	int i, j;
//	for (i = 0; i < 2; i++)
//	{
//		for (j = 0; j < 3; j++)
//		{
//			scanf("%d", &a[i][j]);
//			b[j][i] = a[i][j];
//		}
//	}
//	for (i = 0; i < 3; i++)
//	{
//		for (j = 0; j < 2; j++)
//		{
//			printf("%d ", b[i][j]);
//		}
//		printf("\n");
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int i, j, a[10];
//	int max, temp;
//	for (i = 0; i < 10; i++)
//	{
//		{
//			scanf("%d", &a[10]);
//		}
//	}
//	for (i = 0; i < 10; i++)
//	{
//		for (j = 0; j < 10 - i; i++)
//		{
//			if (a[j] > a[j + 1])
//			{
//				max = j;
//				temp = a[j];
//				a[j] = a[j + 1];
//				a[j + 1] = temp;
//			}
//		}
//	}
//	printf("%d %d", max, a[max]);
//	return 0;
//}

//#include<stdio.h>
//int main()
//{
//	int x;
//	scanf("%d", &x);
//	if (x > 0)
//		printf("%d", 2 * x);
//	else if (x < 0)
//		printf("%d", 2 * x + 1);
//	else
//		printf("%d", x + 2);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int n;
//	scanf("%d", &n);
//	switch (n / 10)
//	{
//	case 10:
//	case 9:
//		printf("A");
//		break;
//	case 8:
//		printf("B");
//		break;
//	case 7:
//		printf("C");
//		break;
//	case 6:
//		printf("D");
//		break;
//	default:
//		printf("E");
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int y;
//	scanf("%d", &y);
//	if ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)
//		printf("s");
//	else
//		printf("f");
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int i, sum = 0, plu = 1;
//	int n;
//	scanf("%d", &n);
//	for (i = 1; i <= n; i++)
//	{
//		sum += i;
//		plu *= i;
//	}
//	printf("%d %d", sum, plu);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int n = 0;
//	int i;
//	scanf("%d", &n);
//	for (i = 2; i < n; i++)
//	{
//		if (n % i == 0)
//		{
//			printf("s");
//			break;
//		}
//	}
//	return 0;
//}
//#include<stdio.h>
//
//void bubble_sort(int* p, int len)
//{
//	int i = 0;
//	int j = 0;
//	for (i = 0; i < len - 1; i++)
//	{
//		int flag = 1;
//		for (j = 0; j < len - 1 - i; j++)
//		{
//			if (p[j] > p[j + 1])
//			{
//				int tmp = p[j];
//				p[j] = p[j + 1];
//				p[j + 1] = tmp;
//				flag = 0;
//			}
//		}
//		if (flag == 1)
//			break;
//	}
//}
//
//int main()
//{
//	int arr[] = { 1,5,9,11,46,79,12 };
//	int sz = sizeof arr / sizeof arr[0];
//	bubble_sort(arr, sz);
//	int i = 0;
//	for (i = 0; i < sz; i++)
//	{
//		printf("%d ", arr[i]);
//	}
//	return 0;
//}
//
//#include<stdio.h>
//int main()
//{
//	int a[2][3];
//	int b[3][2];
//	int i, j;
//	for (i = 0; i < 2; i++)
//	{
//		for (j = 0; j < 3; j++)
//		{
//			scanf("%d", &a[i][j]);
//			b[j][i] = a[i][j];
//		}
//	}
//	for (i = 0; i < 3; i++)
//	{
//		for (j = 0; j < 2; j++)
//		{
//			printf("%d ", b[i][j]);
//		}
//		printf("\n");
//	}
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int a[6], max, i;
//	for (i = 0; i < 6; i++)
//	{
//		scanf("%d", &a[i]);
//	}
//	max = a[0];
//	for (i = 0; i < 6; i++)
//	{
//		if (a[i] > max)
//			max = a[i];
//	}
//	printf("%d", max);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int a[11];
//	int i, t;
//	for (i = 0; i < 10; i++)
//	{
//		scanf("%d", &a[i]);
//	}
//	for (i = 0; i <= 5; i++)
//	{
//		t = a[i];
//		a[i] = a[10 - i];
//		a[10 - i] = t;
//	}
//	for (i = 0; i < 10; i++)
//	{
//		printf("%d", a[i]);
//	}
//	return 0;
//}
//#include<stdio.h>
//#include<string.h>
//int main()
//{
//	char a[100];
//	gets(a);
//	int alp = 0, num = 0, blo = 0, oth = 0;
//	for (int i = 0; i <strlen(a); i++)
//	{
//		if ((a[i] >= 'a' && a[i] <= 'z') || (a[i] >= 'A' && a[i] <= 'Z'))
//			alp++;
//		else if (a[i] >= '0' && a[i] <= '9')
//			num++;
//		else if (a[i] == ' ')
//			blo++;
//		else
//			oth++;
//	}
//	printf("%d %d %d %d", alp, num, blo, oth);
//	return 0;
//}
//#include<stdio.h>
//int main()
//{
//	int letter = 0, number = 0, blank = 0, others = 0, c;        //分别为字母、数字、空格、其他
//	while ((c = getchar()) != '\n') {
//		if (c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z')    //判断是否为字母
//			letter++;
//		else if (c >= '0' && c <= '9')                     //判断是都为数字
//			number++;
//		else if (c == ' ')                                 //判断是否为空格
//			blank++;
//		else                                              //其他
//			others++;
//	}
//	printf("%d %d %d %d\n", letter, number, blank, others);
//	return 0;
//}