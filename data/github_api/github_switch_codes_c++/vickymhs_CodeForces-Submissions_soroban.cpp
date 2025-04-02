#include<bits/stdc++.h>
using namespace std;
int main()
{
	long long int n,r;
	cin>>n;
	if(n==0)
	cout<<"O-|-OOOO\n";
	else
	{
		while(n>0)
	{
		r=n%10;
		switch(r)
		{
			case 0:cout<<"O-|-OOOO\n";
			break;
			case 1:cout<<"O-|O-OOO\n";
			break;
			case 2:cout<<"O-|OO-OO\n";
			break;
			case 3:cout<<"O-|OOO-O\n";
			break;
			case 4:cout<<"O-|OOOO-\n";
			break;
			case 5:cout<<"-O|-OOOO\n";
			break;
			case 6:cout<<"-O|O-OOO\n";
			break;
			case 7:cout<<"-O|OO-OO\n";
			break;
			case 8:cout<<"-O|OOO-O\n";
			break;
			case 9:cout<<"-O|OOOO-\n";
			break;
		}
		n=n/10;
	}
	}
	
}
