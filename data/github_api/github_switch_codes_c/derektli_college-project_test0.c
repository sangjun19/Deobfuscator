int a; 
int b;
int *d = &a;
float c;
void foo(int *i, char b, float c)
{
   //*i = 1;
   printf("a = %d", a);
}

int main(int arg)
{
    int i,a,b,c;
    switch (i){
	case 1: a =1;
	case 2: b = 2;
	default: c = 3;
    };
    printf ("%d %d %d\n", a, b,c);
    
   return 0;
}

