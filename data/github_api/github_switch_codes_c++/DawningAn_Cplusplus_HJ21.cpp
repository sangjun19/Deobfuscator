#include <iostream>

using namespace std;



/*
九键手机键盘上的数字与字母的对应： 1--1， abc--2, def--3, ghi--4, jkl--5, mno--6, pqrs--7, tuv--8 wxyz--9, 0--0，把密码中出现的小写字母都变成九键键盘对应的数字，如：a 变成 2，x 变成 9.
而密码中出现的大写字母则变成小写之后往后移一位，如：X ，先变成小写，再往后移一位，变成了 y ，例外：Z 往后移是 a 。
数字和其它的符号都不做变换。

1≤n≤100 
输入描述：
输入一组密码，长度不超过100个字符。

输出描述：
输出密码变换后的字符串*/

int main(){
    string S;
    while(cin >> S){
        for(unsigned i = 0;i<S.size();i++){
            if(S[i] >= 'a' && S[i] <= 'z'){
                switch (S[i]){
                    case 'a':
                    case 'b':
                    case 'c':
                        S[i] = '2';
                        break;
                    
                    case 'd':
                    case 'e':
                    case 'f':
                        S[i] = '3';
                        break;

                    case 'g':
                    case 'h':
                    case 'i':
                        S[i] = '4';
                        break;

                    case 'j':
                    case 'k':
                    case 'l':
                        S[i] = '5';
                        break;

                    case 'm':
                    case 'n':
                    case 'o':
                        S[i] = '6';
                        break;

                    case 'p':
                    case 'q':
                    case 'r':
                    case 's':
                        S[i] = '7';
                        break;

                    case 't':
                    case 'u':
                    case 'v':
                        S[i] = '8';
                        break;


                    case 'w':
                    case 'x':
                    case 'y':
                    case 'z':
                        S[i] = '9';
                        break;

                }


            }else if(S[i] >='A' &&  S[i] < 'Z'){
                S[i] += 33;
            }else if(S[i] == 'Z'){
                S[i] = 'a';
            }
            
        }

        cout <<S<<endl;
    }

}