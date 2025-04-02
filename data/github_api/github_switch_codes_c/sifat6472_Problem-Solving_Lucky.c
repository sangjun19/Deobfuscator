#include <stdio.h>
char ticket[7];
int idx, first3sum, last3sum;

void solve(){
    first3sum = 0;
    last3sum = 0;
   
    scanf("%s", ticket);
   
    idx = 0;
    while (ticket[idx] != '\0'){
       
        switch (idx){
            case 0 ... 2:
                first3sum += ticket[idx] - '0';
                break;
            case 3 ... 5:
                last3sum += ticket[idx] - '0';
                break;
            default: //None
                break;
        }
       
        idx++;
    }
   
    puts((first3sum == last3sum) ? "YES":"NO");
}


int main() {
    int T;
   
    scanf("%d", &T);
    getchar();
   
    while (T--){
        solve();
    }

    return 0;
}