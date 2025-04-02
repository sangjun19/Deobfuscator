#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

int main(void) {
    int bakiye=1000;
    int islem;
    int tutar;
    
    printf("Bakiyeniz: %d",bakiye);//Kodun başında bakiyesini gösteriyoruz
    
    printf("\n\n*****************İŞLEMLER*****************\n");
    //İşlemleri sıralıyoruz işlemi yapan kişi görsün diye
    printf("1. Para Çekme\n2. Para Yatırma\n3. Para Bakiye Sorgulama\n4. Kart İade\n\n");
    
    printf("İşleminizi giriniz: ");
    scanf("%d",&islem);//Hangi işlemi seçtiğini soruyoruz
    
    //4 olmadığı sürece while döngüsüne gir diyoruz. 4'te kart iade edildiği için
    while (islem != 4) {
        switch(islem){
            case 1:
                printf("Bakiyeniz: %d\n",bakiye);
                printf("Çekmek istediğiniz tutari giriniz : ");
                scanf("%d",&tutar);
                bakiye-=tutar;//Bakiyeden çekmek istediği tutar çıkarılır
                printf("Yeni bakiyeniz : %d\n",bakiye);
                break;
            case 2:
                printf("Bakiyeniz: %d\n",bakiye);
                printf("Yatırmak istediğiniz tutari giriniz : ");
                scanf("%d",&tutar);
                bakiye+=tutar;//Bakiyeyle yatırmak istediği tutar toplanır
                printf("Yeni bakiyeniz : %d\n",bakiye);
                break;
            case 3:
                printf("Bakiyeniz: %d\n\n",bakiye);
                break;
            default:
                printf("Yanlış girdiniz!!\n");//Herhangi bir yanlış girmede hata verilir
                break;
        }
        printf("İşleminizi giriniz: ");
        scanf("%d",&islem);
    }
    //İşlem 4 olduğunda kart iade ediliyor koddan çıkış yapılıyor
    if(islem==4){
        printf("İyi günler...\n");
    }
    printf("\n");
    return 0;
}
