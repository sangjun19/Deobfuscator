#include "game.h"

void meun()
{
    printf("***************************\n");
    printf("*****1.play    0.exit*****\n");
    printf("***************************\n");
}
void game()
{
    //扫雷游戏
    char board[ROWS][COLS] = {0};//创建棋盘
    char show[ROWS][COLS] = {0};//创建显示棋盘（未知）
    intboard(board, ROWS, COLS, '0');//初始化棋盘
    intshow(show, ROWS, COLS,'*');//初始化显示棋盘
    printboard(show, ROWS, COLS);//打印显示棋盘
}

int main()
{
    int inport = 0;
    do
    {
        meun();
        scanf("%d", &inport);
        switch (inport)
        {
            case 1:
                game();
                break;
            case 0:
                printf("退出扫雷\n");
                break;
            default:
                printf("输入错误，请重新输入\n");
                break;
        }
    }while (inport);
    return 0;  
}
//看2024-10-26.c