#include<stdio.h>
#include<conio.h>
void main()
{
int instr=0,acc=0xA937,DR=0,AR=0,instrnum;
int instr1,mar,PC;
clrscr();
printf("\nprogram for direct addressing, I=0");
printf("\nEnter 4xxx for STA");
printf("\nEnter 5xxx for BUN");
printf("\nEnter 6xxx for BSA");
printf("\nEnter 7xxx for ISZ ");
scanf("%x",&instr);
instr1=instr;
mar=(instr&0xFFF);
instrnum=(instr1 & 0xF000)>>12;
printf("THE instruction is %x",instrnum);
switch(instrnum)
{
 case 4: printf("\n Store Accumulator(STA)");
mar=acc;
printf("\n Mar=%x",mar);
break;
 case 5: printf("\n Branch Unconditionally(BUN)");
PC=AR;
break;
 case 6: printf("\n Branch and Save Return Address(BSA)");
mar=PC;
PC=AR+1;
break;
 case 7: printf("\n Increment and skip if zero(ISZ)");
DR=mar;
if(DR==0)
{
int PC=0x0022;
PC=PC+1;
}
else
DR=DR+1;
mar=DR;
printf("\n Now DR is %x",DR);
printf("\n Mar=%x",mar);
break;
}
getch();
}