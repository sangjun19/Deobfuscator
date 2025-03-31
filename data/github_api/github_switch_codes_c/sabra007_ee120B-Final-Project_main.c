/*
*
* Sargis Abrahamyan sabra007@ucr.edu
* Lab section: B21
* Assignment: Final Project
*
* I acknowledge all content contained herein, excluding template or example
* code, is my own original work.
*/

//Timer_LCD_Sound.h contains the code given in the lab manual for timer, LCD, and PWM (sound)
#include "Timer_LCD_Sound.h"
 
// Array to hold notes
const double Notes[] = {261.63, 293.66, 329.63, 349.23, 392.00};

//global variables 

//to get the input
unsigned char tmpA;

unsigned char sequence[9]; // sequence[] will store the sequence of LEDs

unsigned char countLED = 0;    //the number of LEDs to blink at the current level
unsigned char countUser = 0;   //
unsigned char countSeq = 0;    //
unsigned char remainingLives = 3;
unsigned char button;			
unsigned char userInput[9];

unsigned char currentLevel = 2;
unsigned char highestLevel = 0;
unsigned char score = 0;


//Function generate a sequence of 9 random numbers from 1, 2, 4, 8, and 16
void generateSeq()
{
	unsigned char k;
	for (unsigned char i = 0; i < 9; i++)
	{
		k = rand()%5;
		sequence[i] = 0x01 << k;
	}
}

enum SM_LED {Init, Show_Seq, On, Off, WaitInput, GetInput, ledOn, check, waitLow, lvlUp, lives, winState} state;

void tick()
{

	switch(state)
	{
	//-------------------State Transitions---------------------------------
		case Init:
			if(tmpA)
				state = Show_Seq;
		break;
	//---------------------------------------------------------------------	
		case Show_Seq:
			state = On;
		break;
	//---------------------------------------------------------------------
		case On:
			state = Off;
		break;
	//---------------------------------------------------------------------
		case Off:
			if(countLED < currentLevel + 1)
				state = On;
			else
				state = WaitInput;
		break;
	//---------------------------------------------------------------------
		case WaitInput:
			if (tmpA)
				state = GetInput;
			else
			state = WaitInput;
		break;
	//---------------------------------------------------------------------
		case GetInput:
			if(tmpA)
				state = ledOn;	
			else 
				state = GetInput;
		break;
	//---------------------------------------------------------------------
		case ledOn:
			if(tmpA)
				state = ledOn;
			else 
				state = check;
		break;
	//---------------------------------------------------------------------
		case check:
			if(button == sequence[countSeq] && countUser < currentLevel)
			{
				state = GetInput;
				countUser++;
				countSeq++;
			}
			else if(button == sequence[countSeq] && countUser >= currentLevel)
			{
				state = waitLow;

				if (currentLevel == 8)
				{
					state = winState;
				}
			}
			else
				state = lives;
		break;
	//---------------------------------------------------------------------
		case waitLow:
			if (tmpA)
				state = waitLow;
			else
				state = lvlUp;
		break;
	//---------------------------------------------------------------------
		case lvlUp:
			if(tmpA)
				state = Show_Seq;
		break;
	//---------------------------------------------------------------------
		case lives:
			if(remainingLives > 0)
			{
				state = On;
				countSeq = 0;
				countUser = 0;
				countLED = 0;
			} 
			else
			{	
				state = Init;
				LCD_DisplayString(1, "You Lose!");
				LCD_DisplayString2(17, "Highest Level: ");
				score = currentLevel - 1;

				if (score > highestLevel)
					highestLevel = score;
				
				LCD_WriteData(highestLevel + '0');

				remainingLives = 3;
				currentLevel = 2;
				countSeq = 0;
				countUser = 0;
				countLED = 0;
				score = 0;
			}
		break;
	//---------------------------------------------------------------------
		case winState:
			if (tmpA)
				state = Init;
		break;
	//---------------------------------------------------------------------
		default:
			state = Init;
		break;
	//---------------------End of Transitions------------------------------
	}


	switch(state)
	{
	//----------------------State Actions----------------------------------
		case Init:			
				generateSeq();
		break;
	//---------------------------------------------------------------------
		case On:
		{
			LCD_DisplayString(1, "Level ");
			LCD_WriteData((currentLevel - 1) + '0');
			LCD_DisplayString2(17, "Lives: ");
			LCD_WriteData(remainingLives + '0');

			TimerSet(500);

			PORTB = sequence[countLED];

			if (sequence[countLED] == 1)
				set_PWM(Notes[0]);
			else if (sequence[countLED] == 2)
				set_PWM(Notes[1]);
			else if(sequence[countLED] == 4)
				set_PWM(Notes[2]);
			else if(sequence[countLED] == 8)
				set_PWM(Notes[3]);
			else if (sequence[countLED] == 16)
				set_PWM(Notes[4]);
		}
		break;
	//---------------------------------------------------------------------
		case Off:
		{
			PORTB = 0;
			set_PWM(0);
			countLED++;
		}
		break;
	//---------------------------------------------------------------------
		case WaitInput:
			TimerSet(100);
		break;
	//---------------------------------------------------------------------
		case GetInput:
		{
			PORTB = 0;
			set_PWM(0);
		}
		break;
	//---------------------------------------------------------------------
		case ledOn:
		{
			PORTB = tmpA;

			if (tmpA == 1)
				set_PWM(Notes[0]);
			else if (tmpA == 2)
				set_PWM(Notes[1]);
			else if(tmpA == 4)
				set_PWM(Notes[2]);
			else if(tmpA == 8)
				set_PWM(Notes[3]);
			else if (tmpA == 16)
				set_PWM(Notes[4]);
			button = tmpA;
		}
		break;
	//---------------------------------------------------------------------
		case check:
		{
			PORTB = 0;
			set_PWM(0);
		}
		break;
	//---------------------------------------------------------------------
		case waitLow:
		{
				currentLevel++;
				countSeq = 0;
				countUser = 0;
				countLED = 0;
		}
		break;
	//---------------------------------------------------------------------
		case lvlUp:
		{
			LCD_DisplayString2(1, "Press any to continue to level ");
			LCD_WriteData(currentLevel - 1 + '0');
		}
		break;
	//---------------------------------------------------------------------
		case lives:
		{	
			PORTB = 0;
			set_PWM(0);
			remainingLives--;
		}
		break;
	//---------------------------------------------------------------------
		case winState:
		{
			LCD_DisplayString(1, "You Win! Press a button to play again.");
			highestLevel = score;
			remainingLives = 3;
			currentLevel = 2;
			countSeq = 0;
			countUser = 0;
			countLED = 0;
			score = 0;
		}
		break;
	//---------------------------------------------------------------------
		default:
		break;
	//------------------------End Actions----------------------------------
	}

}


int main(void)
{
	srand(time(NULL));

	unsigned long timerPeriod = 100;

	DDRA = 0x00; PORTA = 0xFF;
	DDRB = 0xFF; PORTB = 0x00;
	DDRC = 0xFF; PORTC = 0x00; // LCD data lines
	DDRD = 0xFF; PORTD = 0x00; // LCD control lines

	
	TimerSet(timerPeriod);
	TimerOn();
	LCD_init();
	PWM_on();

	state = Init;
	LCD_DisplayString(4, "SIMON GAME");
	LCD_DisplayString2(17, "press any button");

	while(1)
	{
		tmpA = ~PINA & 0x3F;
		tick();
	
		while (!TimerFlag);
		TimerFlag = 0;
	}
}
