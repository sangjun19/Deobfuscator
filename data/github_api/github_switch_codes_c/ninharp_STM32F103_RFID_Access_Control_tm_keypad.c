/**	
 * |----------------------------------------------------------------------
 * | Copyright (C) Tilen Majerle, 2014
 * | 
 * | This program is free software: you can redistribute it and/or modify
 * | it under the terms of the GNU General Public License as published by
 * | the Free Software Foundation, either version 3 of the License, or
 * | any later version.
 * |  
 * | This program is distributed in the hope that it will be useful,
 * | but WITHOUT ANY WARRANTY; without even the implied warranty of
 * | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * | GNU General Public License for more details.
 * | 
 * | You should have received a copy of the GNU General Public License
 * | along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * |----------------------------------------------------------------------
 */
#include "tm_keypad.h"
#include "feedback.h"
#include "dwt.h"

/* Pins configuration, columns are outputs */
#define KEYPAD_COLUMN_1_HIGH		GPIO_SetBits(KEYPAD_COLUMN_1_PORT, KEYPAD_COLUMN_1_PIN)
#define KEYPAD_COLUMN_1_LOW			GPIO_ResetBits(KEYPAD_COLUMN_1_PORT, KEYPAD_COLUMN_1_PIN)
#define KEYPAD_COLUMN_2_HIGH		GPIO_SetBits(KEYPAD_COLUMN_2_PORT, KEYPAD_COLUMN_2_PIN)
#define KEYPAD_COLUMN_2_LOW			GPIO_ResetBits(KEYPAD_COLUMN_2_PORT, KEYPAD_COLUMN_2_PIN)
#define KEYPAD_COLUMN_3_HIGH		GPIO_SetBits(KEYPAD_COLUMN_3_PORT, KEYPAD_COLUMN_3_PIN)
#define KEYPAD_COLUMN_3_LOW			GPIO_ResetBits(KEYPAD_COLUMN_3_PORT, KEYPAD_COLUMN_3_PIN)
#define KEYPAD_COLUMN_4_HIGH		GPIO_SetBits(KEYPAD_COLUMN_4_PORT, KEYPAD_COLUMN_4_PIN)
#define KEYPAD_COLUMN_4_LOW			GPIO_ResetBits(KEYPAD_COLUMN_4_PORT, KEYPAD_COLUMN_4_PIN)

/* Read input pins */
#define KEYPAD_ROW_1_CHECK			(!GPIO_ReadInputDataBit(KEYPAD_ROW_1_PORT, KEYPAD_ROW_1_PIN))
#define KEYPAD_ROW_2_CHECK			(!GPIO_ReadInputDataBit(KEYPAD_ROW_2_PORT, KEYPAD_ROW_2_PIN))
#define KEYPAD_ROW_3_CHECK			(!GPIO_ReadInputDataBit(KEYPAD_ROW_3_PORT, KEYPAD_ROW_3_PIN))
#define KEYPAD_ROW_4_CHECK			(!GPIO_ReadInputDataBit(KEYPAD_ROW_4_PORT, KEYPAD_ROW_4_PIN))
#define KEYPAD_ROW_5_CHECK			(!GPIO_ReadInputDataBit(KEYPAD_ROW_5_PORT, KEYPAD_ROW_5_PIN))

uint8_t KEYPAD_INT_Buttons[5][4] = {
	{0x10, 0x11, 0x12, 0x13},
	{0x07, 0x08, 0x09, 0x0C},
	{0x04, 0x05, 0x06, 0x0D},
	{0x01, 0x02, 0x03, 0x0E},
	{0x0A, 0x00, 0x0B, 0x0F},
};


/* Private functions */
void TM_KEYPAD_INT_SetColumn(uint8_t column);
uint8_t TM_KEYPAD_INT_CheckRow(uint8_t column);
uint8_t TM_KEYPAD_INT_Read(void);

/* Private variables */
static TM_KEYPAD_Button_t KeypadStatus = TM_KEYPAD_Button_NOPRESSED;

void TM_KEYPAD_Init(void) {
	GPIO_InitTypeDef GPIO_InitStructure;
	/* Columns are output */
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	/* Column 1 */
	// Configure as output mode push/pull
	GPIO_InitStructure.GPIO_Pin = KEYPAD_COLUMN_1_PIN;
	GPIO_Init(KEYPAD_COLUMN_1_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_COLUMN_1_CLK, ENABLE);
	/* Column 2 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_COLUMN_2_PIN;
	GPIO_Init(KEYPAD_COLUMN_2_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_COLUMN_1_CLK, ENABLE);
	/* Column 3 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_COLUMN_3_PIN;
	GPIO_Init(KEYPAD_COLUMN_3_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_COLUMN_1_CLK, ENABLE);
	/* Column 4 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_COLUMN_4_PIN;
	GPIO_Init(KEYPAD_COLUMN_4_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_COLUMN_1_CLK, ENABLE);
	
	/* Rows are inputs */
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	/* Row 1 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_ROW_1_PIN;
	GPIO_Init(KEYPAD_ROW_1_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_ROW_1_CLK, ENABLE);
	/* Row 2 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_ROW_2_PIN;
	GPIO_Init(KEYPAD_ROW_2_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_ROW_2_CLK, ENABLE);
	/* Row 3 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_ROW_3_PIN;
	GPIO_Init(KEYPAD_ROW_3_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_ROW_3_CLK, ENABLE);
	/* Row 4 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_ROW_4_PIN;
	GPIO_Init(KEYPAD_ROW_4_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_ROW_4_CLK, ENABLE);
	/* Row 4 */
	GPIO_InitStructure.GPIO_Pin = KEYPAD_ROW_5_PIN;
	GPIO_Init(KEYPAD_ROW_5_PORT, &GPIO_InitStructure);
	RCC_APB2PeriphClockCmd(KEYPAD_ROW_5_CLK, ENABLE);
	
	/* All columns high */
	TM_KEYPAD_INT_SetColumn(0);
}

TM_KEYPAD_Button_t TM_KEYPAD_Read(void) {
	TM_KEYPAD_Button_t temp;
	
	/* Get keypad status */
	temp = KeypadStatus;
	
	/* Reset keypad status */
	KeypadStatus = TM_KEYPAD_Button_NOPRESSED;
	
	return temp;
}

uint8_t Keypad_EnterPin(uint8_t pin_sec[4])
{
	TM_KEYPAD_Button_t Keypad_Button;
	long timerDebounce = millis();
	bool pin_complete = false;
	bool timeoutReached = false;
	uint8_t pin[PIN_LENGTH];
	for (uint8_t i = 0; i < PIN_LENGTH; i++) {
		pin[i] = 0;
	}
	uint8_t pin_pos = 0;
	long timerTimeout = millis();
	bool pin_correct = true;

	while(!pin_complete && !timeoutReached) {
		/* Read keyboard data */
		Keypad_Button = TM_KEYPAD_Read();

		/* Keypad was pressed */
		if (Keypad_Button != TM_KEYPAD_Button_NOPRESSED && ((millis() - timerDebounce) > 100)) {/* Keypad is pressed */
			if ((Keypad_Button < 9) || (Keypad_Button == TM_KEYPAD_Button_ENT))
				KeyFeedback();
			switch (Keypad_Button) {
				case TM_KEYPAD_Button_0:        /* Button 0 pressed */
					pin[pin_pos++] = 0;
					break;
				case TM_KEYPAD_Button_1:        /* Button 1 pressed */
					pin[pin_pos++] = 1;
					break;
				case TM_KEYPAD_Button_2:        /* Button 2 pressed */
					pin[pin_pos++] = 2;
					break;
				case TM_KEYPAD_Button_3:        /* Button 3 pressed */
					pin[pin_pos++] = 3;
					break;
				case TM_KEYPAD_Button_4:        /* Button 4 pressed */
					KeyFeedback();
					pin[pin_pos++] = 4;
					break;
				case TM_KEYPAD_Button_5:        /* Button 5 pressed */
					pin[pin_pos++] = 5;
					break;
				case TM_KEYPAD_Button_6:        /* Button 6 pressed */
					pin[pin_pos++] = 6;
					break;
				case TM_KEYPAD_Button_7:        /* Button 7 pressed */
					pin[pin_pos++] = 7;
					break;
				case TM_KEYPAD_Button_8:        /* Button 8 pressed */
					pin[pin_pos++] = 8;
					break;
				case TM_KEYPAD_Button_9:        /* Button 9 pressed */
					pin[pin_pos++] = 9;
					break;
				case TM_KEYPAD_Button_ENT:        /* Button Enter pressed */
					pin_complete = true;
					break;
				default:
					break;
			}
			if (pin_pos > 3)
				pin_pos = 0;
			Keypad_Button = TM_KEYPAD_Button_NOPRESSED;
			timerDebounce = millis();
			timerTimeout = millis();
		}

		if ((millis()-timerTimeout) >= KEYPAD_INPUT_TIMEOUT) {
			timeoutReached = true;
			return STATUS_TIMEOUT;
		}
	}
	printf("PIN Entered: ");
	for (uint8_t i = 0; i < PIN_LENGTH; i++) {
		if (pin_sec[i] != pin[i])
			pin_correct = false;
		printf("%d", pin[i]);
	}
	printf("\r\n");
	if (pin_correct)
		return STATUS_OK;
	else
		return STATUS_INVALID;
}

/* Private */
void TM_KEYPAD_INT_SetColumn(uint8_t column) {
	/* Set rows high */
	KEYPAD_COLUMN_1_HIGH;
	KEYPAD_COLUMN_2_HIGH;
	KEYPAD_COLUMN_3_HIGH;
	KEYPAD_COLUMN_4_HIGH;
	
	/* Set column low */
	if (column == 1) {
		KEYPAD_COLUMN_1_LOW;
	}
	if (column == 2) {
		KEYPAD_COLUMN_2_LOW;
	}
	if (column == 3) {
		KEYPAD_COLUMN_3_LOW;
	}
	if (column == 4) {
		KEYPAD_COLUMN_4_LOW;
	}
}

uint8_t TM_KEYPAD_INT_CheckRow(uint8_t column) {
	/* Read rows */
	
	/* Scan row 1 */
	if (KEYPAD_ROW_1_CHECK) {
		return KEYPAD_INT_Buttons[0][column - 1];	
	}
	/* Scan row 2 */
	if (KEYPAD_ROW_2_CHECK) {
		return KEYPAD_INT_Buttons[1][column - 1];
	}
	/* Scan row 3 */
	if (KEYPAD_ROW_3_CHECK) {
		return KEYPAD_INT_Buttons[2][column - 1];
	}
	/* Scan row 4 */
	if (KEYPAD_ROW_4_CHECK) {
		return KEYPAD_INT_Buttons[3][column - 1];
	}
	/* Scan row 5 */
	if (KEYPAD_ROW_5_CHECK) {
		return KEYPAD_INT_Buttons[4][column - 1];
	}

	/* Not pressed */
	return KEYPAD_NO_PRESSED;
}

uint8_t TM_KEYPAD_INT_Read(void) {
	uint8_t check;
	/* Set row 1 to LOW */
	TM_KEYPAD_INT_SetColumn(1);
	/* Check rows */
	check = TM_KEYPAD_INT_CheckRow(1);
	if (check != KEYPAD_NO_PRESSED) {
		return check;
	}
	
	/* Set row 2 to LOW */
	TM_KEYPAD_INT_SetColumn(2);
	/* Check columns */
	check = TM_KEYPAD_INT_CheckRow(2);
	if (check != KEYPAD_NO_PRESSED) {
		return check;
	}
	
	/* Set row 3 to LOW */
	TM_KEYPAD_INT_SetColumn(3);
	/* Check columns */
	check = TM_KEYPAD_INT_CheckRow(3);
	if (check != KEYPAD_NO_PRESSED) {
		return check;
	}

	/* Set column 4 to LOW */
	TM_KEYPAD_INT_SetColumn(4);
	/* Check rows */
	check = TM_KEYPAD_INT_CheckRow(4);
	if (check != KEYPAD_NO_PRESSED) {
		return check;
	}
	
	/* Not pressed */
	return KEYPAD_NO_PRESSED;
}

void TM_KEYPAD_Update(void) {
	static uint16_t millis = 0;
	
	/* Every X ms read */
	if (++millis >= KEYPAD_READ_INTERVAL && KeypadStatus == TM_KEYPAD_Button_NOPRESSED) {
		/* Reset */
		millis = 0;
		
		/* Read keyboard */
		KeypadStatus = (TM_KEYPAD_Button_t) TM_KEYPAD_INT_Read();
	}
}

