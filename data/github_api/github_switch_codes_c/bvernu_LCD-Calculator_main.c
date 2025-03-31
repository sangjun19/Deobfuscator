#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#define RS_PIN      (1U << 2)
#define RW_PIN      (1U << 3)
#define EN_PIN      (1U << 4)
#define DATA_PORT   GPIO_PORTB_DATA_R

//GPIO
#define RCGCGPIO (*((volatile unsigned long *)0x400FE608))
#define GPIOAFSEL (*(( volatile unsigned long *)0x40004420))
#define GPIOPCTL (*((volatile unsigned long *)0x4000452C))

//PORT A
#define GPIO_PORTA_DATA_R       (*((volatile unsigned long *)0x400043FC))
#define GPIO_PORTA_DEN_R        (*((volatile unsigned long *)0x4000451C))
#define GPIO_PORTA_DIR_R       (*((volatile unsigned long *)0x40004400))

//port B
#define GPIO_PORTB_DATA_R       (*((volatile unsigned long *)0x400053FC))
#define GPIO_PORTB_DEN_R       (*((volatile unsigned long *)0x4000551C))
#define GPIO_PORTB_DIR_R        (*((volatile unsigned long *)0x40005400))

//PORT C
#define GPIODATA_C (*(( volatile unsigned long *) 0x400063FC))
#define GPIODEN_C (*(( volatile unsigned long *) 0x4000651C))
#define GPIOCR_C (*(( volatile unsigned long *)0x40006524))
#define GPIODIR_C (*(( volatile unsigned long *)0x40006400))

//PORT E
#define GPIOCR_E (*(( volatile unsigned long *)0x40024524))
#define GPIOPDR_E (*((volatile unsigned long *)0x40024514))
#define GPIODIR_E (*(( volatile unsigned long *)0x40024400))
#define GPIODATA_E (*(( volatile unsigned long *) 0x400243FC))
#define GPIODEN_E (*(( volatile unsigned long *) 0x4002451C))

void Delay(int numb){
    unsigned long j = 0;
    for (j = 0; j < numb; j++){
    }
}

void delay_ms(uint32_t millisec) {
    uint32_t i, j;
    for (i = 0; i < millisec; i++)
        for (j = 0; j < 3000; j++){}
}

void WriteToIR(uint8_t command) {
    DATA_PORT = command;           // Send the command
    GPIO_PORTA_DATA_R &= ~(1U << 2);  // RS low for command mode
    GPIO_PORTA_DATA_R &= ~(1U << 3) ;  // Set RW low for write mode
    GPIO_PORTA_DATA_R |= (1U << 4);   // Enable high
    delay_ms(1);
    GPIO_PORTA_DATA_R &= ~(1U << 4);  // Enable low
    delay_ms(3);
}

void WriteToDR(char data) {
    DATA_PORT = data;
    GPIO_PORTA_DATA_R |= (1U << 2);   // RS high
    GPIO_PORTA_DATA_R &= ~(1U << 3) ;  // Set RW low
    GPIO_PORTA_DATA_R |= (1U << 4);   // Enable high
    delay_ms(1);
    GPIO_PORTA_DATA_R &= ~(1U << 4);
    delay_ms(3);
}

void LCD_init() {
    RCGCGPIO |= (1U << 0) | (1U << 1);
    delay_ms(100);

    GPIO_PORTA_DIR_R |= (1U << 2) | (1U << 3) | (1U << 4);
    GPIO_PORTA_DEN_R |= (1U << 2) | (1U << 3) | (1U << 4);

    GPIO_PORTB_DIR_R = 0xFF;
    GPIO_PORTB_DEN_R = 0xFF;

    WriteToIR(0x38);  // 8-bit, 2-line, 5x7 font
    WriteToIR(0x06);  // incrementing cursor
    WriteToIR(0x0F);  // Display, cursor and blinker on
    WriteToIR(0x01);  // clearing the display
    delay_ms(3);
}

void LCD_print(char message) {
    if (message) {
        WriteToDR(message);
    }
}


///last project
void KeyPad_Init(){
    RCGCGPIO |= 0x14;
    GPIODIR_C |= 0xF0;
    GPIODIR_E &= ~0x0F;
    GPIOPDR_E |= 0x0F;
    GPIODEN_C |= 0xF0;
    GPIODEN_E |= 0x0F;
}

const char keypad[4][4] = {
        {'1', '2', '3', 'A'},
        {'4', '5', '6', 'B'},
        {'7', '8', '9', 'C'},
        {'*', '0', '#', 'D'},
    };

char Check_Keypad(void) {
    int col, row;
    for (col = 0; col < 4; col++) {
        GPIODATA_C = (1 << (col + 4));
        for (row = 0; row < 4; row++) {
            if ((GPIODATA_E & (1 << row)) != 0) {
                while((GPIODATA_E & (1 << row)) != 0);
                Delay(50000);
                return keypad[row][col];
            }
        }
    }
    return 0;
}
////end of last project

void clearingtheDisplay() {
    WriteToIR(0x01);
    delay_ms(2);
}

typedef enum {
    InitialState,
    AState,
    BState,
    DisplayState
} CalculatorState;

CalculatorState currentState = InitialState;
uint32_t A = 0;
uint32_t B = 0;
bool buildforA = true;

void resetCalculator() {
    A = 0;
    B = 0;
    buildforA = true;
    clearingtheDisplay();
}

void displayTopRow(char c) {
    LCD_print(c);
}

void giveanswer(uint32_t number) {
    clearingtheDisplay();
    WriteToIR(0xC0);
    int numDigits = 1;
    uint32_t temp = number;
    char buffer[100];
    while (temp /= 10) {
        numDigits++;
    }
    int x = 0;
    while (x < 16) {
        buffer[x] = '0';
        x++;
    }
    int i = numDigits - 1;
    while (i >= 0) {
        buffer[i] = '0' + (number % 10);
        number /= 10;
        i--;
    }
    buffer[numDigits] = '\0';
    int y = 0;
    while (buffer[y] != '\0') {
        LCD_print(buffer[y]);
        y++;
    }
    Delay(5000000);
}



void changingState(CalculatorState newState) {
    switch (newState) {
        case InitialState:
            resetCalculator();
            currentState = AState;
            break;
        case AState:
            clearingtheDisplay();
            currentState = AState;
            break;
        case BState:
            clearingtheDisplay();
            buildforA = false;
            currentState = BState;
            break;
        case DisplayState:
            giveanswer(A * B);
            currentState = AState;
            resetCalculator();
            break;
    }
}

void takingInDig(char input_digit) {
    uint32_t *current_number_ptr = buildforA ? &A : &B;
    *current_number_ptr = *current_number_ptr * 10 + (input_digit - '0');
    displayTopRow(input_digit);
}

void processKeypadInput(char key) {
    switch (key) {
        case 'C':
            changingState(InitialState);
            break;
        case '*':
            changingState(BState);
            break;
        case '#':
            changingState(DisplayState);
            break;
        default:
            if (key >= '0' && key <= '9') {
                takingInDig(key);
            }
            break;
    }
}

int main() {
    LCD_init();
    KeyPad_Init();
    while (1) {
        char c = Check_Keypad();
        if (c != '\0') {
           processKeypadInput(c);
        }
    }
}
