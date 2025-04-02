//*****************************************************************************
//
// Copyright (C) 2014 Texas Instruments Incorporated - http://www.ti.com/ 
// 
// 
//  Redistribution and use in source and binary forms, with or without 
//  modification, are permitted provided that the following conditions 
//  are met:
//
//    Redistributions of source code must retain the above copyright 
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the 
//    documentation and/or other materials provided with the   
//    distribution. 
//
//    Neither the name of Texas Instruments Incorporated nor the names of
//    its contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
//  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
//
//*****************************************************************************


//*****************************************************************************
//
// Application Name     -   Battleships!
// Application Overview -   Fatima Shaik's and Sarbani Kumar's final project for
//                          EEC 172. This project is a recreation of the classic
//                          board game, Battleships, except it uses the remote and
//                          GPIO buttons as modes of user input. Meanwhile the output
//                          of the game is shown on an OLED and email sent by AWS.
// Application Website  - https://fatimazshaik.github.io/Battleships172/
// Application Video - https://www.youtube.com/watch?v=IHRAA0A09kU
//
//*****************************************************************************


//*****************************************************************************
//
//! \addtogroup ssl
//! @{
//
//*****************************************************************************

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Simplelink includes
#include "simplelink.h"

//Driverlib includes
#include "hw_types.h"
#include "hw_ints.h"
#include "spi.h"
#include "hw_nvic.h"
#include "hw_memmap.h"
#include "hw_common_reg.h"
#include "interrupt.h"
#include "hw_apps_rcm.h"
#include "prcm.h"
#include "rom.h"
#include "rom_map.h"
#include "prcm.h"
#include "gpio.h"
#include "systick.h"
#include "utils.h"
#include "uart.h"

//Common interface includes
#include "pin_mux_config.h"
#include "gpio_if.h"
#include "common.h"
#include "uart_if.h"

// Custom includes
#include "utils/network_utils.h"

// Includes Adafruit
#include "Adafruit_GFX.h"
#include "Adafruit_SSD1351.h"
#include "glcdfont.h"

//Include test
#include "oled_test.h"


//FINAL PROJECT MACROS:
#define NUMSMALLSHIPS 2
#define NUMBIGSHIPS 1
#define NUMSHIPS (NUMSMALLSHIPS + NUMBIGSHIPS)
#define ARRAYDIM 6

//STATES:
enum state {SetUp, SendShips, Attack, Wait, EndGame};


//NEED TO UPDATE THIS FOR IT TO WORK!
#define DATE                22    /* Current Date */
#define MONTH               05     /* Month 1-12 */
#define YEAR                2024  /* Current year */
#define HOUR                13    /* Time - hours */
#define MINUTE              47    /* Time - minutes */
#define SECOND              0     /* Time - seconds */


#define APPLICATION_NAME      "SSL"
#define APPLICATION_VERSION   "SQ24"
#define SERVER_NAME           "a3jpkk3le4er78-ats.iot.us-east-1.amazonaws.com" // CHANGE ME - chnaged
#define GOOGLE_DST_PORT       8443


#define POSTHEADER "POST /things/strawberryShortcake/shadow HTTP/1.1\r\n"             // CHANGE ME - chnaged
#define HOSTHEADER "Host: a3jpkk3le4er78-ats.iot.us-east-1.amazonaws.com\r\n"  // CHANGE ME - changed
#define CHEADER "Connection: Keep-Alive\r\n"
#define CTHEADER "Content-Type: application/json; charset=utf-8\r\n"
#define CLHEADER1 "Content-Length: "
#define CLHEADER2 "\r\n\r\n"

#define DATA1 "{" \
            "\"state\": {\r\n"                                              \
                "\"desired\" : {\r\n"                                       \
                    "\"var\" :\""                                           \
                        "Hello phone, "                                     \
                        "message from CC3200 via AWS IoT!"                  \
                        "\"\r\n"                                            \
                "}"                                                         \
            "}"                                                             \
        "}\r\n\r\n"

//*****************************************************************************
//                 GLOBAL VARIABLES -- Start
//*****************************************************************************

#if defined(ccs) || defined(gcc)
extern void (* const g_pfnVectors[])(void);
#endif
#if defined(ewarm)
extern uVectorEntry __vector_table;
#endif

//Button Decoded Strings
static uint8_t zero[33] = "00100000110111110000100011110111";
static uint8_t one[33] = "00100000110111111000100001110111";
static uint8_t up[33] = "00100000110111110100100010110111";
static uint8_t two[33] = "00100000110111110100100010110111";
static uint8_t three[33] = "00100000110111111100100000110111";
static uint8_t right[33] = "00100000110111110010100011010111";
static uint8_t four[33] = "00100000110111110010100011010111";
static uint8_t okay[33] = "00100000110111111010100001010111";
static uint8_t five[33] = "00100000110111111010100001010111";
static uint8_t left[33] = "00100000110111110110100010010111";
static uint8_t six[33] = "00100000110111110110100010010111";
static uint8_t seven[33] = "00100000110111111110100000010111";
static uint8_t down[33] = "00100000110111110001100011100111";
static uint8_t eight[33] = "00100000110111110001100011100111";
static uint8_t nine[33] = "00100000110111111001100001100111";
static uint8_t mute[33] = "00100000110111110101000010101111";
static uint8_t last[33] = "00100000110111110101100010100111";
static volatile uint8_t prevPress[33];

#define MASTER_MSG       "This is CC3200 SPI Master Application\n\r"
//#define APPLICATION_VERSION     "1.4.0"

#define SPI_IF_BIT_RATE  100000
#define TR_BUFF_SIZE     100

static unsigned char g_ucTxBuff[TR_BUFF_SIZE];

// some helpful macros for systick

// the cc3200's fixed clock frequency of 80 MHz
// note the use of ULL to indicate an unsigned long long constant
#define SYSCLKFREQ 80000000ULL

// macro to convert ticks to microseconds
#define TICKS_TO_US(ticks) \
    ((((ticks) / SYSCLKFREQ) * 1000000UL) + \
    ((((ticks) % SYSCLKFREQ) * 1000000UL) / SYSCLKFREQ))\

// macro to convert microseconds to ticks
#define US_TO_TICKS(us) ((SYSCLKFREQ / 1000000ULL) * (us))

// systick reload value set to 40ms period
// (PERIOD_SEC) * (SYSCLKFREQ) = PERIOD_TICKS
//40ms -> 3200000UL
// Ryan changed it to 8800000UL
#define SYSTICK_RELOAD_VAL 8800000UL

// track systick counter periods elapsed
// if it is not 0, we know the transmission ended
volatile int systick_cnt = 0;
volatile int systick_rst = 0;

extern void (* const g_pfnVectors[])(void);

uint64_t incrementer =0;

//Stopwatch Variables
volatile unsigned long long delta = 0;
volatile unsigned long long delta_us = 0;

//IR Variables
volatile unsigned char ir_intflag;

//UART Variables
volatile unsigned char uart_intflag;

//binary buffer variable
volatile unsigned char bufferStart = 0;
static uint8_t buffer[1024];
volatile uint16_t buf_idx = 0;

//oled variables:
//X & Y Position of Transmitter Text
volatile int xtranpos = -6;
volatile int ytranpos = 0;

//X & Y Position of Receiver Text
volatile int xrecpos = -6;
volatile int yrecpos = 64;

//Delay Variables
volatile uint8_t shortDelay = 0;
volatile unsigned long countSysTickHandler = 0;

//Sending buffer variables:
static volatile uint8_t message[1024];
volatile int message_idx = -1;
volatile int prevMessageLen = -1;
volatile int repeatcounter = 0;

//Capitalization Variables:
volatile int capitalizeFlag = 0;
volatile int subtractorCap = 0;

//Array of ASCII
int punctuation[4]= {32, 33, 63, 46};


//Sending buffer variables:
static volatile uint8_t shipLocationString[1024];
volatile uint8_t shipPoints = 0; //number of ship points
volatile uint8_t conflictFlag = 0; //number of ship points
volatile int xposConflicted = 0;
volatile int yposConflicted = 0;
volatile uint8_t playerTurn;

//MORE VARIABLES:
volatile int xpos = 0;
volatile int ypos = 15;
static volatile uint8_t player_name[1024];

// AWS Variables
static volatile uint8_t AWS_message[1024];
volatile int player_win = 0;

//Create Empty Array to store ship locations
volatile int myShipBoard[ARRAYDIM][ARRAYDIM];

//Create an Empty Array to store op ship locations
volatile uint8_t opArrayBoard[ARRAYDIM][ARRAYDIM];

//Create an Empty Array to store attacked areas
volatile uint8_t attackBoard[ARRAYDIM][ARRAYDIM];

//Create an Empty Array to Store Enemy Ship Coords
volatile uint8_t enemyShipCoords[NUMSHIPS][2];

volatile uint8_t shipCounter;
volatile int prevx;
volatile int prevy;

volatile uint8_t xArray; //= OLEDtoArray(xpos)
volatile uint8_t yArray; //= OLEDtoArray(ypos)
static volatile uint8_t myArrayStringBuffer[1024];
static volatile uint8_t opArrayStringBuffer[1024];
volatile uint8_t pos;
volatile uint8_t won = 0;
volatile uint8_t playerTurn;

volatile int bigShipsCounter = 0;
volatile int smallShipsCounter = 0;

volatile int xposBigShip1;
volatile int xposBigShip2;
volatile int yposBigShip1;
volatile int yposBigShip2;

volatile int prevxposBigShip1;
volatile int prevxposBigShip2;
volatile int prevyposBigShip1;
volatile int prevyposBigShip2;


volatile int xArrayBig1;
volatile int xArrayBig2;
volatile int yArrayBig1;
volatile int yArrayBig2;

volatile int vertFlag; //set to 1 when vertical orientation

volatile unsigned char SW3_intflag;
volatile unsigned long SW3_intcount;

//*****************************************************************************
//                 GLOBAL VARIABLES -- End: df
//*****************************************************************************
typedef struct PinSetting {
    unsigned long port;
    unsigned int pin;
} PinSetting;

//Sarbani
//static const PinSetting ir = {.port = GPIOA0_BASE, .pin = GPIO_PIN_6}; //pin 61 ############## CHANGE THIS
//Fatima
static const PinSetting ir = {.port = GPIOA3_BASE, .pin = 0x10}; //pin 18 ############## CHANGE THIS


static const PinSetting switch3 = { .port = GPIOA1_BASE, .pin = 0x20};
//****************************************************************************
//                      LOCAL FUNCTION PROTOTYPES
//****************************************************************************
static int set_time();
static void BoardInit(void);
static int http_post(int);

static int http_post(int iTLSSockID){
    char acSendBuff[512];
    char acRecvbuff[1460];
    char cCLLength[200];
    char* pcBufHeaders;
    int lRetVal = 0;

    pcBufHeaders = acSendBuff;
    strcpy(pcBufHeaders, POSTHEADER);
    pcBufHeaders += strlen(POSTHEADER);
    strcpy(pcBufHeaders, HOSTHEADER);
    pcBufHeaders += strlen(HOSTHEADER);
    strcpy(pcBufHeaders, CHEADER);
    pcBufHeaders += strlen(CHEADER);
    strcpy(pcBufHeaders, "\r\n\r\n");

    int dataLength = strlen(DATA1);

    strcpy(pcBufHeaders, CTHEADER);
    pcBufHeaders += strlen(CTHEADER);
    strcpy(pcBufHeaders, CLHEADER1);

    pcBufHeaders += strlen(CLHEADER1);
    sprintf(cCLLength, "%d", dataLength);

    strcpy(pcBufHeaders, cCLLength);
    pcBufHeaders += strlen(cCLLength);
    strcpy(pcBufHeaders, CLHEADER2);
    pcBufHeaders += strlen(CLHEADER2);

    strcpy(pcBufHeaders, DATA1);
    pcBufHeaders += strlen(DATA1);

    int testDataLength = strlen(pcBufHeaders);

    UART_PRINT(acSendBuff);


    //
    // Send the packet to the server */
    //
    lRetVal = sl_Send(iTLSSockID, acSendBuff, strlen(acSendBuff), 0);
    if(lRetVal < 0) {
        UART_PRINT("POST failed. Error Number: %i\n\r",lRetVal);
        sl_Close(iTLSSockID);
        GPIO_IF_LedOn(MCU_RED_LED_GPIO);
        return lRetVal;
    }
    lRetVal = sl_Recv(iTLSSockID, &acRecvbuff[0], sizeof(acRecvbuff), 0);
    if(lRetVal < 0) {
        UART_PRINT("Received failed. Error Number: %i\n\r",lRetVal);
        //sl_Close(iSSLSockID);
        GPIO_IF_LedOn(MCU_RED_LED_GPIO);
           return lRetVal;
    }
    else {
        acRecvbuff[lRetVal+1] = '\0';
        UART_PRINT(acRecvbuff);
        UART_PRINT("\n\r\n\r");
    }

    return 0;
}

/**
 * Reset SysTick Counter
 */
static inline void SysTickReset(void) {
    // any write to the ST_CURRENT register clears it
    // after clearing it automatically gets reset without
    // triggering exception logic
    // see reference manual section 3.2.1
    HWREG(NVIC_ST_CURRENT) = 1;

    // clear the global count variable
    systick_cnt = 0;
}
static void GPIOA1IntHandler(void) { // SW3 handler
    unsigned long ulStatus;

    ulStatus = MAP_GPIOIntStatus (GPIOA1_BASE, true);
    MAP_GPIOIntClear(GPIOA1_BASE, ulStatus);        // clear interrupts on GPIOA1
    SW3_intcount++;
    SW3_intflag=1;
}

/**
 * SysTick Interrupt Handler
 *
 * Keep track of whether the systick counter wrapped
 */
static void SysTickHandler(void) {
    // increment every time the systick handler fires
    systick_cnt++;
    countSysTickHandler++;

}

/**
 * UART Interrupt Handler
 *
 * Raises UART Flag
 */
void UARTHandler(){
    uart_intflag = 1;
    UARTIntClear(UARTA1_BASE,UART_INT_RX);
    UARTIntEnable(UARTA1_BASE,UART_INT_RX);
}

/**
 * GPIOirHandler Interrupt Handler
 *
 * Raises an IR Flag and parses time that has passed since previous interrupt
 */
static void GPIOirHandler(void){ //IR Handler
    unsigned long ulStatus;
    ulStatus = MAP_GPIOIntStatus (ir.port, true);
    GPIOIntClear(ir.port, ulStatus);       // clear interrupts on GPIOA3

    //grab time measurements
    delta = SYSTICK_RELOAD_VAL - SysTickValueGet();
    delta_us = TICKS_TO_US(delta);

    if((delta_us>13350) && (delta_us<13650)){
        if((countSysTickHandler * 110)<2000){ //keep @ ms level
            shortDelay = 1; //short delay
        }
        else{
            shortDelay = 0; //long delay
        }
        countSysTickHandler = 0;
        bufferStart = 1;
        buf_idx = 0;
    }
    else if(buf_idx==32){
        bufferStart = 0;
        buf_idx = 0;
        countSysTickHandler = 0;
    }
    else if((delta_us>1000) && (delta_us<1250) && bufferStart){
        buffer[buf_idx] = '0';
        buf_idx++;

    }
    else if((delta_us>2000) && (delta_us<2500) && bufferStart){
        buffer[buf_idx] = '1';
        buf_idx++;
    }
    ir_intflag = 1; //set flag = 1
    SysTickReset(); //reset systick
}


//*****************************************************************************
//
//! Board Initialization & Configuration
//!
//! \param  None
//!
//! \return None
//
//*****************************************************************************
static void BoardInit(void) {
/* In case of TI-RTOS vector table is initialize by OS itself */
#ifndef USE_TIRTOS
  //
  // Set vector table base
  //
#if defined(ccs)
    MAP_IntVTableBaseSet((unsigned long)&g_pfnVectors[0]);
#endif
#if defined(ewarm)
    MAP_IntVTableBaseSet((unsigned long)&__vector_table);
#endif
#endif
    //
    // Enable Processor
    //
    MAP_IntMasterEnable();
    MAP_IntEnable(FAULT_SYSTICK);

    PRCMCC3200MCUInit();
}

static void SysTickInit(void) {

    // configure the reset value for the systick countdown register
    MAP_SysTickPeriodSet(SYSTICK_RELOAD_VAL);

    // register interrupts on the systick module
    MAP_SysTickIntRegister(SysTickHandler);

    // enable interrupts on systick
    // (trigger SysTickHandler when countdown reaches 0)
    MAP_SysTickIntEnable();

    // enable the systick module itself
    MAP_SysTickEnable();
}




//*****************************************************************************
//
//! This function updates the date and time of CC3200.
//!
//! \param None
//!
//! \return
//!     0 for success, negative otherwise
//!
//*****************************************************************************

static int set_time() {
    long retVal;

    g_time.tm_day = DATE;
    g_time.tm_mon = MONTH;
    g_time.tm_year = YEAR;
    g_time.tm_sec = HOUR;
    g_time.tm_hour = MINUTE;
    g_time.tm_min = SECOND;

    retVal = sl_DevSet(SL_DEVICE_GENERAL_CONFIGURATION,
                          SL_DEVICE_GENERAL_CONFIGURATION_DATE_TIME,
                          sizeof(SlDateTime),(unsigned char *)(&g_time));

    ASSERT_ON_ERROR(retVal);
    return SUCCESS;
}

//function to calculate OLED cord relative to array
int OLEDtoArray(int OLEDCord){
    int arrayCord = (OLEDCord % 10) - 1;
    return arrayCord;
}


//*****************************************************************************
//
//! Main 
//!
//! \param  none
//!
//! \return None
//!
//***************************************** ************************************
  void main() {
    vertFlag = 0;
    int repeatletterflag = 0;
    long lRetVal = -1;

    // Initialize board configuration
    BoardInit();

    PinMuxConfig();

    unsigned long ulUARTStatus;
    unsigned long ulStatus;

    // Enable the SPI module clock
    MAP_PRCMPeripheralClkEnable(PRCM_GSPI,PRCM_RUN_MODE_CLK);

    InitTerm();
    ClearTerm();

    //Initialize UARTA1 for use
    PRCMPeripheralReset(PRCM_UARTA1);
    MAP_UARTConfigSetExpClk(UARTA1_BASE, MAP_PRCMPeripheralClockGet(PRCM_UARTA1), UART_BAUD_RATE, (UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE | UART_CONFIG_PAR_NONE));
    UARTIntRegister(UARTA1_BASE, UARTHandler); //Register UART Interrupt Handler

    // Reset the peripheral
    MAP_PRCMPeripheralReset(PRCM_GSPI);

    // Initialize the message
    memcpy(g_ucTxBuff,MASTER_MSG,sizeof(MASTER_MSG));

    // Reset SPI
    MAP_SPIReset(GSPI_BASE);

    // Configure SPI interface
    MAP_SPIConfigSetExpClk(GSPI_BASE,MAP_PRCMPeripheralClockGet(PRCM_GSPI),
                                 SPI_IF_BIT_RATE,SPI_MODE_MASTER,SPI_SUB_MODE_0,
                                 (SPI_SW_CTRL_CS |
                                 SPI_4PIN_MODE |
                                 SPI_TURBO_OFF |
                                 SPI_CS_ACTIVEHIGH |
                                 SPI_WL_8));

    // Enable SPI for communication
    MAP_SPIEnable(GSPI_BASE);

    //Initialization of the OLED
    Adafruit_Init();

    // Register the interrupt handlers
    MAP_GPIOIntRegister(ir.port, GPIOirHandler); //Register GPIO Interrupt Handler
    MAP_GPIOIntRegister(GPIOA1_BASE, GPIOA1IntHandler);

    //Configure  GPIO_FALLING_EDGE interrupts on IR Pin
    MAP_GPIOIntTypeSet(ir.port, ir.pin, GPIO_FALLING_EDGE);
    MAP_GPIOIntTypeSet(switch3.port, switch3.pin, GPIO_RISING_EDGE);    // SW3

    //Clear Interrupts
    ulStatus = MAP_GPIOIntStatus(ir.port, false);
    MAP_GPIOIntClear(ir.port, ulStatus); // clear interrupts on ir

    ulUARTStatus = UARTIntStatus(UARTA1_BASE, 0);
    UARTIntClear(UARTA1_BASE, ulUARTStatus);  // clear interrupts on UART

    ulStatus = MAP_GPIOIntStatus(switch3.port, false);
    MAP_GPIOIntClear(switch3.port, ulStatus);           // clear interrupts on GPIOA1

    // clear global variables
    ir_intflag = 0;
    uart_intflag = 0;
    SW3_intcount=0;
        SW3_intflag=0;

    //Enable IR interrupts
    MAP_GPIOIntEnable(ir.port, ir.pin);

    // Enable SW2 and SW3 interrupts
        MAP_GPIOIntEnable(switch3.port, switch3.pin);

    //Enable UART interrupts
    UARTIntEnable(UARTA1_BASE, UART_INT_RX); //Want interrupt to occurs when receive interrupt

    //Enabling UART
    UARTFIFOLevelSet(UARTA1_BASE, UART_FIFO_TX1_8, UART_FIFO_RX1_8);
    UARTEnable(UARTA1_BASE);

    //Enable Systick:
    SysTickInit();

    //Reset Timer:
    SysTickReset();

    // initialize global default app configuration
    g_app_config.host = SERVER_NAME;
    g_app_config.port = GOOGLE_DST_PORT;

    //Connect the CC3200 to the local access point
    lRetVal = connectToAccessPoint();

    //Set time so that encryption can be used
    lRetVal = set_time();

    if(lRetVal < 0) {
        UART_PRINT("Unable to set time in the device");
        LOOP_FOREVER();
    }

    //Connect to the website with TLS encryption
    lRetVal = tls_connect();
    if(lRetVal < 0) {
        ERR_PRINT(lRetVal);
    }

    http_post_message("testing..1...2..3!", lRetVal);
    Report("POST was Successful!!\n\r");

    //*********************** Start Screen **************************//
    fillScreen(BLACK);
    testfastlines(RED, GREEN);
    while(1) {
        testdrawchar("BATTLESHIPS!", WHITE, 35, 64);
        MAP_UtilsDelay(100000);
        break;
    }
    fillScreen(BLACK);
    testdrawchar("What's your name?", WHITE, 0, 0);

    //Game Logic
    while(1){
        if (ir_intflag) {
            ir_intflag = 0;
            if(buf_idx==32){
                if((strcmp(last, buffer)==0) || (strcmp(mute, buffer)==0) || (strcmp(one, buffer)==0)){ //if mute or last
                    //do nothing
                }
                else if((strcmp(buffer, prevPress)==0) && shortDelay){
                    repeatcounter++;
                    drawChar(xpos, ypos, ' ', BLACK, BLACK, 1); //erase prev letter
                    repeatletterflag = 1;
                }
                else{
                    message_idx++;
                    strcpy(prevPress, buffer);
                    repeatcounter = 0;
                    repeatletterflag = 0;
                    if((xpos == 114) && (ypos == 56)){
                        xpos = 0;
                        ypos = 0;
                    }
                    else if(xpos != 114){
                        xpos+=6;
                    }
                    else{
                        xpos = 0;
                        ypos +=8;
                    }
                }
                if(strcmp(zero, buffer)==0){
                    drawChar(xpos, ypos, punctuation[repeatcounter%4], RED, BLACK, 1); //draw space
                    message[message_idx] = punctuation[repeatcounter%4];
                }
                else if(strcmp(one, buffer)==0){
                   if(capitalizeFlag){
                       subtractorCap = 32;
                       capitalizeFlag = 0;
                   }
                   else{
                       subtractorCap = 0;
                       capitalizeFlag = 1;
                   }
                }
                else if(strcmp(two, buffer)==0){
                    drawChar(xpos, ypos, (97-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 97-subtractorCap + (repeatcounter%3); //chnaged 97 to 65
                }
                else if(strcmp(three, buffer)==0){
                    drawChar(xpos, ypos, (100-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 100-subtractorCap + (repeatcounter%3);
                }
                else if(strcmp(four, buffer)==0){
                    drawChar(xpos, ypos, (103-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 103-subtractorCap + (repeatcounter%3);
                }
                else if(strcmp(five, buffer)==0) {
                    drawChar(xpos, ypos, (106-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 106-subtractorCap + (repeatcounter%3);
                }
                else if(strcmp(six, buffer)==0) {
                    drawChar(xpos, ypos, (109-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 109-subtractorCap + (repeatcounter%3);
                }
                else if(strcmp(seven, buffer)==0) {
                    drawChar(xpos, ypos, (112-subtractorCap + (repeatcounter%4)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 112-subtractorCap + (repeatcounter%4);
                }
                else if(strcmp(eight, buffer)==0) {
                    drawChar(xpos, ypos, (116-subtractorCap + (repeatcounter%3)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 116-subtractorCap + (repeatcounter%3);
                }
                else if(strcmp(nine, buffer)==0) {
                    drawChar(xpos, ypos, (119-subtractorCap + (repeatcounter%4)), RED, BLACK, 1); //draw letter
                    message[message_idx] = 119-subtractorCap  + (repeatcounter%4);
                }
                else if(strcmp(last, buffer)==0) { //LAST = Delete
                    drawChar(xpos, ypos, 32, RED, BLACK, 1);
                    if((xpos==0)&&(ypos==0)){
                        xpos = 0;
                        ypos = 0;
                    }
                    else if (xpos!=0){
                        xpos-=6;
                    }
                    else{
                        xpos = 114;
                        ypos -=8;
                    }
                    if(message_idx!=-1){
                        message[message_idx] = ' ';
                        message_idx--;
                    }
                }
                else if(strcmp(mute, buffer)==0) {
                    strcpy(player_name, message);
                    fillScreen(BLACK);
                    break;
                }
            }
        }
    }
    //Clearing Arrays
    int i = 0;
    int j = 0;
    for (i = 0; i< ARRAYDIM; i+=1){
        for (j = 0; j< ARRAYDIM; j+=1){
            myShipBoard[i][j] = 0; //if 0 no ship
            attackBoard[i][j] = 0; //haven't attacked yet
        }
    }
    i = 0;
    j = 0;
    for (i = 0; i< NUMSHIPS; i+=1){
        for (j = 0; j< 2; j+=1){
            enemyShipCoords[i][j] = 0; //if 0 no ship
        }
    }

    enum state currentState = SetUp;

    //fillScreen(BLACK);
    message_idx = -1;
    int delay = 160000;

    while(1){
            if(currentState == SetUp){ //Start Setting Up Ships
                //Initial State of Variables
                shipCounter = 0;
                prevx = 53;
                prevy = 53;
                xpos = 53;
                ypos = 53;
                pos = 0;
                strcpy(myArrayStringBuffer, "");
                strcpy(opArrayStringBuffer, "");
                strcpy(shipLocationString, "");

                //Initial OLED ACTION
                grid();
                fillCircle(xpos, ypos, 5, RED);
                while(shipCounter < NUMSHIPS){
                    while(smallShipsCounter< NUMSMALLSHIPS){
                        if(ir_intflag){ //Detect motion to switch ship location
                            ir_intflag = 0;
                            if(buf_idx==32){
                                if(strcmp(up, buffer)==0){
                                    ypos += 21;
                                    ypos = (ypos >= 116) ? 116 : ypos;
                                }
                                else if(strcmp(down, buffer)==0){
                                    ypos -= 21;
                                    ypos = (ypos <= 11 ) ? 11 : ypos;
                                }
                                else if(strcmp(left, buffer)==0){
                                    xpos -= 21;
                                    xpos = (xpos <= 11) ? 11 : xpos;
                                }
                                else if(strcmp(right, buffer)==0){
                                    xpos += 21;
                                    xpos = (xpos >= 116) ? 116 : xpos;
                                }
                                //convert xpos and ypos to be relative to the position of the the array
                                xArray = OLEDtoArray(xpos);
                                yArray = OLEDtoArray(ypos);

                                //check if spot has been taken
                                if(myShipBoard[xArray][yArray] == 0){
                                    fillCircle(prevx, prevy, 5, BLACK);
                                    fillCircle(xpos, ypos, 5, RED); //FILL CIRCLE
                                    prevx = xpos;
                                    prevy = ypos;
                                }
                                else{
                                    xpos = prevx;
                                    ypos = prevy;
                                }
                                if(strcmp(okay, buffer)==0){
                                    unsigned long ulStatus;
                                    ulStatus = MAP_GPIOIntStatus (ir.port, true);
                                    GPIOIntDisable(ir.port, ulStatus);       // clear interrupts on GPIOA3

                                    shipCounter+=1;
                                    smallShipsCounter+=1;

                                    myShipBoard[xArray][yArray] = 1; //updating array that shows where my ships located

                                    //Storing attack coordinates to send over
                                    shipLocationString[shipPoints] = xArray + '0';
                                    shipPoints+=1;
                                    shipLocationString[shipPoints] = yArray+ '0';
                                    shipPoints+=1;

                                    if(smallShipsCounter< NUMSMALLSHIPS){
                                        do { //choose a new random position for the ship
                                            xpos = (rand() % 6)*21+11;
                                            ypos = (rand() % 6)*21+11;
                                            xArray = OLEDtoArray(xpos);
                                            yArray = OLEDtoArray(ypos);
                                        } while(myShipBoard[xArray][yArray] != 0);

                                        fillCircle(xpos, ypos, 5, RED); //draw new ship
                                        prevx = xpos;
                                        prevy = ypos;
                                        GPIOIntEnable(ir.port, ulStatus);
                                    }
                                }
                            }
                        }
                    }
                    do { //choose a new random position for the ship
                                                   xposBigShip1 = (rand() % 6)*21+11;
                                                   xposBigShip2 = xposBigShip1;
                                                   yposBigShip1 = (rand() % 5)*21+11;
                                                   yposBigShip2 = yposBigShip1 + 21;
                                                   xArrayBig1 = OLEDtoArray(xposBigShip1);
                                                   xArrayBig2 = OLEDtoArray(xposBigShip2);
                                                   yArrayBig1 = OLEDtoArray(yposBigShip1);
                                                   yArrayBig2 = OLEDtoArray(yposBigShip2);
                                               } while((myShipBoard[xArrayBig1][yArrayBig1] != 0) && (myShipBoard[xArrayBig2][yArrayBig2] != 0) );
                                               fillCircle(xposBigShip1, yposBigShip1, 5, BLUE); //draw new ship
                                               fillCircle(xposBigShip2, yposBigShip2, 5, BLUE); //draw new ship

                                               prevxposBigShip1 = xposBigShip1;
                                               prevxposBigShip2 = xposBigShip2;
                                               prevyposBigShip1 = yposBigShip1;
                                               prevyposBigShip2 = yposBigShip2;
                    while(bigShipsCounter< NUMBIGSHIPS){

                        if (SW3_intflag) {
                            SW3_intflag=0;  // clear flag
                            if(vertFlag == 0){ //not vertical
                                if(xposBigShip2 < 116){
                                    prevxposBigShip1 = xposBigShip1;
                                                                prevxposBigShip2 = xposBigShip2;
                                                                prevyposBigShip1 = yposBigShip1;
                                                                prevyposBigShip2 = yposBigShip2;
                                    vertFlag = 1;
                                    yposBigShip1 = yposBigShip2;
                                    xposBigShip1 = xposBigShip2 + 21;
                                    fillCircle(prevxposBigShip1, prevyposBigShip1, 5, BLACK); //erase old ship
                                                                fillCircle(prevxposBigShip2, prevyposBigShip2, 5, BLACK); //erase old ship
                                                                fillCircle(xposBigShip1, yposBigShip1, 5, BLUE); //draw new ship
                                                                fillCircle(xposBigShip2, yposBigShip2, 5, BLUE); //draw new ship
                                }
                            }
                            else{
                                if(yposBigShip2 > 11){ //&& xposBigShip2 < 116
                                    prevxposBigShip1 = xposBigShip1;
                                                                prevxposBigShip2 = xposBigShip2;
                                                                prevyposBigShip1 = yposBigShip1;
                                                                prevyposBigShip2 = yposBigShip2;
                                    vertFlag = 0;
                                    xposBigShip1 = xposBigShip2;
                                    yposBigShip1 = yposBigShip2 - 21;
                                    fillCircle(prevxposBigShip1, prevyposBigShip1, 5, BLACK); //erase old ship
                                                                fillCircle(prevxposBigShip2, prevyposBigShip2, 5, BLACK); //erase old ship
                                                                fillCircle(xposBigShip1, yposBigShip1, 5, BLUE); //draw new ship
                                                                fillCircle(xposBigShip2, yposBigShip2, 5, BLUE); //draw new ship
                                }
                            }

                        }

                        if(ir_intflag){ //Detect motion to switch ship location
                            ir_intflag = 0;
                            if(buf_idx==32){
                                prevxposBigShip1 = xposBigShip1;
                                prevxposBigShip2 = xposBigShip2;
                                prevyposBigShip1 = yposBigShip1;
                                prevyposBigShip2 = yposBigShip2;
                                if(strcmp(up, buffer)==0){
                                    if(vertFlag == 0){
                                        yposBigShip2 += 21;
                                        yposBigShip2 = (yposBigShip2 >= 116) ? 116 : yposBigShip2;
                                        yposBigShip1 = yposBigShip2- 21;
                                    }
                                    else{
                                        yposBigShip2 +=21;
                                        yposBigShip2 = (yposBigShip2 >= 116) ? 116 : yposBigShip2;
                                        yposBigShip1 = yposBigShip2;
                                    }
                                }
                                else if(strcmp(down, buffer)==0){
                                    if(vertFlag == 0){
                                        yposBigShip1 -= 21;
                                        yposBigShip1 = (yposBigShip1 <= 11 ) ? 11 : yposBigShip1;
                                        yposBigShip2 = yposBigShip1 + 21;
                                    }
                                    else{
                                        yposBigShip2 -=21;
                                        yposBigShip1 = (yposBigShip1 <= 11 ) ? 11 : yposBigShip1;
                                        yposBigShip1 = yposBigShip2;
                                    }
                                }
                                else if(strcmp(left, buffer)==0){
                                    if(vertFlag == 0){
                                        xposBigShip1 -= 21;
                                        xposBigShip1 = (xposBigShip1 <= 11) ? 11 : xposBigShip1;
                                        xposBigShip2 = xposBigShip1;
                                                                    }
                                    else{
                                        xposBigShip1 -= 21;
                                        xposBigShip1 = (xposBigShip1 <= 11) ? 11 : xposBigShip1;
                                        if(xposBigShip1==11){
                                            xposBigShip2 = 32;
                                        }
                                        else{
                                            xposBigShip2 = xposBigShip1 + 21;
                                        }
                                        yposBigShip1 = yposBigShip2;
                                    }
                                }
                                else if(strcmp(right, buffer)==0){
                                    if(vertFlag == 0){
                                        xposBigShip2 += 21;
                                        xposBigShip2 = (xposBigShip2 >= 116) ? 116 : xposBigShip2;
                                        xposBigShip1 = xposBigShip2;
                                    }
                                    else{
                                        xposBigShip2 += 21;
                                        xposBigShip1 = (xposBigShip1 <= 116) ? 116 : xposBigShip1;
                                        xposBigShip1 = xposBigShip2 -21;
                                        yposBigShip1 = yposBigShip2;
                                    }

                                }

                                //convert xpos and ypos to be relative to the position of the the array
                                xArrayBig1 = OLEDtoArray(xposBigShip1);
                                xArrayBig2 = OLEDtoArray(xposBigShip2);
                                yArrayBig1 = OLEDtoArray(yposBigShip1);
                                yArrayBig2 = OLEDtoArray(yposBigShip2);

                                //check if spot has been taken
                                if((myShipBoard[xArrayBig1][yArrayBig1] == 0) && (myShipBoard[xArrayBig2][yArrayBig2] == 0) ){
                                    fillCircle(prevxposBigShip1, prevyposBigShip1, 5, BLACK); //erase old ship
                                    fillCircle(prevxposBigShip2, prevyposBigShip2, 5, BLACK); //erase old ship
                                    fillCircle(xposBigShip1, yposBigShip1, 5, BLUE); //draw new ship
                                    fillCircle(xposBigShip2, yposBigShip2, 5, BLUE); //draw new ship

                                    prevxposBigShip1 = xposBigShip1;
                                    prevxposBigShip2 = xposBigShip2;
                                    prevyposBigShip1 = yposBigShip1;
                                    prevyposBigShip2 = yposBigShip2;
                                }
                                else{
                                    xposBigShip1 = prevxposBigShip1;
                                    xposBigShip2 = prevxposBigShip2;
                                    yposBigShip1 = prevyposBigShip1;
                                    yposBigShip2 = prevyposBigShip2;
                                }//otherwise don't change it
                                if(strcmp(okay, buffer)==0){ //Successfully put down a ship
                                     shipCounter+=1;
                                     bigShipsCounter+=1;

                                     myShipBoard[xArrayBig1][yArrayBig1] = 1; //updating array that shows where my ships located
                                     myShipBoard[xArrayBig2][yArrayBig2] = 1;

                                     //Storing attack coordinates to send over
                                     shipLocationString[shipPoints] = xArrayBig1 + '0';
                                     shipPoints+=1;
                                     shipLocationString[shipPoints] = yArrayBig1+ '0';
                                     shipPoints+=1;
                                     shipLocationString[shipPoints] = xArrayBig2 + '0';
                                     shipPoints+=1;
                                     shipLocationString[shipPoints] = yArrayBig2+ '0';
                                     shipPoints+=1;
                                }
                            }
                        }
                    }


                }
                if(shipCounter >= NUMSHIPS){
                    fillCircle(xpos, ypos, 5, BLACK); //draw new ship
                    UARTCharPutNonBlocking(UARTA1_BASE, 't');
                    for(i=0; i <= shipPoints; i++){ //shipLocationString
                        UARTCharPutNonBlocking(UARTA1_BASE, shipLocationString[i]);
                        Report("SENT shipLocationString[%d]: %c\n\r", i, shipLocationString[i]);
                    }
                    UARTCharPutNonBlocking(UARTA1_BASE, '\0');
                    i=0;
                    currentState = SendShips;
               }

            } //End Setting up Ships State

            //Sending Ships Over
            if(currentState == SendShips){
                while(uart_intflag==0){};
                if(uart_intflag){
                    uart_intflag =0;
                    while(UARTCharsAvail(UARTA1_BASE)){
                        char c = UARTCharGetNonBlocking(UARTA1_BASE);
                            pos += sprintf(&opArrayStringBuffer[i], "%c", c);  // Add each element and a space
                            i+=1;
                            if(c=='t'){
                                i=0;
                            }
                    }
                }
                if(strlen(opArrayStringBuffer) == shipPoints){
                    for (i = 0; i< (shipPoints/2); i+=1){
                                for (j = 0; j< 2; j+=1){
                                    enemyShipCoords[i][j] = opArrayStringBuffer[i*2 + j]- '0';
                                }
                    }
                    for (i = 0; i< (shipPoints/2); i+=1){
                        opArrayBoard[enemyShipCoords[i][0]][enemyShipCoords[i][1]] = 1;
                    }
                    //Clear Board
                    fillScreen(BLACK);
                    grid();

                    //Initial X and Y position
                    prevx = 53;
                    prevy = 53;
                    xpos = 53;
                    ypos = 53;
                    shipCounter = 0;
                    fillCircle(xpos, ypos, 5, YELLOW);
                    currentState = Attack;
                }
            }
            if(currentState == Attack){
                if(ir_intflag){
                    Report("BYE\n\r");
                    ir_intflag = 0;
                    if(buf_idx==32){
                        if(conflictFlag == 1){
                            if( (attackBoard[xArray][yArray] == 1) && (opArrayBoard[xArray][yArray] == 1)){
                                fillCircle(xposConflicted, yposConflicted, 5, GREEN);
                            }else{
                                fillCircle(xposConflicted, yposConflicted, 5, WHITE);
                            }
                        }
                        if(strcmp(up, buffer)==0){
                            ypos += 21;
                            ypos = (ypos >= 116) ? 116 : ypos;
                        }
                        else if(strcmp(down, buffer)==0){
                            ypos -= 21;
                            ypos = (ypos <= 11) ? 11 : ypos;
                        }
                        else if(strcmp(left, buffer)==0){
                            xpos -= 21;
                            xpos = (xpos <= 11) ? 11 : xpos;
                        }
                        else if(strcmp(right, buffer)==0){
                            xpos += 21;
                            xpos = (xpos >= 116) ? 116 : xpos;
                        }
                        xArray = OLEDtoArray(xpos);
                        yArray = OLEDtoArray(ypos);

                        if(attackBoard[xArray][yArray] == 0){
                            conflictFlag = 0;
                            fillCircle(prevx, prevy, 5, BLACK);
                            fillCircle(xpos, ypos, 5, YELLOW);
                            prevx = xpos;
                            prevy = ypos;
                        }
                        else{
                            xposConflicted = xpos;
                            yposConflicted = ypos;
                            fillCircle(prevx, prevy, 5, BLACK);
                            fillCircle(xpos, ypos, 3, RED);
                            conflictFlag = 1;

                        }
                        if((strcmp(okay, buffer)==0) && conflictFlag == 0){ //Successfully put down a ship
                            xArray = OLEDtoArray(xpos);
                            yArray = OLEDtoArray(ypos);
                            attackBoard[xArray][yArray] = 1;
                            if(opArrayBoard[xArray][yArray] == 1){
                                shipCounter +=1;
                                fillCircle(xpos, ypos, 5, GREEN); //draw new ship
                            }
                            else{
                                fillCircle(xpos, ypos, 5, WHITE); //draw new ship
                            }
                            do { //choose a new random position for the ship
                                xpos = (rand() % 6)*21+11;
                                ypos = (rand() % 6)*21+11;
                                xArray = OLEDtoArray(xpos);
                                yArray = OLEDtoArray(ypos);
                                } while(attackBoard[xArray][yArray] != 0);
                            fillCircle(xpos, ypos, 5, YELLOW); //draw new ship
                            prevx = xpos;
                            prevy = ypos;
                            char send = 'a';
                            UARTCharPutNonBlocking(UARTA1_BASE, send);
                        }
                    }
                }

                if(uart_intflag){
                    uart_intflag =0;
                    char playerTurn;
                    while(UARTCharsAvail(UARTA1_BASE)){
                        playerTurn = UARTCharGetNonBlocking(UARTA1_BASE);
                    }
                }

                if(shipCounter >= (shipPoints/2)){
                    char send = 's';
                    UARTCharPutNonBlocking(UARTA1_BASE, send);
                }
                if(uart_intflag){
                    uart_intflag =0;
                    while(UARTCharsAvail(UARTA1_BASE)){
                        playerTurn = UARTCharGetNonBlocking(UARTA1_BASE);
                    }
                }
                if(playerTurn == 's'){
                    fillScreen(BLACK);
                    testdrawchar("ENDGAME YOU WON!", WHITE, 35, 64);
                    char send = 's';
                    UARTCharPutNonBlocking(UARTA1_BASE, send);
                    strcpy(AWS_message, player_name);
                    strcat(AWS_message, " won!");
                    http_post_message(AWS_message, lRetVal);
                    break;
                }
            }
        }
}

  static int http_post_message(char *message, int iTLSSockID){
      char acSendBuff[512];
      char acRecvbuff[1460];
      char cCLLength[200];
      char postMessage[1024];
      char* pcBufHeaders;
      int lRetVal = 0;

      pcBufHeaders = acSendBuff;
      strcpy(pcBufHeaders, POSTHEADER);
      pcBufHeaders += strlen(POSTHEADER);
      strcpy(pcBufHeaders, HOSTHEADER);
      pcBufHeaders += strlen(HOSTHEADER);
      strcpy(pcBufHeaders, CHEADER);
      pcBufHeaders += strlen(CHEADER);
      strcpy(pcBufHeaders, "\r\n\r\n");


      //do concatenation n store in a variable postMessage
      snprintf(postMessage, sizeof(postMessage), "{\"state\": {\r\n\"desired\" : {\r\n\"var\" :\"%s\"\r\n}}}\r\n\r\n", message);

      int dataLength = strlen(postMessage);

      strcpy(pcBufHeaders, CTHEADER);
      pcBufHeaders += strlen(CTHEADER);
      strcpy(pcBufHeaders, CLHEADER1);

      pcBufHeaders += strlen(CLHEADER1);
      sprintf(cCLLength, "%d", dataLength);

      strcpy(pcBufHeaders, cCLLength);
      pcBufHeaders += strlen(cCLLength);
      strcpy(pcBufHeaders, CLHEADER2);
      pcBufHeaders += strlen(CLHEADER2);

      strcpy(pcBufHeaders, postMessage);
      pcBufHeaders += strlen(postMessage);

      int testDataLength = strlen(pcBufHeaders);

      UART_PRINT(acSendBuff);

      //
      // Send the packet to the server */
      //
      lRetVal = sl_Send(iTLSSockID, acSendBuff, strlen(acSendBuff), 0);
      if(lRetVal < 0) {
          UART_PRINT("POST failed. Error Number: %i\n\r",lRetVal);
          sl_Close(iTLSSockID);
          GPIO_IF_LedOn(MCU_RED_LED_GPIO);
          return lRetVal;
      }
      lRetVal = sl_Recv(iTLSSockID, &acRecvbuff[0], sizeof(acRecvbuff), 0);
      if(lRetVal < 0) {
          UART_PRINT("Received failed. Error Number: %i\n\r",lRetVal);
          //sl_Close(iSSLSockID);
          GPIO_IF_LedOn(MCU_RED_LED_GPIO);
             return lRetVal;
      }
      else {
          acRecvbuff[lRetVal+1] = '\0';
          UART_PRINT(acRecvbuff);
          UART_PRINT("\n\r\n\r");
      }

      return 0;
  }

//*****************************************************************************
//
// Close the Doxygen group.
//! @}
//
//*****************************************************************************
