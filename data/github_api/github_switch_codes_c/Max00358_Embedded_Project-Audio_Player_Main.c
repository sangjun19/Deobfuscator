/*
 * "Hello World" example.
 *
 * This example prints 'Hello from Nios II' to the STDOUT stream. It runs on
 * the Nios II 'standard', 'full_featured', 'fast', and 'low_cost' example
 * designs. It runs with or without the MicroC/OS-II RTOS and requires a STDOUT
 * device in your system's hardware.
 * The memory footprint of this hosted application is ~69 kbytes by default
 * using the standard reference design.
 *
 * For a reduced footprint version of this template, and an explanation of how
 * to reduce the memory footprint for a given application, see the
 * "small_hello_world" template.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <system.h>
#include <sys/alt_alarm.h>
#include <sys/alt_irq.h>
#include <io.h>

#include "fatfs.h"
#include "diskio.h"

#include "ff.h"
#include "monitor.h"
#include "uart.h"

#include "alt_types.h"

#include <altera_up_avalon_audio.h>
#include <altera_up_avalon_audio_and_video_config.h>
#include <altera_avalon_pio_regs.h>

#define ESC 27
#define CLEAR_LCD_STRING "[2J"

FILINFO Finfo;
FATFS *fs;             /* Pointer to file system object */
FATFS Fatfs[_VOLUMES]; /* File system object for each logical drive */
FIL File1;      /* File objects */
DIR Dir;               /* Directory object */

FILE* lcd;
char filenames[20][20];
uint32_t filesizes[20];
int file_count = 0;
volatile int file_index = 0;

volatile int btn_interest = -1;
volatile int btn_state = 0; // 0 = not pressed, 1 = pressed

volatile int stop = 0;
volatile int is_playing = 0;
volatile int is_switched = 0;

uint8_t Buff[1024] __attribute__ ((aligned(4)));  /* Working buffer */
alt_up_audio_dev* audio_dev;
volatile unsigned int iter = 0;

static void put_rc(FRESULT rc);
static int get_btn_pressed(void);
unsigned int byte_to_16_bit(uint8_t* ptr);
void on_btn_released(void);
int ends_with(const char *str, const char *suffix);
int isWav(const char* filename);
void write_lcd(const char *str);
void clear_lcd(void);
void on_stop(void);
char* get_state(void);
void update_lcd(void);
static void btn_interrupt(void* context, alt_u32 id);
static void timer_interrupt(void* context, alt_u32 id);

void normal_play(int blen);
void fast_play(int blen);
void slow_play(int blen);
void left_play(int blen);
void play_curr_buf(uint32_t read_bytes, int skip_bytes, int mono);

static void put_rc(FRESULT rc)
{
    const char *str =
        "OK\0"
        "DISK_ERR\0"
        "INT_ERR\0"
        "NOT_READY\0"
        "NO_FILE\0"
        "NO_PATH\0"
        "INVALID_NAME\0"
        "DENIED\0"
        "EXIST\0"
        "INVALID_OBJECT\0"
        "WRITE_PROTECTED\0"
        "INVALID_DRIVE\0"
        "NOT_ENABLED\0"
        "NO_FILE_SYSTEM\0"
        "MKFS_ABORTED\0"
        "TIMEOUT\0"
        "LOCKED\0"
        "NOT_ENOUGH_CORE\0"
        "TOO_MANY_OPEN_FILES\0";
    FRESULT i;

    for (i = 0; i != rc && *str; i++)
    {
        while (*str++);
    }
    xprintf("rc=%u FR_%s\n", (uint32_t)rc, str);
}

static int get_btn_pressed(void) {
	int val = ~IORD(BUTTON_PIO_BASE, 0);//'~' because buttons are active low
	if (val & 1) {
		return 0;
	} else if (val & 2) {
		return 1;
	} else if (val & 4) {
		return 2;
	} else if (val & 8) {
		return 3;
	}
	return -1;
}

unsigned int byte_to_16_bit(uint8_t* ptr) {
	uint8_t b1 = ptr[0];
	uint8_t b0 = ptr[1];
	unsigned int result = 0;
	result += b0;
	result <<= 8;
	result += b1;
	return result;
}

void update_lcd(void) {
    clear_lcd();
    fprintf(lcd, "\n%d - %s\n", file_index, filenames[file_index]);
    fprintf(lcd, "%s\n", get_state());
}

void on_btn_released(void) {
    char strbuf[32];
    switch(btn_interest)
    {
    case 0:     //skip to next song
        file_index = (file_index + 1) % file_count; //% file_count so that track num never overflows & become larger than file_count
        if ((!stop) && is_playing) {
            is_switched = 1;
        }
        on_stop();
        update_lcd();
        break;
    case 1:     //play or pause
        is_playing = !is_playing;
        update_lcd();
        break;
    case 2:     //stop playing
        on_stop();
        break;
    case 3:     //skip to previous track
        file_index --;
        if (file_index < 0) {
            file_index = file_count-1;
        }
        if ((!stop) && is_playing) {
            is_switched = 1;
        }
        on_stop();
        update_lcd();
        break;
    }
}

int ends_with(const char *str, const char *suffix)//check if song ends with .wav
{
    int str_length = strlen(str);
    int suffix_len = strlen(suffix);
    if (suffix_len > str_length)
    {
        return 0;
    }

    for (int i = 0; i < suffix_len; i++)//compare last 3 char of the two strings
    {
        if (str[str_length - i - 1] != suffix[suffix_len - i - 1])
        {
            return 0;
        }
    }
    return 1;
}

int isWav(const char* filename) {
    return ends_with(filename, ".WAV") || ends_with(filename, ".wav");
}

void write_lcd(const char *str)
{
    fprintf(lcd, "%s\n", str);
}

void clear_lcd()
{
    fprintf(lcd, "%c%s", ESC, CLEAR_LCD_STRING);
}

static void btn_interrupt(void* context, alt_u32 id) {
	IOWR(BUTTON_PIO_BASE, 3, 0x0);  //write 0x0 to offset 3 of BUTTON_PIO_BASE to clear edge capture
	IOWR(TIMER_0_BASE, 1, 0x5);     //
}

static void timer_interrupt(void* context, alt_u32 id) {
	IOWR(TIMER_0_BASE, 0, 0x0);
	int btn = get_btn_pressed(); // -1 indicates a release
    if (btn == -1 && btn_state == 0) {
        return;
    }

	if (btn_interest != -1 && btn_state == 1 && btn == -1) {//if button state is originally pressed and now btn is not pressed, then it's released
        // btn_interest is released
        btn_state = 0;
        on_btn_released();
	} else if (btn_state == 0 && btn != -1) {
		btn_state = 1;
        btn_interest = btn;
        // do something with the button
	} 
}


void play_curr_buf(uint32_t read_bytes, int skip_bytes, int mono) {
	unsigned int r_buf = 0;
	unsigned int l_buf = 0;
    int i = 0;
    while (i < read_bytes && stop != 1) {
        if (i+3 >= read_bytes) {
            break;
        }
        unsigned int lspace = alt_up_audio_write_fifo_space (audio_dev, ALT_UP_AUDIO_RIGHT);
        unsigned int rspace = alt_up_audio_write_fifo_space (audio_dev, ALT_UP_AUDIO_LEFT);
        iter = lspace < rspace ? lspace : rspace; // use the minimum
        for(int j = 0; j < iter && i+3 < read_bytes; j++) {
            l_buf = byte_to_16_bit(Buff+i);
            i += 2;
            r_buf = mono ? 0 : byte_to_16_bit(Buff+i);
            i += 2;
            while (!is_playing);
            alt_up_audio_write_fifo (audio_dev, &(r_buf), 1, ALT_UP_AUDIO_RIGHT);
            alt_up_audio_write_fifo (audio_dev, &(l_buf), 1, ALT_UP_AUDIO_LEFT);
            if (skip_bytes) {
                i += 4;
            }
        }
    }
}

void normal_play(int blen) {
    uint32_t p1 = filesizes[file_index];
    while (p1 > 0)
	{
	// <<<<<<<<<<<<<<<<<<<<<<<<< YOUR fp CODE GOES IN HERE >>>>>>>>>>>>>>>>>>>>>>
    	// 2 channels, with B0 & B1 for left, B2 & B3 for right
        if (stop == 1) {
            return;
        }
    	uint32_t read_bytes;
    	f_read(&File1, Buff, blen, &read_bytes);
        play_curr_buf(read_bytes, 0, 0);
		p1 -= read_bytes;
	}
}

void fast_play(int blen) {
    uint32_t p1 = filesizes[file_index];
    while (p1 > 0)
	{
	// <<<<<<<<<<<<<<<<<<<<<<<<< YOUR fp CODE GOES IN HERE >>>>>>>>>>>>>>>>>>>>>>
    	// 2 channels, with B0 & B1 for left, B2 & B3 for right
        if (stop == 1) {
            return;
        }
    	uint32_t read_bytes;
    	f_read(&File1, Buff, blen, &read_bytes);
        play_curr_buf(read_bytes, 1, 0);
		p1 -= read_bytes;
	}
}

void slow_play(int blen) {
    uint32_t p1 = 2*filesizes[file_index];
    while (p1 > 0)
	{
	// <<<<<<<<<<<<<<<<<<<<<<<<< YOUR fp CODE GOES IN HERE >>>>>>>>>>>>>>>>>>>>>>
    	// 2 channels, with B0 & B1 for left, B2 & B3 for right
        if (stop == 1) {
            return;
        }

    	uint32_t read_bytes;
    	f_read(&File1, Buff, blen/2, &read_bytes);
        
        int dup_index = blen-1;
        int uniq_index = blen/2-1;
        while (dup_index > 0)
        {
            Buff[dup_index] = Buff[uniq_index];
            Buff[dup_index-1] = Buff[uniq_index-1];
            Buff[dup_index-2] = Buff[uniq_index];
            Buff[dup_index-3] = Buff[uniq_index-1];
            dup_index -= 4;
            uniq_index -= 2;
        }
        read_bytes *= 2;

        play_curr_buf(read_bytes, 0, 0);

		p1 -= read_bytes;
	}
}

void left_play(int blen) {
    uint32_t p1 = filesizes[file_index];
    while (p1 > 0)
	{
	// <<<<<<<<<<<<<<<<<<<<<<<<< YOUR fp CODE GOES IN HERE >>>>>>>>>>>>>>>>>>>>>>
    	// 2 channels, with B0 & B1 for left, B2 & B3 for right
        if (stop == 1) {
            return;
        }
    	uint32_t read_bytes;
    	f_read(&File1, Buff, blen, &read_bytes);
        play_curr_buf(read_bytes, 0, 1);
		p1 -= read_bytes;
	}
}

void on_stop(void) {
    if (stop) {
        return;
    }
    is_playing = 1;
    iter = -1;
    stop = 1;
}

char* get_state(void) {
    if (stop) {
        return "Stopped";
    }
    if (is_playing) {
        return "Playing";
    }
    if ((!stop) && !(is_playing)) {
        return "Paused";
    }
    printf("Get State Error: stopped=%d, is_playing=%d\n", stop, is_playing);
    return "error";
}


int main()
{
    uint32_t blen = sizeof(Buff);
	// set up button & timer interrupt
	IOWR(TIMER_0_BASE, 1, 0x0); //clear control
	IOWR(TIMER_0_BASE, 0, 0x0); //clear status
	alt_irq_register(TIMER_0_IRQ, (void*) 0, timer_interrupt);
	IOWR(TIMER_0_BASE, 2, 0xA120); //50 MHz of clock speed, set timeout period
	IOWR(TIMER_0_BASE, 3, 0x7);

	alt_irq_register(BUTTON_PIO_IRQ, (void*) 0, btn_interrupt);
	IOWR(BUTTON_PIO_BASE, 2, 0xF); //0x1: PB0


    printf("Hello from Nios II!\n");
    disk_initialize(0);
    f_mount(0, &Fatfs[0]);

    uint8_t res = f_opendir(&Dir, "");
    if (res)
    {
        // if res in non-zero there is an error; print the error.
        put_rc(res);
        return 0;
    }

    for (int i = 0; i < 20;)
    {
        res = f_readdir(&Dir, &Finfo);
        if ((res != FR_OK) || !Finfo.fname[0])
        {
            break;
        }

        if (Finfo.fattrib & AM_DIR || !isWav(Finfo.fname))
        {
            continue;
        }

        int fn_len = strlen(Finfo.fname);
        strncpy(filenames[i], Finfo.fname, fn_len);
        filenames[i][fn_len] = '\0';
        filesizes[i] = Finfo.fsize;

        i++;
        file_count = i;
    }

    lcd = fopen("/dev/lcd_display", "w");
    if (lcd != NULL) {
        fprintf(lcd, "");
    } else {
        fclose(lcd);
    	return 0;
    }

    stop = 1;
    update_lcd();
    audio_dev = alt_up_audio_open_dev ("/dev/Audio");

    while (1) {
        stop = 0;
        while(is_playing == 0);
        
        f_open(&File1, filenames[file_index], 1);

        int sw_val = IORD(SWITCH_PIO_BASE, 0) & 3;
        switch (sw_val)
        {
        case 0:
            normal_play(blen);
            break;
        case 1:
            slow_play(blen);
            break;
        case 2:
            fast_play(blen);
            break;
        case 3:
            left_play(blen);
            break;
        }
        f_close(&File1);

        is_playing = 0;
        iter = 0;

        if (is_switched) {
            is_playing = 1;
            is_switched = 0;
            stop = 0;
        } else {
            stop = 1;
        }
        update_lcd();
    }

    if (lcd != NULL)
    {
        clear_lcd();
    }
    fclose(lcd);
	IOWR(TIMER_0_BASE, 1, 0x0);
	IOWR(TIMER_0_BASE, 0, 0x0);

    printf("Goodbye from Nios II!\n");
    return 0;
}
