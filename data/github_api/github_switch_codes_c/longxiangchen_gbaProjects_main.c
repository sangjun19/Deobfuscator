#include <stdlib.h>
#include <stdio.h>
#include "gba.h"
#include "mode4.h"
#include "start.h"
#include "game.h"
#include "sound.h"

// imports
PLAYER player;
ESCAPE escape;

// prototypes
void initialize();

// state prototypes
void goToStart();
void start();
void goToGame();
void game();
void goToPause();
void pause();
void goToWin();
void win();
void goToLose();
void lose();
void scoreBoard();
void goToScoreboard();

// random prototype
void srand();

// text buffer
char buffer[41];

// states
enum
{
    START,
    GAME,
    PAUSE,
    WIN,
    LOSE,
    Scoreboard
};
int state;

// buttons 
unsigned short buttons;
unsigned short oldButtons;

// random seed
int rSeed;

int main()
{
    initialize();

    while (1)
    {
        // update button variables
        oldButtons = buttons;
        buttons = REG_BUTTONS;

        // state machine
        switch (state) {
            case START:
                start();
                break;
            case GAME:
                game();
                break;
            case PAUSE:
                pause();
                break;
            case WIN:
                win();
                break;
            case LOSE:
                lose();
                break;
            case Scoreboard:
                scoreBoard();
                break;
        }
    }
}

// sets up GBA
void initialize()
{
    unsigned short colors[NUMCOLORS] = {BLACK, GREY, MAROON, GAMEBG, GOLD, BROWN, SALMON, PINK};
    for (int i = 0; i < NUMCOLORS; i++) {
        PALETTE[256-NUMCOLORS+i] = colors[i];
    }
    REG_DISPCTL = MODE(4) | BG2_ENABLE | DISP_BACKBUFFER;
    
    REG_SOUNDCNT_X = SND_ENABLED;

    REG_SOUNDCNT_L = DMG_VOL_LEFT(5) |
                   DMG_VOL_RIGHT(5) |
                   DMG_SND1_LEFT |
                   DMG_SND1_RIGHT |
                   DMG_SND2_LEFT |
                   DMG_SND2_RIGHT |
                   DMG_SND3_LEFT |
                   DMG_SND3_RIGHT |
                   DMG_SND4_LEFT |
                   DMG_SND4_RIGHT;
    REG_SOUNDCNT_H = DMG_MASTER_VOL(2);

    buttons = REG_BUTTONS;
    oldButtons = 0;

    goToStart();
}

// sets up the start state
void goToStart() {
    DMANow(3, startPal, PALETTE, 256);

    drawFullscreenImage4(startBitmap);
    waitForVBlank();    
    flipPage();

    state = START;

    // begin the seed randomization
    rSeed = 0;
}

// runs every frame of the start state
void start() {
    rSeed++;
    // locking frame rate to 60fps
    waitForVBlank();
    if (BUTTON_PRESSED(BUTTON_START)) {
        srand(rSeed); 
        goToGame();
        initGame();
    }

    REG_SND2CNT = DMG_ENV_VOL(0) | DMG_DIRECTION_DECR | DMG_STEP_TIME(0) | DMG_DUTY_50;
    REG_SND2FREQ = NOTE_G6 | SND_RESET | DMG_FREQ_TIMED;
    REG_SND1SWEEP = DMG_SWEEP_NUM(0) | DMG_SWEEP_STEPTIME(0) | DMG_SWEEP_DOWN;
}

// sets up the game state
void goToGame() {
    state = GAME;
}

// Runs every frame of the game state
void game() {
    updateGame();
    drawGame();

    // update the score
    drawString4(10, 1, "time elapsed: ", GOLDID);
    sprintf(buffer, "%d", score);
    drawString4(90, 1, buffer, GOLDID);

    waitForVBlank();
    flipPage();


    if (BUTTON_PRESSED(BUTTON_START)) {
        goToPause();
    }
    // win and lose conditions

    if (!player.lives) {
        goToLose();
    }

    if (collision(player.x, player.y, player.width, player.height, escape.x, escape.y, escape.width, escape.height)) {
        goToWin();
    }
    if (player.dodge) {
        player.dodgeTimer++;
        if (player.dodgeTimer == 50) {
            player.dodgeTimer = 0;
            player.dodge = 0;
            player.dodgeCooldown = 60 * 5;
        }
    }
    player.dodgeCooldown--;

    REG_SND2CNT = DMG_ENV_VOL(0) | DMG_DIRECTION_DECR | DMG_STEP_TIME(0) | DMG_DUTY_50;
    REG_SND2FREQ = NOTE_G6 | SND_RESET | DMG_FREQ_TIMED;
    REG_SND1SWEEP = DMG_SWEEP_NUM(0) | DMG_SWEEP_STEPTIME(0) | DMG_SWEEP_DOWN;
}

// Sets up the pause state
void goToPause() {
    fillScreen4(BROWNID);
    drawString4(136, 8, "got too stressed?", PINKID);
    drawString4(130, 18, "you're paused now!", PINKID); 

    waitForVBlank();
    flipPage();


    state = PAUSE;
}

// Runs every frame of the pause state
void pause() {
    waitForVBlank();
    if (BUTTON_PRESSED(BUTTON_START))
        goToGame();
    else if (BUTTON_PRESSED(BUTTON_SELECT))
        goToScoreboard();
}

// Sets up the win state
void goToWin() {
    if (score < bestTime) {
        bestTime = score;
    }
    fillScreen4(MAROONID);
    drawString4(100, 8, "YOU WON!", GOLDID);
    
    drawString4(30, 18, "PRESS START TO RESTART THE GAME", GOLDID);

    drawString4(56, 28, "YOUR TIME: ", PINKID);
    drawString4(56, 38, "BEST TIME: ", PINKID);
    sprintf(buffer, "%d seconds", score);
    drawString4(127, 28, buffer, PINKID);
    sprintf(buffer, (bestTime != 10000) ? "%d seconds" : "NIL", bestTime);
    drawString4(127, 38, buffer, PINKID);
    // TODO 2.3: wait for vBlank and flip the page
    waitForVBlank();
    flipPage();
    state = WIN;
}

// Runs every frame of the win state
void win() {
    waitForVBlank();
    if (BUTTON_PRESSED(BUTTON_START)) {
        goToStart();
    }
}

// Sets up the lose state
void goToLose() {
    score = 0;
    fillScreen4(SALMONID);
    drawString4(90, 8, "GAME OVER", PINKID);
    drawString4(80, 130, "press start to try again", PINKID);
    drawString4(56, 18, "YOUR TIME: ", PINKID);
    drawString4(56, 28, "BEST TIME: ", PINKID);
    sprintf(buffer, (score == 0) ? "you died" : "%d seconds", score);
    drawString4(126, 18, buffer, PINKID);
    sprintf(buffer, (bestTime != 10000) ? "%d seconds" : "NIL", bestTime);
    drawString4(126, 28, buffer, PINKID);

    // TODO 2.4: wait for vBlank and flip the page
    waitForVBlank();
    flipPage();

    state = LOSE;
}

// Runs every frame of the lose state
void lose() {
    waitForVBlank();
    if (BUTTON_PRESSED(BUTTON_START)) {
        goToStart();
    }
}

void goToScoreboard() {
    fillScreen4(SALMONID);
    drawString4(56, 28, "BEST TIME: ", PINKID);
    sprintf(buffer, (bestTime != 10000) ? "%d seconds" : "NIL", bestTime);
    drawString4(126, 28, buffer, PINKID);
    waitForVBlank();
    flipPage();
    state = Scoreboard;
}

void scoreBoard() {
    waitForVBlank();
    if (BUTTON_PRESSED(BUTTON_START)) {
        goToPause();
    }
}