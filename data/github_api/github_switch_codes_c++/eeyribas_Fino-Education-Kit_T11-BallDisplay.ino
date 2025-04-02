// Tennis - Ball Display

// The library belonging to the Fino Education Kit is included. 
#include "Fino.h"

// An object is instantiated from the Fino class.  
Fino fino;

// Definiting rocket length
#define ROCKET_LENGTH   3

// Definiting bouncing variables
#define VERTICAL_BOUNCED    1
#define HORIZONTAL_BOUNCED -1

// Definiting hitting variables
#define NEW_GAME_TRANSITION_TIME  50
#define ADVANCEMENT_TIME_OF_BALL  200
#define HITTING_BALL_NO           0
#define HITTING_BALL_MIDDLE       1
#define HITTING_BALL_LEFT         2
#define HITTING_BALL_RIGHT        3

// Racket is defined along the x-axis.
byte rocket_x;

// Ball is defined the axises.
int ball_x;
int ball_y;
int next_ball_y;

// Defining the arrival time and direction variables of the ball.
byte direction;
int arrival_time;


void setup() 
{
// This code facilitates configuring the hardware settings of the Fino Education Kit. 
// With this code, the settings for buttons, LEDs, screen, speaker, joystick, and sensors of the Fino Education Kit are configured.
  fino.Init();  

// This code ensures the deletion of the image on the screen. 
// Sometimes, image information may remain on the screen, which can be misleading when starting to write a new code program.
  fino.Clear();

// This is the code that adjusts the brightness of the screen. 
// Brightness ranges from 0 to 15.
  fino.SetIntensity(10);

// The initial values of the variables are assigned.
  ball_x = random(1, 7);
  ball_y = 1;
  direction = random(3, 6);
  delay(1500);
  fino.Clear();

// The arrival time of the ball is calculated.
  arrival_time = fino.AllTimer(ADVANCEMENT_TIME_OF_BALL, Movement);
}
 
 
void loop() 
{
// The timer starts.
  fino.Timer();

// Continuous display of the ball on the screen.
  if (next_ball_y != ball_y)
    fino.SetRow(next_ball_y, 0);

// The rocket is positioned according to the value read from the joystick.
  rocket_x = map(fino.ReadJoystickX(), 0, 1020, 8 - ROCKET_LENGTH, -1);

// The LED is controlled based on the joystick's X value.
  if (fino.ReadJoystickX() <= 300) {
    fino.YellowLedHigh();
    fino.RedLedLow();
  } else if (fino.ReadJoystickX() >= 700) {
    fino.YellowLedLow();
    fino.RedLedHigh();
  }

// Drawing a ball on an 8x8 Dot Matrix.
  fino.SetRow(ball_y, byte(1 << (ball_x)));

// The racket coordinates are found.
  byte rocket_coordinate = byte(0xFF >> (8 - ROCKET_LENGTH) << rocket_x);

// Drawing a rcoket on an 8x8 Dot Matrix.
  fino.SetRow(7, rocket_coordinate);

  delay(1500);
  fino.Clear();
  ball_x = random(1, 7);
  ball_y = 1;
  direction = random(3, 6);
  delay(1500);
  fino.Clear();

  delay(10);
}

int BouncingControl() 
{
  if (!ball_x || !ball_y || ball_x == 7 || ball_y == 6) {
    return ((ball_y == 0 || ball_y == 6) ? VERTICAL_BOUNCED : HORIZONTAL_BOUNCED);
  }

  return 0;
}

int Hitting() 
{
  if (ball_y != 6 || ball_x < rocket_x || ball_x > rocket_x + ROCKET_LENGTH)
    return HITTING_BALL_NO;

  if (ball_x == rocket_x + ROCKET_LENGTH / 2)
    return HITTING_BALL_MIDDLE;

  return ball_x < rocket_x + ROCKET_LENGTH / 2 ? HITTING_BALL_LEFT : HITTING_BALL_RIGHT;
}

void Movement() 
{
  int bouncing = BouncingControl();

  if (bouncing) {

    switch (direction) {
      case 0:
        direction = 4;
        break;
      case 1:
        direction = (bouncing == VERTICAL_BOUNCED) ? 7 : 3;
        break;
      case 2:
        direction = 6;
        break;
      case 6:
        direction = 2;
        break;
      case 7:
        direction = (bouncing == VERTICAL_BOUNCED) ? 1 : 5;
        break;
      case 5:
        direction = (bouncing == VERTICAL_BOUNCED) ? 3 : 7;
        break;
      case 3:
        direction = (bouncing == VERTICAL_BOUNCED) ? 5 : 1;
        break;
      case 4:
        direction = 0;
        break;
    }
  }

  switch (Hitting()) {
    case HITTING_BALL_LEFT:
      if (direction == 0) {
        direction =  7;
      } else if (direction == 1) {
        direction = 0;
      }
      break;
    case HITTING_BALL_RIGHT:
      if (direction == 0) {
        direction = 1;
      } else if (direction == 7) {
        direction = 0;
      }
      break;
  }
 
  if ((direction == 0 && ball_x == 0) || (direction == 4 && ball_x == 7))
    direction++;
  if (direction == 0 && ball_x == 7)
    direction = 7;
  if (direction == 4 && ball_x == 0)
    direction = 3;
  if (direction == 2 && ball_y == 0)
    direction = 3;
  if (direction == 2 && ball_y == 6)
    direction = 1;
  if (direction == 6 && ball_y == 0)
    direction = 5;
  if (direction == 6 && ball_y == 6)
    direction = 7;

  if (ball_x == 0 && ball_y == 0)
    direction = 3;
  if (ball_x == 0 && ball_y == 6)
    direction = 1;
  if (ball_x == 7 && ball_y == 6)
    direction = 7;
  if (ball_x == 7 && ball_y == 0)
    direction = 5;
 
  next_ball_y = ball_y;
  if (2 < direction && direction < 6)
    ball_y++;
  else if (direction != 6 && direction != 2)
    ball_y--;

  if (0 < direction && direction < 4)
    ball_x++;
  else if (direction != 0 && direction != 4)
    ball_x--;

  ball_x = max(0, min(7, ball_x));
  ball_y = max(0, min(6, ball_y));
}