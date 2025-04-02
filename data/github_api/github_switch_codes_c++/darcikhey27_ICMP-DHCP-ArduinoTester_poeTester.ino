#include <ICMPPing.h>
#include <util.h>
#include <string.h>
#include <Wire.h>
#include <SPI.h>
#include <Ethernet.h>
#include <Adafruit_RGBLCDShield.h>
#include <utility/Adafruit_MCP23017.h>

// These #defines make it easy to set the backlight color
#define RED 0x1
#define YELLOW 0x3
#define GREEN 0x2
#define TEAL 0x6
#define BLUE 0x4
#define VIOLET 0x5
#define WHITE 0x7


Adafruit_RGBLCDShield lcd;
EthernetClient client;

// the mac for your arduino
byte mac[] = {
  0x00, 0xAA, 0xBB, 0xCC, 0xDE, 0x02
};

IPAddress pingAddr(8, 8, 8, 8 ); // ip address to test ping
//IPAddress pingAddr(192, 168, 252, 252); // ip address to ping
SOCKET pingSocket = 0;
char buffer [256];
ICMPPing ping(pingSocket, (uint16_t)random(0, 255));

String ip = "";

void setup() {

  lcd = Adafruit_RGBLCDShield();
  Serial.begin(9600);
  lcd.begin(16, 2);
  uint8_t i = 0;
  // init ping variables
  printTop("PoE works");
  printBottom("Getting IP");
  lcd.setBacklight(WHITE);
  
// check that an IP was obtained
  if (Ethernet.begin(mac) == 0) {
    Serial.println("Failed to configure Ethernet using DHCP");
    // no point in carrying on, so do nothing forevermore:
    printBottom("DHCP failed");
    lcd.setBacklight(RED);
    for (;;)
      ;
  }
  // print your local IP address:
  printIPAddress();
}

// main loop
void loop() {
  // set the cursor to column 0, line 1
  // (note: line 1 is the second row, since counting begins with 0):

  uint8_t buttons = lcd.readButtons();

  if (buttons) {
    lcd.clear();
    lcd.setCursor(0, 0);

    if (buttons & BUTTON_UP) {
      pinger();

    }
    if (buttons & BUTTON_DOWN) {
      lcd.print("DOWN ");
      lcd.setBacklight(YELLOW);
      showMyIP();
    }
    if (buttons & BUTTON_LEFT) {
      lcd.print("LEFT ");
      lcd.setBacklight(GREEN);
      //renewIP();
    }

    if (buttons & BUTTON_RIGHT) {
      lcd.print("RIGHT ");
      lcd.setBacklight(TEAL);
    }
    
    if (buttons & BUTTON_SELECT) {
      lcd.print("SELECT ");
      lcd.setBacklight(VIOLET);
      showMenu();
    }
  }

  lcd.setBacklight(WHITE);
  //printIPAddress();

} // end loop
void showMyIP() {
  lcd.clear();
  printTop("IP address:");
  printBottom(ip);
}
void pinger() {
  lcd.print("Pinging 8.8.8.8");
  Ethernet.begin(mac, Ethernet.localIP());
  
  for (int i = 0; i < 4; i++) {
    ICMPEchoReply echoReply = ping(pingAddr, 4);

    if (echoReply.status == SUCCESS) {
      sprintf(buffer,
              "Reply[%d] from: %d.%d.%d.%d: bytes=%d time=%ldms TTL=%d",
              echoReply.data.seq,
              echoReply.addr[0],
              echoReply.addr[1],
              echoReply.addr[2],
              echoReply.addr[3],
              REQ_DATASIZE,
              millis() - echoReply.data.time,
              echoReply.ttl);
      lcd.setBacklight(GREEN);
      printBottom("OK");
    }
    else {
      sprintf(buffer, "Echo request failed; %d", echoReply.status);
      Serial.println(Ethernet.localIP());
      lcd.setBacklight(RED);
      printBottom("Ping failed");
    }
    Serial.println(buffer);
    delay(500);
  }

}

void showMenu() {
  printTop("U>newIP,L");
  printBottom("D:showIP, R");
}

void printTop(String msg) {
  lcd.setCursor(0, 0);
  lcd.print(msg);
}

void printBottom(String msg) {
  lcd.setCursor(0, 1);
  lcd.print(msg);
}

void renewIPaddress() {
  /* the Ethernet.maintain() ask for a renewal of dhcp lease, it is not necessary
    0: nothing happened
    1: renew failed
    2: renew success
    3: rebind fail
    4: rebind success
  */
  switch (Ethernet.maintain()) {
    case 1:
      //renewed fail
      Serial.println("Error: renewal failed");
      break;
    case 2:
      //renewed success
      Serial.println("Renewed success");
      //print your local IP address:
      printIPAddress();
      break;
    case 3:
      //rebind fail
      Serial.println("Error: rebind fail");
      break;
    case 4:
      //rebind success
      Serial.println("Rebind success");
      //print your local IP address:
      printIPAddress();
      break;
    default:
      //nothing happened
      break;
  }
}

void printIPAddress() {
  Serial.print("My IP address:");
  //Serial.println(ip);
  //String ip;

  for (byte thisByte = 0; thisByte < 4; thisByte++) {
    // print the value of each byte of the IP address:
    Serial.print(Ethernet.localIP()[thisByte], DEC);
    Serial.print(".");
    ip += Ethernet.localIP()[thisByte];
    ip += ".";
  }
  printTop("IP address:");
  printBottom(ip);
  Serial.println();
}
