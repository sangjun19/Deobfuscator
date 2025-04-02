#include <WiFi101.h>
#include <ArduinoECCX08.h>
#include "Adafruit_MCP9808.h"
#define LEDOn(L) pinMode(L, OUTPUT);

WiFiSSLClient client;
Adafruit_MCP9808 tempsensor = Adafruit_MCP9808();

void setup()
{
  LEDOn(12);
  WiFi.setPins(7, 22, 1, 23);
  WiFi.begin("<SSID>", "<password>");
  ECCX08.begin();
  String SN = ECCX08.serialNumber();
  ECCX08.end();
  tempsensor.begin(0x18);
  tempsensor.wake();
  String Temp = String(tempsensor.readTempF());
  tempsensor.shutdown();
  pinMode(19, OUTPUT);
  digitalWrite(19, 1);
  delay(3000);
  Serial1.begin(38400);
  for (int i = 0; i < 4; i++)
  {
    digitalWrite(19, !digitalRead(19));
    delay(100);
  }
  LEDOn(13);
  int TempNum = 0, SYS = 0, DIA = 0, PPM = 0;
  char data[20] = "";
  while (TempNum != 16)
  {
    while (Serial1.available())
    {
      char c = Serial1.read();
      if ((c >= 'a' && c <= 'z'))
        sprintf(data, "%s%c", data, c);
      else if (c == '\n')
      {
        if (strstr(data, "endtest") != NULL)
        {
          while (TempNum != 16)
          {
            if (Serial1.available())
            {
              c = Serial1.read();
              switch (c)
              {
              case '0' ... '9':
                TempNum += (c - 48) * (TempNum > 15 ? 1 : 16);
                break;
              case 'A' ... 'F':
                TempNum += (c - 55) * (TempNum > 15 ? 1 : 16);
                break;
              case '\r':
                SYS == 0 ? SYS = TempNum : DIA == 0 ? (DIA = TempNum)
                                                    : (PPM = TempNum);
                TempNum = 0;
                break;
              default:
                break;
              }
            }
          }
        }
        else
          sprintf(data, "%s", "");
      }
    }
  }
  while (WiFi.status() != WL_CONNECTED)
    yield();
  LEDOn(14);
  String Values = "{\"Device\":\"" + SN + "\",\"SYS\":" + String(SYS) + ",\"DIA\":" + String(DIA) + ",\"PPM\":" + String(PPM) + ",\"Temp\":" + Temp + "}";
  const char *host = "script.google.com";
  client.connect(host, 443);
  client.println("POST /macros/s/AKfycbyoSWcckb-yQXup4B3c7EZ3RAdPbsxjj2nrmuuPAOOUwDek2oJTI7J6-94WXY2MDpso/exec HTTP/1.1");
  client.println("Host: " + (String)host);
  client.println("Content-Type: application/json");
  client.println("Content-Length: " + String(Values.length()));
  client.println();
  client.println(Values);
  LEDOn(15);
}

void loop() {}
