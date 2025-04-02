//
// Copyright (C) 2023 by Daniel Teichmann <daniel@teichm-sh.de>
//
// This package is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; version 2 of the License.
//
// This package is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>

#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"

#include <HTTPClient.h>
#include <HTTPUpdate.h>

#include <ESP32AnalogRead.h>

#include "creds.h"

/*
ESP32 powered Smarthome button
==============================

!!! DON'T FORGOT TO ADJUST THE CREDENTIALS/SETTINGS IN `creds.h` !!!

Hardware Connections
======================
Push Button connected to a GPIO pin pulled down with a 10K Ohm resistor.
Note you can only use the following pins : 0,2,4,12-15,25-27,32-39

You can build a voltage divider between GND and VBAT.
Connect the voltage divider to an ADC pin.
Please use 47k Ohm resistor or slightly more (I've tested 100k Ohm and it worked).
More resistence -> more unstable but longer battery life

*/

#define COMPILE_TIME __DATE__ " " __TIME__
#define SOFTWARE_VERSION "1.0.2"

// Function prototypes
void print_wakeup_reason();
void MQTT_connect();
void update_started();
void update_finished();
void update_progress(int, int);
void update_error(int);
void sub_callback_ota(char*, uint16_t);

ESP32AnalogRead adc;

WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, MQTT_SERVER, MQTT_SERVERPORT, MQTT_USERNAME, MQTT_KEY);

Adafruit_MQTT_Publish   pub_feed_button = Adafruit_MQTT_Publish(&mqtt, HOSTNAME MQTT_BUTTON_TOPIC);
Adafruit_MQTT_Publish   pub_feed_status = Adafruit_MQTT_Publish(&mqtt, HOSTNAME MQTT_STATUS_TOPIC);
Adafruit_MQTT_Subscribe sub_feed_ota    = Adafruit_MQTT_Subscribe(&mqtt, HOSTNAME MQTT_OTA_TOPIC);

long otamode_then_global   = 0;
long otamode_then_reminder = millis();
long otamode_then_led      = millis();
bool otaMode = false;

#define uS_TO_S_FACTOR 1000000ULL  /* Conversion factor for micro seconds to seconds */

RTC_DATA_ATTR int bootCount = 0;
void setup() {
  Serial.begin(115200);
  Serial.printf("Booting for the %i time\n", bootCount++);
  Serial.printf("Software Version: %s\n", SOFTWARE_VERSION);
  Serial.printf("Compile Date: %s\n", COMPILE_TIME);

  pinMode(EXTERNAL_BUTTON_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  adc.attach(EXTERNAL_BAT_ADC_PIN);

  print_wakeup_reason();

  // Status Updates or pressing external button should wake up the ESP32
  Serial.printf("Sending status message every %s seconds.\n", String(TIME_SLEEP_STATUS_UPDATES));
  esp_sleep_enable_timer_wakeup(TIME_SLEEP_STATUS_UPDATES * uS_TO_S_FACTOR);
  Serial.printf("Sending MQTT message to '%s' if external button connected to '%i' gets pressed.\n", HOSTNAME MQTT_BUTTON_TOPIC, EXTERNAL_BUTTON_PIN);
  esp_sleep_enable_ext0_wakeup(EXTERNAL_BUTTON_PIN, HIGH);

  // WiFi Setup
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  if (checkForOTA()) {
    otaMode = true;

    // Hostname defaults to esp3232-[MAC]
    ArduinoOTA.setHostname(HOSTNAME);

    ArduinoOTA
      .onStart([]() {
        String type;
        if (ArduinoOTA.getCommand() == U_FLASH)
          type = "sketch";
        else // U_SPIFFS
          type = "filesystem";

        // NOTE: if updating SPIFFS this would be the place to unmount SPIFFS using SPIFFS.end()
        Serial.println("Start updating " + type);
      })
      .onEnd([]() {
        Serial.println("\nEnd");
      })
      .onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
      })
      .onError([](ota_error_t error) {
        Serial.printf("Error[%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");
      });

      ArduinoOTA.begin();
  }

  sub_feed_ota.setCallback(sub_callback_ota);
  mqtt.subscribe(&sub_feed_ota);

  Serial.println("Setup end.");
}

void sub_callback_ota(char *data, uint16_t len) {
  Serial.printf("--> OTA URL received: '%s'\n", data);
  Serial.printf("Downloading OTA update from: '%s'…\n", data);

  // The line below is optional. It can be used to blink the LED on the board during flashing
  // The LED will be on during download of one buffer of data from the network. The LED will
  // be off during writing that buffer to flash
  // On a good connection the LED should flash regularly. On a bad connection the LED will be
  // on much longer than it will be off. Other pins than LED_BUILTIN may be used. The second
  // value is used to put the LED on. If the LED is on with HIGH, that value should be passed
  httpUpdate.setLedPin(LED_BUILTIN, LOW);

  httpUpdate.onStart(update_started);
  httpUpdate.onEnd(update_finished);
  httpUpdate.onProgress(update_progress);
  httpUpdate.onError(update_error);

  t_httpUpdate_return ret = httpUpdate.update(client, data);

  switch (ret) {
    case HTTP_UPDATE_FAILED:
      Serial.printf("HTTP_UPDATE_FAILED Error (%d): %s\n", httpUpdate.getLastError(), httpUpdate.getLastErrorString().c_str());
      break;

    case HTTP_UPDATE_NO_UPDATES:
      Serial.println("HTTP_UPDATE_NO_UPDATES");
      break;

    case HTTP_UPDATE_OK:
      Serial.println("HTTP_UPDATE_OK");
      break;
  }
}

// Check if OTA mode should be activated
// Returns: true/false
bool checkForOTA() {
  // Check if pulled LOW or HIGH
  bool ACTIVE = EXTERNAL_BUTTON_NORMALLY_LOW ? true : false;

  long then = millis();
  // Hold down button for 3 seconds -> OTA mode activated (LED should blink rapidly)
  while (digitalRead(EXTERNAL_BUTTON_PIN) == ACTIVE) {
    Serial.printf("OTA mode will be activated in %f seconds…\n", (3000.0f - (millis() - then)) / 1000.0f);
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // Flash LED.

    delay(100);

    if (millis() - then >= 3*1000) {
      Serial.println("OTA mode actived.");
      then = millis();
      otamode_then_global   = millis();
      otamode_then_reminder = millis();
      return true;
    }
  }

  return false;
}

float bat_voltage;
float bat_percentage;
void loop() {
  if (otaMode) {
    // Countdown
    if (millis() - otamode_then_reminder >= 10 * 1000) {
      Serial.printf("Waiting for OTA (waited for %i of %i seconds already…)\n", (millis() - otamode_then_global) / 1000, OTA_SEC_WAIT_FOR_UPLOAD);
      Serial.println();
      otamode_then_reminder = millis();
    }

    // Flashing LED
    if (millis() - otamode_then_led >= 1000) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(1000);
      otamode_then_led = millis();
    }

    // Handle Timeout
    if (millis() - otamode_then_global >= OTA_SEC_WAIT_FOR_UPLOAD * 1000) {
      Serial.println("Exiting OTA mode (timeout)…");
      goto deepsleepnow;
    }

    // If we get a OTA URL via MQTT then we will update via an external webserver.
    mqtt.processPackets(1000);
    ArduinoOTA.handle();

    return;
  }

  // Only if someone pressed the external button (which causes the ESP32 to wake up)
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.printf("MQTT (send): '%s': '%s'\n", HOSTNAME MQTT_BUTTON_TOPIC, "pressed");
    MQTT_connect();
    pub_feed_button.publish("pressed");
  }

  bat_voltage    = adc.readVoltage() * 1.7f;
  bat_percentage = ((bat_voltage - 3.1f) / (4.2f - 3.1f)) * 100;

  Serial.printf("Bat Voltage:    %f\n", bat_voltage);
  Serial.printf("Li-Ion Max:     %f\n", 4.2f);
  Serial.printf("Li-Ion Min:     %f\n", 3.1f);
  Serial.printf("Bat Percentage: %f%\n", bat_percentage);

  char buffer[400];
  sprintf(buffer, "{\"bat_voltage\": %f, \"bat_percentage\": %f, \"software_version\": \"%s\"}", bat_voltage, bat_percentage, SOFTWARE_VERSION);

  Serial.printf("MQTT (send): '%s': '%s'\n", HOSTNAME MQTT_STATUS_TOPIC, buffer);
  MQTT_connect();
  pub_feed_status.publish(buffer);
  delay(200);

  Serial.println("Disconnecting from broker…");
  mqtt.disconnect();
  delay(100);

deepsleepnow:
  // Go to deep sleep now and wait for external button to be pressed (or timer wakeup for status messages)
  Serial.println("Going to sleep now…");
  esp_deep_sleep_start();
}

// Function to connect and reconnect as necessary to the MQTT server.
// Should be called in the loop function and it will take care if connecting.
void MQTT_connect() {
  int8_t ret;

  // Stop if already connected.
  if (mqtt.connected()) {
    return;
  }

  Serial.println("Trying to connect to MQTT broker…");
  uint8_t retries = 3;
  while ((ret = mqtt.connect()) != 0) { // connect will return 0 for connected
       Serial.println(mqtt.connectErrorString(ret));
       Serial.println("Retrying MQTT connection in 3 seconds...");
       mqtt.disconnect();
       delay(3000);  // wait 5 seconds
       retries--;
       if (retries == 0) {
         // basically die and wait for WDT to reset me
         while (1);
       }
  }
  Serial.println("Successfully connected to MQTT broker.");
}

// Method to print the reason by which ESP32
// has been awaken from sleep
void print_wakeup_reason() {
  esp_sleep_wakeup_cause_t wakeup_reason;

  wakeup_reason = esp_sleep_get_wakeup_cause();

  switch(wakeup_reason)
  {
    case ESP_SLEEP_WAKEUP_EXT0 : Serial.println("Wakeup caused by external signal using RTC_IO"); break;
    case ESP_SLEEP_WAKEUP_EXT1 : Serial.println("Wakeup caused by external signal using RTC_CNTL"); break;
    case ESP_SLEEP_WAKEUP_TIMER : Serial.println("Wakeup caused by timer"); break;
    case ESP_SLEEP_WAKEUP_TOUCHPAD : Serial.println("Wakeup caused by touchpad"); break;
    case ESP_SLEEP_WAKEUP_ULP : Serial.println("Wakeup caused by ULP program"); break;
    default : Serial.printf("Wakeup was not caused by deep sleep: %d\n", wakeup_reason); break;
  }
}

void update_started() {
  Serial.println("CALLBACK:  HTTP update process started");
}

void update_finished() {
  Serial.println("CALLBACK:  HTTP update process finished");
}

void update_progress(int cur, int total) {
  Serial.printf("CALLBACK:  HTTP update process at %d of %d bytes...\n", cur, total);
}

void update_error(int err) {
  Serial.printf("CALLBACK:  HTTP update fatal error code %d\n", err);
}