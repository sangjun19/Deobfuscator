/* 
 * File:   main.cpp Programme pour test du capteur BME280
 * Author: philippe SIMIER Touchard Washington Le Mans
 *
 * Created on 9 février 2022, 17:11
 * 
 * Bibliothèque : BME280 @ 3.0.0
 * installation : pio lib -g install 901
 * 
 * Utiliser la platform = espressif32 @ ~3.5.0
 * sinon Modifier la bibliothèque fichier BME280.cpp
 * ajouter return true aux methodes
 *        bool BME280::InitializeFilter() 
 *        bool BME280::WriteSettings()
 */

#include <Arduino.h>
#include <Wire.h>
#include <BME280.h>
#include <SPI.h>
#include <BME280I2C.h>


#define SERIAL_BAUD 115200

BME280I2C::Settings parametrage(
        BME280::OSR_X1,
        BME280::OSR_X1,
        BME280::OSR_X1,
        BME280::Mode_Forced,
        BME280::StandbyTime_1000ms,
        BME280::Filter_Off,
        BME280::SpiEnable_False,
        BME280I2C::I2CAddr_0x76 // 0x77 I2C address pour BME 280 Adafruit. (ou 0x76)
        );

BME280I2C bme(parametrage);


void printBME280Data(Stream* client);

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
    } // Wait

    Wire.begin();
    while (!bme.begin()) {
        Serial.println("Could not find BME280 sensor!");
        delay(1000);
    }

    switch (bme.chipModel()) {
        case BME280::ChipModel_BME280:
            Serial.println("Found BME280 sensor! Success.");
            break;
        case BME280::ChipModel_BMP280:
            Serial.println("Found BMP280 sensor! No Humidity available.");
            break;
        default:
            Serial.println("Found UNKNOWN sensor! Error!");
    }
}

void loop() {
    printBME280Data(&Serial);
    delay(500);
}

void printBME280Data(Stream* client) {
    float temp(NAN), hum(NAN), pres(NAN);

    BME280::TempUnit tempUnit(BME280::TempUnit_Celsius);
    BME280::PresUnit presUnit(BME280::PresUnit_Pa);

    bme.read(pres, temp, hum, tempUnit, presUnit);


    client->print("Temp: ");
    client->print(temp);
    client->print(" °C");
    client->print("\tHumidité: ");
    client->print(hum);
    client->print("%");
    client->print("\tPression: ");
    client->print(pres / 100);
    client->println(" hPa");

    delay(1000);
}

