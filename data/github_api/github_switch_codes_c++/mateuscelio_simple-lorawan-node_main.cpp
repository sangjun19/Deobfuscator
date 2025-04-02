#include "Adafruit_BME280.h"
#include "Arduino.h"
#include "FlashStorage_STM32.h"
#include "SPI.h"
#include "STM32LowPower.h"
#include "arduino_lmic.h"
#include "hal/hal.h"

#define LORA_PORT 1
#define EEPROM_FRAME_COUT_ADDR 0

u1_t NWKSKEY[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF};

u1_t APPSKEY[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF};

u4_t DEVADDR = 0x260CAEBC;

// PINS
// PA7 - MOSI
// PA6 - MISO
// PA5 - SCK
// PA4 - NSS
// PB6 - RST
// PB3 - DI00
// PA15 - DI01
//
#define RADIO_MOSI PA7
#define RADIO_MISO PA6
#define RADIO_SCK PA5
#define RADIO_NSS PA4
#define RADIO_RST PB5
#define RADIO_DI00 PB3
#define RADIO_DI01 PA15

#define SENSOR_SDA PB7
#define SENSOR_SCL PB6

const lmic_pinmap lmic_pins = {
    .nss = RADIO_NSS,
    .rst = RADIO_RST,
    .dio = {RADIO_DI00, RADIO_DI01, LMIC_UNUSED_PIN}};

TwoWire sensor_i2c_interface;
Adafruit_BME280 sensor_bme;

void os_getArtEui(u1_t *buf) {}
void os_getDevEui(u1_t *buf) {}
void os_getDevKey(u1_t *buf) {}

void do_send(osjob_t *j);
void sleep_node();
u4_t get_frame_count();
void save_frame_count(u4_t frame_count);
void read_sensor();

void setup_radio(u1_t sub_band = 1);
unsigned char lora_data_buffer[8] = {};
static osjob_t sendjob;

void setup() {

  Serial.begin(115200);

  sensor_i2c_interface.begin(SENSOR_SDA, SENSOR_SCL);

  if (sensor_bme.begin(0x76, &sensor_i2c_interface)) {
    Serial.println("Sensor initialized!");
    sensor_bme.setSampling(Adafruit_BME280::MODE_FORCED);
  } else {
    Serial.println("Failed to initialize sensor");
  };

  pinMode(PB12, OUTPUT);
  digitalWrite(PB12, LOW);

  LowPower.begin();
  read_sensor();

  setup_radio();
  do_send(&sendjob);
}

void loop() { os_runloop_once(); }

void setup_radio(u1_t sub_band) {
  os_init();
  LMIC_reset();
  // Use only channel 0 of sub_band

  // enable all channels of sub_band and disable others
  LMIC_selectSubBand(sub_band);
  //
  LMIC_disableSubBand(sub_band);

  LMIC_enableChannel(0 + sub_band * 8);

  //  LMIC_selectSubBand(sub_band);
  //  for (int i = 1; i < 9; i++) {
  //    LMIC_disableChannel(i + 8);
  //  }
  LMIC_setSession(0x13, DEVADDR, NWKSKEY, APPSKEY);

  u4_t fc = get_frame_count();
  if (fc != 0xffffffff)
    LMIC_setSeqnoUp(fc);

  LMIC_setDrTxpow(DR_SF10, 14);
}

void read_sensor() {

  uint16_t temperature = (sensor_bme.readTemperature() + 273.15) * 100;
  uint16_t humidity = sensor_bme.readHumidity() * 100;
  uint32_t pressure = sensor_bme.readPressure();

  Serial.print(F("Temperature = "));
  Serial.print(temperature);
  Serial.println(" *K");

  Serial.print(F("Humidity = "));
  Serial.print(humidity);
  Serial.println(" %");

  Serial.print(F("Pressure = "));
  Serial.print(pressure);
  Serial.println(" hPa");
  Serial.println();
  Serial.println(sensor_bme.readPressure());

  lora_data_buffer[0] = temperature >> 8;
  lora_data_buffer[1] = temperature;

  lora_data_buffer[2] = humidity >> 8;
  lora_data_buffer[3] = humidity;

  lora_data_buffer[4] = pressure >> 24;
  lora_data_buffer[5] = pressure >> 16;
  lora_data_buffer[6] = pressure >> 8;
  lora_data_buffer[7] = pressure;

  for (int i = 0; i < sizeof(lora_data_buffer); i++) {
    if (i % 4 == 0)
      Serial.print(" ");
    Serial.print(lora_data_buffer[i], HEX);
  }
  Serial.println("");
}

void do_send(osjob_t *j) {
  lmic_tx_error_t error = 0;

  // Check if there is not a current TX/RX job running
  if (LMIC.opmode & OP_TXRXPEND) {
    Serial.println("OP_TXRXPEND, not sending\r\n");
  } else {
    Serial.println("Packet queued\r\n");
    // Prepare upstream data transmission at the next possible time.
    error = LMIC_setTxData2(LORA_PORT, lora_data_buffer,
                            sizeof(lora_data_buffer), 0);
  }

  switch (error) {
  case LMIC_ERROR_TX_BUSY:
    Serial.println("LMIC_ERROR_TX_BUSY\r\n");
    break;
  case LMIC_ERROR_TX_TOO_LARGE:
    Serial.println("LMIC_ERROR_TX_TOO_LARGE\r\n");
    break;
  case LMIC_ERROR_TX_NOT_FEASIBLE:
    Serial.println("LMIC_ERROR_TX_NOT_FEASIBLE\r\n");
    break;
  case LMIC_ERROR_TX_FAILED:
    Serial.println("LMIC_ERROR_TX_FAILED\r\n");
    break;
  default:
    break;
  }
}

void onEvent(ev_t ev) {
  Serial.print(os_getTime());
  Serial.print(": ");
  switch (ev) {
    //  case EV_SCAN_TIMEOUT:
    //    Serial.println(F("EV_SCAN_TIMEOUT"));
    //    break;
    //  case EV_BEACON_FOUND:
    //    Serial.println(F("EV_BEACON_FOUND"));
    //    break;
    //  case EV_BEACON_MISSED:
    //    Serial.println(F("EV_BEACON_MISSED"));
    //    break;
    //  case EV_BEACON_TRACKED:
    //    Serial.println(F("EV_BEACON_TRACKED"));
    //    break;
    //  case EV_JOINING:
    //    Serial.println(F("EV_JOINING"));
    //    break;
    //  case EV_JOINED:
    //    Serial.println(F("EV_JOINED"));
    //    break;
    /*
    || This event is defined but not used in the code. No
    || point in wasting codespace on it.
    ||
    || case EV_RFU1:
    ||     Serial.println(F("EV_RFU1"));
    ||     break;
    */
    //  case EV_JOIN_FAILED:
    //    Serial.println(F("EV_JOIN_FAILED"));
    //    break;
    //  case EV_REJOIN_FAILED:
    //    Serial.println(F("EV_REJOIN_FAILED"));
    //    break;
  case EV_TXCOMPLETE:
    Serial.println(F("EV_TXCOMPLETE (includes waiting for RX windows)"));
    Serial.print("Seq num ");
    Serial.println(LMIC_getSeqnoUp());
    if (LMIC.txrxFlags & TXRX_ACK)
      Serial.println(F("Received ack"));
    if (LMIC.dataLen) {
      Serial.println(F("Received "));
      Serial.println(LMIC.dataLen);
      Serial.println(F(" bytes of payload"));
    }

    sleep_node();
    break;
  case EV_LOST_TSYNC:
    Serial.println(F("EV_LOST_TSYNC"));
    break;
  case EV_RESET:
    Serial.println(F("EV_RESET"));
    break;
  case EV_RXCOMPLETE:
    // data received in ping slot
    Serial.println(F("EV_RXCOMPLETE"));
    break;
  case EV_LINK_DEAD:
    Serial.println(F("EV_LINK_DEAD"));
    break;
  case EV_LINK_ALIVE:
    Serial.println(F("EV_LINK_ALIVE"));
    break;
  /*
  || This event is defined but not used in the code. No
  || point in wasting codespace on it.
  ||
  || case EV_SCAN_FOUND:
  ||    Serial.println(F("EV_SCAN_FOUND"));
  ||    break;
  */
  case EV_TXSTART:
    Serial.println(F("EV_TXSTART"));
    break;
  case EV_TXCANCELED:
    Serial.println(F("EV_TXCANCELED"));
    break;
  case EV_RXSTART:
    /* do not print anything -- it wrecks timing */
    break;
    //  case EV_JOIN_TXCOMPLETE:
    //    Serial.println(F("EV_JOIN_TXCOMPLETE: no JoinAccept"));
    //    break;
  default:
    Serial.print(F("Unknown event: "));
    Serial.println((unsigned)ev);
    break;
  }
}

void sleep_node() {
  LMIC_shutdown();
  save_frame_count(LMIC_getSeqnoUp());
  Serial.println("Shutting down");
  LowPower.shutdown(320000);
}

u4_t get_frame_count() {
  u4_t data;
  EEPROM.get(EEPROM_FRAME_COUT_ADDR, data);
  return data;
}
void save_frame_count(u4_t frame_count) {
  EEPROM.put(EEPROM_FRAME_COUT_ADDR, (u4_t)frame_count);
  EEPROM.commit();
}
