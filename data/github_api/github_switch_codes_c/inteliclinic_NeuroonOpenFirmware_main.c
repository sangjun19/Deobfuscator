/**
 * Copyright (c) 2014 - 2017, Nordic Semiconductor ASA
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form, except as embedded into a Nordic
 *    Semiconductor ASA integrated circuit in a product or a software update for
 *    such product, must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other
 *    materials provided with the distribution.
 *
 * 3. Neither the name of Nordic Semiconductor ASA nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * 4. This software, with or without modification, must only be used with a
 *    Nordic Semiconductor ASA integrated circuit.
 *
 * 5. Any software provided in binary form under this license must not be reverse
 *    engineered, decompiled, modified and/or disassembled.
 *
 * THIS SOFTWARE IS PROVIDED BY NORDIC SEMICONDUCTOR ASA "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NORDIC SEMICONDUCTOR ASA OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/** @file
 *
 * @defgroup ble_sdk_app_template_main main.c
 * @{
 * @ingroup ble_sdk_app_template
 * @brief Template project main file.
 *
 * This file contains a template for creating a new application. It has the code necessary to wakeup
 * from button, advertise, get a connection restart advertising on disconnect and if no new
 * connection created go back to system-off mode.
 * It can easily be used as a starting point for creating a new application, the comments identified
 * with 'YOUR_JOB' indicates where and how you can customize.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "nordic_common.h"
#include "nrf.h"
#include "app_error.h"
/*#include "boards.h"*/
#include "softdevice_handler.h"
#include "app_timer.h"
#include "FreeRTOS.h"
#include "task.h"
#include "timers.h"
#include "semphr.h"

/*#include "bsp.h"*/
/*#include "bsp_btn_ble.h"*/
#include "nrf_gpio.h"
#include "nrf_drv_clock.h"
#include "nrf_power.h"

#define NRF_LOG_MODULE_NAME "APP"
#include "nrf_log.h"
#include "nrf_log_ctrl.h"

#include "ic_bluetooth.h"
#include "ic_driver_uart.h"
#include "ic_driver_button.h"
#include "ic_command_task.h"

#include "ic_service_ads.h"

#include "ic_service_stream1.h"

#include "ic_config.h"
#include "ic_easy_ltc_driver.h"

#include "ic_driver_ltc.h"
#include "ic_driver_actuators.h"
#include "ic_service_ltc.h"

#include "ic_ble_service.h"

#include "ic_service_time.h"

#include "ic_service_bas.h"

#include "ic_driver_bq27742.h"
#include "ic_driver_wdt.h"

#define APP_TIMER_PRESCALER             0                                           /**< Value of the RTC1 PRESCALER register. */
#define APP_TIMER_OP_QUEUE_SIZE         4                                           /**< Size of timer operation queues. */

#define SEC_PARAM_BOND                  1                                           /**< Perform bonding. */
#define SEC_PARAM_MITM                  0                                           /**< Man In The Middle protection not required. */
#define SEC_PARAM_LESC                  0                                           /**< LE Secure Connections not enabled. */
#define SEC_PARAM_KEYPRESS              0                                           /**< Keypress notifications not enabled. */
#define SEC_PARAM_IO_CAPABILITIES       BLE_GAP_IO_CAPS_NONE                        /**< No I/O capabilities. */
#define SEC_PARAM_OOB                   0                                           /**< Out Of Band data not available. */
#define SEC_PARAM_MIN_KEY_SIZE          7                                           /**< Minimum encryption key size. */
#define SEC_PARAM_MAX_KEY_SIZE          16                                          /**< Maximum encryption key size. */

#define DEAD_BEEF                       0xDEADBEEF                                  /**< Value used as error code on stack dump, can be used to identify stack location on stack unwind. */

#define IC_NRF_RESETREAS_WDT_BIT (0x01<<1)
#define IC_NRF_RESETREAS_SREQ (0x01<<2)
#define IC_NRF_RESETREAS_EXTI_BIT (0x01<<16)
#define IC_NRF_RESETREAS_LPCOMP (0x01<<17)

enum shutdown_source_e{
  IC_PWR_DOWN_SRC,
  IC_USB_PLUG_SRC,
  IC_USB_UNPLUG_SRC
}m_shutdown_source;

typedef enum{
  IC_RESET_REAS_WATCHDOG,
  IC_RESET_REAS_SREQ,
  IC_RESET_REAS_BUTTON,
  IC_RESET_REAS_PWR
}e_reset_reason;

static TaskHandle_t m_init_task;
static TaskHandle_t m_cleanup_task;
static e_reset_reason m_reset_reason;

static inline e_reset_reason get_reset_reason(void){
  if(NRF_POWER->RESETREAS & IC_NRF_RESETREAS_WDT_BIT)
    return IC_RESET_REAS_WATCHDOG;
  else if(NRF_POWER->RESETREAS & IC_NRF_RESETREAS_EXTI_BIT)
    return IC_RESET_REAS_BUTTON;
  else if(NRF_POWER->RESETREAS & IC_NRF_RESETREAS_SREQ)
    return IC_RESET_REAS_SREQ;
  else
    return IC_RESET_REAS_PWR;
}

/**@brief Callback function for asserts in the SoftDevice.
 *
 * @details This function will be called in case of an assert in the SoftDevice.
 *
 * @warning This handler is an example only and does not fit a final product. You need to analyze
 *          how your product is supposed to react in case of Assert.
 * @warning On assert from the SoftDevice, the system can only recover on reset.
 *
 * @param[in] line_num   Line number of the failing ASSERT call.
 * @param[in] file_name  File name of the failing ASSERT call.
 */
void assert_nrf_callback(uint16_t line_num, const uint8_t * p_file_name)
{
    app_error_handler(DEAD_BEEF, line_num, p_file_name);
}

/**@brief Function for the Power manager.
 */
static void power_manage(void)
{
  /*__auto_type err_code = sd_app_evt_wait();*/
  /*
   *__SEV();
   *__WFE();
   */
  __WFE();
  /*APP_ERROR_CHECK(err_code);*/
}

void vApplicationIdleHook( void )
{
  NRF_LOG_FLUSH();
  ic_wdt_refresh();
  power_manage();
}

void app_error_fault_handler(uint32_t id, uint32_t pc, uint32_t info)
{
  __auto_type _err = (error_info_t *)info;

  NRF_LOG_ERROR("{%s}[%d]\r\n", (uint32_t)_err->p_file_name, _err->line_num);
  NRF_LOG_FINAL_FLUSH();
  // On assert, the system can only recover with a reset.
#ifndef DEBUG
  NVIC_SystemReset();
#else
  app_error_save_and_stop(id, pc, info);
#endif // DEBUG
}

static void power_down_all_systems(void){
  nrf_gpio_cfg_output(15);
  nrf_gpio_cfg_output(16);
  nrf_gpio_cfg_output(IC_LTC_POWER_PIN);
  nrf_gpio_pin_clear(15);
  nrf_gpio_pin_clear(16);
  nrf_gpio_pin_clear(IC_LTC_POWER_PIN);
  /*nrf_gpio_cfg_default(15);*/
  /*nrf_gpio_cfg_default(16);*/
  /*nrf_gpio_cfg_default(IC_LTC_POWER_PIN);*/
  nrf_gpio_cfg_default(IC_SPI_FLASH_SS_PIN);
  nrf_gpio_cfg_default(IC_UART_RX_PIN);
  nrf_gpio_cfg_default(IC_UART_TX_PIN);
  nrf_gpio_cfg_default(IC_SPI_AFE_RESET_PIN);
  nrf_gpio_cfg_default(IC_SPI_AFE_PDN_PIN);
}

static void power_up_all_systems(void){
  nrf_gpio_cfg_output(15);
  nrf_gpio_pin_set(15);
  nrf_gpio_cfg_output(16);
  nrf_gpio_pin_set(16);
  nrf_gpio_cfg_output(IC_LTC_POWER_PIN);
  nrf_gpio_pin_set(IC_LTC_POWER_PIN);
}

#define WELCOME_PERIOD pdMS_TO_TICKS(500)
#define PERIOD pdMS_TO_TICKS(2000)

static void on_connect(void){
  ic_actuator_set_ramp_func(IC_LEFT_BLUE_LED,   WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_BLUE_LED,  WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_RED_LED,    WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_RED_LED,   WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_GREEN_LED,  WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, 0, 0);
}

static void on_disconnect(void){
  ic_actuator_set_ramp_func(IC_LEFT_RED_LED, WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, 0, 0);
  ic_actuator_set_off_func(IC_VIBRATOR, 0, 0, 0);
  ic_actuator_set_off_func(IC_POWER_LEDS, 0, 0, 0);

  ic_actuator_set_ramp_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD>>1, 0, 3);
  ic_actuator_set_ramp_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD>>1, 0, 3);
  /*ic_actuator_set_triangle_func(IC_LEFT_BLUE_LED, PERIOD, 0, 5);*/
  /*ic_actuator_set_triangle_func(IC_RIGHT_BLUE_LED, PERIOD, 0, 5);*/
}

#ifndef CHARGE_BYPASS
static void on_charging(void){
  ic_actuator_set_ramp_func(IC_RIGHT_RED_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_GREEN_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_GREEN_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_BLUE_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_BLUE_LED, PERIOD, 0, 0);
  ic_actuator_set_off_func(IC_VIBRATOR, 0, 0, 0);
  ic_actuator_set_off_func(IC_POWER_LEDS, 0, 0, 0);

  ic_actuator_set_ramp_func(IC_LEFT_RED_LED, PERIOD, 0, 50);
  /*ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, PERIOD<<1, 0, 50);*/
}

static void on_charged(void){
  ic_actuator_set_ramp_func(IC_RIGHT_RED_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_BLUE_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_GREEN_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_LEFT_RED_LED, PERIOD, 0, 0);
  ic_actuator_set_ramp_func(IC_RIGHT_BLUE_LED, PERIOD, 0, 0);
  ic_actuator_set_off_func(IC_VIBRATOR, 0, 0, 0);
  ic_actuator_set_off_func(IC_POWER_LEDS, 0, 0, 0);

  ic_actuator_set_ramp_func(IC_LEFT_GREEN_LED, PERIOD, 0, 30);
}

static TimerHandle_t m_charging_timer_handle = NULL;

static void charging_timer_callback(TimerHandle_t xTimer){
  __auto_type _bat = ic_bq_getChargerState();
  if(_bat == BATT_CHARGED){
    __auto_type _timer_ret_val = pdFAIL;
    STOP_TIMER(xTimer, 0, _timer_ret_val);
    if(_timer_ret_val != pdPASS)
      NRF_LOG_ERROR("Charging timer still running");
    on_charged();
  }
}
#endif

static void welcome(void){
  ic_actuator_set_triangle_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
}

static void welcome_wdt(void){
  ic_actuator_set_triangle_func(IC_LEFT_RED_LED, PERIOD, PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, PERIOD, PERIOD+1, 63);
  vTaskDelay(pdMS_TO_TICKS(PERIOD>>2));
}

void showoff(void){
  ic_actuator_set_blink_func(IC_POWER_LEDS, WELCOME_PERIOD>>2, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_VIBRATOR, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));

  power_down_all_systems();

  NRF_POWER->SYSTEMOFF = 1;
}

static void showoff_light(void){
  ic_actuator_set_blink_func(IC_POWER_LEDS, WELCOME_PERIOD>>2, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
  ic_actuator_set_triangle_func(IC_VIBRATOR, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD));
}

void bye_bye(void){
  ic_actuator_set_off_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  ic_actuator_set_off_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  ic_actuator_set_off_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  ic_actuator_set_off_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  ic_actuator_set_off_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  ic_actuator_set_off_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);

  ic_actuator_set_triangle_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_LEFT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_GREEN_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));
  ic_actuator_set_triangle_func(IC_RIGHT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 63);
  vTaskDelay(pdMS_TO_TICKS(WELCOME_PERIOD>>2));

}

static void m_deep_sleep(void){
  NRF_LOG_INFO("{%s}\n", (uint32_t)__func__);
  m_shutdown_source = IC_PWR_DOWN_SRC;
  RESUME_TASK(m_cleanup_task);
}

#ifndef CHARGE_BYPASS
static void on_plug(){
  m_shutdown_source = IC_USB_PLUG_SRC;
  RESUME_TASK(m_cleanup_task);
}

static void on_unplug(){
  __auto_type _timer_ret_val = pdFAIL;
  m_shutdown_source = IC_USB_UNPLUG_SRC;
  STOP_TIMER(m_charging_timer_handle, 0, _timer_ret_val);
  UNUSED_VARIABLE(_timer_ret_val);
  RESUME_TASK(m_cleanup_task);
}
#endif

static void cleanup_task (void *arg){
  NRF_LOG_INFO("{%s}\n", (uint32_t)__func__);

  if(*(enum shutdown_source_e *)arg == IC_PWR_DOWN_SRC ||
      *(enum shutdown_source_e *)arg == IC_USB_PLUG_SRC){

    sd_power_reset_reason_clr(NRF_POWER->RESETREAS);

    ic_bluetooth_disable();
    ic_service_stream1_deinit();
    ic_ads_service_deinit();

    bye_bye();
  }else{
    ic_actuator_set_ramp_func(IC_LEFT_BLUE_LED, WELCOME_PERIOD, WELCOME_PERIOD, 0);
    ic_actuator_set_ramp_func(IC_LEFT_RED_LED, WELCOME_PERIOD, WELCOME_PERIOD, 0);
  }
  vTaskDelay(512);
  ic_ltc_service_deinit();

  power_down_all_systems();

  if(*(enum shutdown_source_e *)arg == IC_USB_PLUG_SRC)
    NVIC_SystemReset();
  else
    NRF_POWER->SYSTEMOFF = 1;
}

static void shutdown_cmd(u_BLECmdPayload _) { ic_bq_shutdown(); }
static void showoff_cmd(u_BLECmdPayload _) { showoff_light(); }
static void program_BQ_cmd(u_BLECmdPayload _) { ic_bq_flash_image(); }

static void init_task (void *arg){
  UNUSED_PARAMETER(arg);
  power_up_all_systems();
  ic_wdt_init();

  ic_neuroon_exti_init();
  ic_ltc_service_init();


#ifndef CHARGE_BYPASS
  if(!ic_button_pressed(IC_BUTTON_USB_CONNECT_PIN)){
    if(m_charging_timer_handle == NULL)
      m_charging_timer_handle = xTimerCreate(
          "CHARGING",
          IC_CHARGING_TICK_PERIOD,
          pdTRUE,
          (void *) 0,
          charging_timer_callback);
    NRF_LOG_INFO("----====USB CONNECTED====----\n");

    ic_btn_usb_unplug_handle_init(on_unplug);

    if(ic_bq_getChargerState() == BATT_CHARGED){
      on_charged();
    }else{
      on_charging();
      __auto_type _timer_ret_val = pdFAIL;
      START_TIMER(m_charging_timer_handle, 0, _timer_ret_val);
      if(_timer_ret_val != pdPASS)
        NRF_LOG_ERROR("Charging timer not started");
    }
    vTaskDelete(NULL);
    taskYIELD();
  }

  ic_btn_usb_plug_handle_init(on_plug);
  #endif

  if (m_reset_reason != IC_RESET_REAS_WATCHDOG &&
      m_reset_reason != IC_RESET_REAS_BUTTON){
    showoff_light();
    vTaskDelay(IC_BUTTON_LONG_PRESS_OFFSET);
  }

  if(m_reset_reason == IC_RESET_REAS_WATCHDOG)
    welcome_wdt();
  else if(m_reset_reason == IC_RESET_REAS_BUTTON){
    vTaskDelay(IC_BUTTON_LONG_PRESS_OFFSET);
    if(!ic_button_pressed(IC_BUTTON_PWR_BUTTON_PIN) && m_reset_reason == IC_RESET_REAS_BUTTON){
      power_down_all_systems();
      NRF_POWER->SYSTEMOFF = 1;
    }
    else
      welcome();
  }



  ic_btn_pwr_long_press_handle_init(m_deep_sleep);

  ic_ads_service_init();
  ic_service_stream1_init();
  ic_ble_module_init();
  sd_power_reset_reason_clr(NRF_POWER->RESETREAS);
  ic_service_timestamp_init();
  cmd_module_init();
  ic_init_battery_update();
  on_disconnect();
  cmd_task_connect_to_shutdown_cmd(shutdown_cmd);
  cmd_task_connect_to_test_cmd(showoff_cmd);
  cmd_task_connect_to_flashBQ_cmd(program_BQ_cmd);
  vTaskDelete(NULL);
  taskYIELD();
}

void main_on_ble_evt(ble_evt_t * p_ble_evt){
  switch(p_ble_evt->header.evt_id){
    case BLE_GAP_EVT_CONNECTED:
      on_connect();
      break;
    case BLE_GAP_EVT_DISCONNECTED:
      on_disconnect();
      break;
  }
}

void vApplicationStackOverflowHook(TaskHandle_t xTask, signed char *pcTaskName){
  NRF_LOG_INFO("Stack overflowed: %s\n\r", (uint32_t)pcTaskName);
}

/**@brief Function for application main entry.
 */
int main(void)
{
    // Initialize.
    __auto_type err_code = NRF_LOG_INIT(GET_TICK_COUNT);
    APP_ERROR_CHECK(err_code);
    power_down_all_systems();

    err_code = nrf_drv_clock_init();
    APP_ERROR_CHECK(err_code);


    if(pdPASS != xTaskCreate(init_task, "INIT", 384, NULL, 3, &m_init_task)){
      APP_ERROR_HANDLER(NRF_ERROR_NO_MEM);
    }

    if(pdPASS != xTaskCreate(cleanup_task, "INIT", 192, (void *)&m_shutdown_source, 4, &m_cleanup_task)){
      APP_ERROR_HANDLER(NRF_ERROR_NO_MEM);
    }
    vTaskSuspend(m_cleanup_task);

    NRF_LOG_INFO("Reset reason: %d\n", NRF_POWER->RESETREAS);
    m_reset_reason = get_reset_reason();
    NRF_LOG_INFO("GPREGRET: %d\n", NRF_POWER->GPREGRET);
    nrf_power_resetreas_clear(0xFFFFFFFF);

    /*SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;*/
    NRF_LOG_INFO("starting scheduler\n");
    vTaskStartScheduler();

    for (;;)
    {
      /*APP_ERROR_HANDLER(NRF_ERROR_FORBIDDEN);*/
      power_manage();
    }
}

/**
 * @}
 */
