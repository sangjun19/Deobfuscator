/************************************************************************************
                  ** C **
Module:
Author: Unic-Lab <https://unic-lab.ru/>
************************************************************************************/

#include "main_tasks.h"
#include "app_config.h"
#include "version.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "stm32f1xx_hal.h"
#include <stdbool.h>

#include "leds.h"
#include "reset.h"
#include "ext_volt.h"
#include "current_sense.h"
#include "panel_type.h"
#include "nearby_panel.h"
#include "actuators.h"
#include "buttons.h"
#include "flash_storage.h"

#ifdef ON_DEBUG_MESSAGE
	#include "debug.h"
	#include "string.h"
	#include <stdio.h>

	char debug_buf[256] = {0};
#endif

//===================================================================================

#define CHECK_EXT_VOLTAGE_PERIOD_MS				60000																	// период измерения внешнего напряжения
#define MOTOR_MOVING_INDICATION_PERIOD_MS		500																		// период индикации движущихся актуаторов
#define CURRENT_DETECT_DELAY_MS					500																		// задержка перед измерением тока после запуска процесса
#define BUTTONS_ANTI_BOUNCE_MS					100																		// задержка для устранения дребезга кнопок

//===================================================================================

typedef enum {
	READY = 0,
	RUNNING,
	WAITING
} ProcessState_t;

typedef enum {
	PULL_DOWN_POS = 0,
	PULL_UP_POS,
	MOVING_POS,
	NEUTRAL_POS
} MotorPosition_t;

typedef enum {
	NO_ERROR = 0,
	EXT_POWER_ERROR,
	UP_MOTOR_TIMEOUT_ERROR,
	DOWN_MOTOR_TIMEOUT_ERROR,
	SIDE_MOTOR_TIMEOUT_ERROR,
	UP_MOTOR_NULL_CURRENT_ERROR,
	DOWN_MOTOR_NULL_CURRENT_ERROR,
	SIDE_MOTOR_NULL_CURRENT_ERROR
} ErrorsType_t;

static ProcessState_t check_ext_voltage_loop(uint32_t meas_delay_ms);
static ErrorsType_t check_device_errors(void);
static void error_indication_loop(ErrorsType_t error_type);
static void motor_state_indication_loop(void);
static void check_nearby_panel_loop(void);
static void motor_control_loop(void);
static void current_sense_res_loop(void);
static void timer_control_loop(void);
static void buttons_control_loop(void);
static void storage_backup_loop(void);

//===================================================================================

typedef struct {
	bool flag_isAdcСonvReady;
	bool flag_isNeedRevPolMainMotorPwr;
	bool flag_isNeedApplyForceMotor;
	bool flag_isNeedRemoveForceMotor;
	bool flag_isNeedStopUpMotor;
	bool flag_isNeedStopDownMotor;
	bool flag_isNeedStopSideMotor;
	bool flag_isNeedStopAllMotor;
	bool flag_isMainPowerPresent;
	bool flag_isReservePowerPresent;
	bool flag_isUpButtonPressed;
	bool flag_isDownButtonPressed;
	bool flag_isStopButtonPressed;
} globalFlags_t;

typedef struct {
	bool error_ExtPowerError;
	bool error_UpMotorTimeoutError;
	bool error_DownMotorTimeoutError;
	bool error_SideMotorTimeoutError;
	bool error_UpMotorNullCurentError;
	bool error_DownMotorNullCurentError;
	bool error_SideMotorNullCurentError;
} globalErrors_t;

typedef struct {
	MotorPosition_t state_UpMotor;
	MotorPosition_t state_DownMotor;
	MotorPosition_t state_SideMotor;
	MotorPosition_t state_UpMotorInFlash;
	MotorPosition_t state_DownMotorInFlash;
	MotorPosition_t state_SideMotorInFlash;
} motorState_t;

typedef struct {
	PanelType_t gvar_panel_type;
	NearbyPanelState_t gvar_nearby_panel_state;
	NearbyPanelState_t gvar_nearby_panel_state_InFlash;
} globalVars_t;

//===================================================================================

// структура глобальных флагов
globalFlags_t globalFlags = {
	.flag_isAdcСonvReady = false,
	.flag_isNeedRevPolMainMotorPwr = false,
	.flag_isNeedApplyForceMotor = false,
	.flag_isNeedRemoveForceMotor = false,
	.flag_isNeedStopUpMotor = false,
	.flag_isNeedStopDownMotor = false,
	.flag_isNeedStopSideMotor = false,
	.flag_isNeedStopAllMotor = false,
	.flag_isMainPowerPresent = false,
	.flag_isReservePowerPresent = false,
	.flag_isUpButtonPressed = false,
	.flag_isDownButtonPressed = false,
	.flag_isStopButtonPressed = false
};

// структура глобальных ошибок
globalErrors_t globalErrors = {
	.error_ExtPowerError = false,
	.error_UpMotorTimeoutError = false,
	.error_DownMotorTimeoutError = false,
	.error_SideMotorTimeoutError = false,
	.error_UpMotorNullCurentError = false,
	.error_DownMotorNullCurentError = false,
	.error_SideMotorNullCurentError = false
};

// структура глобальных состояний моторов
motorState_t motorState = {
	.state_UpMotor = PULL_DOWN_POS,
	.state_DownMotor = PULL_DOWN_POS,
	.state_SideMotor = PULL_DOWN_POS,
	.state_UpMotorInFlash = PULL_DOWN_POS,
	.state_DownMotorInFlash = PULL_DOWN_POS,
	.state_SideMotorInFlash = PULL_DOWN_POS
};

// структура глобальных переменных
globalVars_t globalVars = {
	.gvar_panel_type = COMMON,																							// тип панели, управляемой ПО
	.gvar_nearby_panel_state = PANNEL_NO_CONNECT,																		// состояние соседней панели
 	.gvar_nearby_panel_state_InFlash = PANNEL_NO_CONNECT
};

//===================================================================================

// Заголовки задач
osThreadId superloop_TaskHandle;

// Заголовки мьютексов
osMutexId adcUse_MutexHandle;

// Заголовки таймеров
osTimerId upMotor_TimerHandle;
osTimerId downMotor_TimerHandle;
osTimerId sideMotor_TimerHandle;

//===================================================================================

// Прототипы задач
void taskFunc_superloop(void const* argument);

// Прототипы callback таймеров
void main_task_upMotor_timerCallback(void const * argument);
void main_task_downMotor_timerCallback(void const * argument);
void main_task_sideMotor_timerCallback(void const * argument);

//===================================================================================

// Инициализация задач
void main_tasks_initTasks(void)
{
	osThreadDef(t_superloop, taskFunc_superloop, osPriorityNormal, 0, 1024);
	superloop_TaskHandle = osThreadCreate(osThread(t_superloop), NULL);
}

// Инициализация mutex
void main_tasks_initOsMutex(void)
{
	osMutexDef(m_adcUse);																								// создание mutex для управления доступом к ADC
	adcUse_MutexHandle = osMutexCreate(osMutex(m_adcUse));
}

// Инициализация таймеров
void main_tasks_initOsTimer(void)
{
	osTimerDef(ti_upMotor, main_task_upMotor_timerCallback);															// создание таймера для использования при таймауте движения верхнего мотора
  	upMotor_TimerHandle = osTimerCreate(osTimer(ti_upMotor), osTimerOnce, NULL);

	osTimerDef(ti_downMotor, main_task_downMotor_timerCallback);														// создание таймера для использования при таймауте движения нижнего мотора
  	downMotor_TimerHandle = osTimerCreate(osTimer(ti_downMotor), osTimerOnce, NULL);

	osTimerDef(ti_sideMotor, main_task_sideMotor_timerCallback);														// создание таймера для использования при таймауте движения моторов боковой панели
  	sideMotor_TimerHandle = osTimerCreate(osTimer(ti_sideMotor), osTimerOnce, NULL);

}

//===================================================================================

// Задачи
void taskFunc_superloop(void const* argument)
{
	// Вспомогательная переменная
	bool flag_isConditionTrue = false;

	// Индикация появившегося питания
	leds_ledOn(GREEN);

	// Определение типа панели
	globalVars.gvar_panel_type = panel_type_get_type();

	//	Чтение состояний акутаторов и наличия соседней панели из хранилища
	if (true == flash_storage_restore((uint8_t*)&motorState.state_UpMotorInFlash, (uint8_t*)&motorState.state_DownMotorInFlash, (uint8_t*)&motorState.state_SideMotorInFlash, (uint8_t*)&globalVars.gvar_nearby_panel_state_InFlash)) {
		motorState.state_UpMotor = motorState.state_UpMotorInFlash;
		motorState.state_DownMotor = motorState.state_DownMotorInFlash;
		motorState.state_SideMotor = motorState.state_SideMotorInFlash;
		globalVars.gvar_nearby_panel_state = globalVars.gvar_nearby_panel_state_InFlash;
	}

	// Проверка внешних напряжений
	while (check_ext_voltage_loop(0) != READY) {};

	// Проверка условия необходимости автоматического выдвижения актуаторов
	// Если актуаторы до перезагрузки или пропажи питания не были выдвинуты, то планируем их выдвинуть автоматически
	if (COMMON == globalVars.gvar_panel_type) {
		if ((PULL_DOWN_POS == motorState.state_UpMotor) && (PULL_DOWN_POS == motorState.state_DownMotor)) {
			flag_isConditionTrue = true;
		}
	} else {
		if ((PULL_DOWN_POS == motorState.state_UpMotor) && (PULL_DOWN_POS == motorState.state_DownMotor) && (PULL_DOWN_POS == motorState.state_SideMotor)) {
			flag_isConditionTrue = true;
		}
	}

	if (true == flag_isConditionTrue) {
		if (false != globalFlags.flag_isMainPowerPresent) {
			globalFlags.flag_isNeedApplyForceMotor = true;
		}
	}

	// Вывод версии ПО
	#ifdef ON_DEBUG_MESSAGE
		snprintf(debug_buf, sizeof(debug_buf), "FW: %d.%d.%d.%d\r\n", PRODUCT_ID, PRODUCT_VERSION, PRODUCT_VARIANT, PRODUCT_HARD);
		HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
	#endif

	while(1)
	{
		// Цикл проверки внешних напряжений
		// Задержка перед измерением в 3 сек нужна, т.к. после отключения внешнего реле на reserve входе есть напряжение разряжаемых конденсаторов VND5050
		check_ext_voltage_loop(3000);

		// Проверка наличия ошибок
		ErrorsType_t error_type = check_device_errors();

		// Цикл индикации ошибки
		error_indication_loop(error_type);

		// Если питание в допустимых пределах - управляем моторами, кнопками
		if (true != globalErrors.error_ExtPowerError) {
			// Цикл обработки кнопок
			buttons_control_loop();

			// Цикл проверки подключенности соседней панели (актуально только для обычной панели)
			if (COMMON == globalVars.gvar_panel_type) {
				check_nearby_panel_loop();
			}

			// Цикл управление моторами
			motor_control_loop();

			// Цикл управление таймерами для таймаутов
			timer_control_loop();

			// Цикл обработки данных с current sense resistors
			current_sense_res_loop();

			// Цикл сохранения во Flash состояний моторов
			storage_backup_loop();

			// Цикл индикации состояния моторов
			motor_state_indication_loop();
		} else {
			globalFlags.flag_isNeedApplyForceMotor = false;
			globalFlags.flag_isNeedRemoveForceMotor = false;
			globalFlags.flag_isNeedStopAllMotor = false;
		}

		reset_iwdg_refresh();
	}

	// Should never come here, else reset MCU
	HAL_NVIC_SystemReset();
}

//===================================================================================

static void storage_backup_loop(void)
{
	// Если актуаторы не в движении сохраняем состояния во Flash при необходимости
	if ((MOVING_POS != motorState.state_UpMotor) && (MOVING_POS != motorState.state_DownMotor) && (MOVING_POS != motorState.state_SideMotor)) {
		if ((motorState.state_UpMotorInFlash != motorState.state_UpMotor) ||
			(motorState.state_DownMotorInFlash != motorState.state_DownMotor) ||
			(motorState.state_SideMotorInFlash != motorState.state_SideMotor) ||
			(globalVars.gvar_nearby_panel_state_InFlash != globalVars.gvar_nearby_panel_state)) {

				// При успешной записи во Flash менеям состояния глобальных переменных
				if (true == flash_storage_backup(motorState.state_UpMotor, motorState.state_DownMotor, motorState.state_SideMotor, globalVars.gvar_nearby_panel_state)) {
					motorState.state_UpMotorInFlash = motorState.state_UpMotor;
					motorState.state_DownMotorInFlash = motorState.state_DownMotor;
					motorState.state_SideMotorInFlash = motorState.state_SideMotor;
					globalVars.gvar_nearby_panel_state_InFlash = globalVars.gvar_nearby_panel_state;
				}

				#ifdef ON_DEBUG_MESSAGE
					snprintf(debug_buf, sizeof(debug_buf), "state_UpMotor: %d, state_DownMotor: %d, state_SideMotor: %d, nearby_panel_state: %d\r\n", motorState.state_UpMotor, motorState.state_DownMotor, motorState.state_SideMotor, globalVars.gvar_nearby_panel_state);
					HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
				#endif
		}
	}
}

static void buttons_control_loop(void)
{
	#define DEAFULT_STATE	0

	static uint32_t prev_time_ms = 0;
	static uint8_t loop_state = DEAFULT_STATE;

	uint32_t curr_time_ms = HAL_GetTick();

	switch (loop_state)
	{
		case 0:
		{
			if (true == globalFlags.flag_isStopButtonPressed) {
				loop_state = 1;
			} else if (true == globalFlags.flag_isDownButtonPressed) {
				loop_state = 2;
			} else if (true == globalFlags.flag_isUpButtonPressed) {
				loop_state = 3;
			} else {
				prev_time_ms = curr_time_ms;
			}
			break;
		}
		case 1:	//STOP
		{
			if (curr_time_ms - prev_time_ms >= BUTTONS_ANTI_BOUNCE_MS) {
				loop_state = 4;
				globalFlags.flag_isStopButtonPressed = false;
				if (PRESSED_BTN == buttons_getState(STOP_BTN)) {
					// Устанавливаем флаг необходимости остановки всех движущихся актуаторов
					globalFlags.flag_isNeedStopAllMotor = true;
					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "STOP BUTTON - PRESSED\r\n", strlen("STOP BUTTON - PRESSED\r\n"), 500);
					#endif
				}
			}
			break;
		}
		case 2:	// DOWN
		{
			if (curr_time_ms - prev_time_ms >= BUTTONS_ANTI_BOUNCE_MS) {
				loop_state = 4;
				globalFlags.flag_isDownButtonPressed = false;
				if (PRESSED_BTN == buttons_getState(DOWN_BTN)) {
					// Устанавливаем флаг необходимости опускания актуаторов
					globalFlags.flag_isNeedRemoveForceMotor = true;
					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "DOWN BUTTON - PRESSED\r\n", strlen("DOWN BUTTON - PRESSED\r\n"), 500);
					#endif
				}
			}
			break;
		}
		case 3:	// UP
		{
			if (curr_time_ms - prev_time_ms >= BUTTONS_ANTI_BOUNCE_MS) {
				loop_state = 4;
				globalFlags.flag_isUpButtonPressed = false;
				if (PRESSED_BTN == buttons_getState(UP_BTN)) {
					// Устанавливаем флаг необходимости выдвижения актуаторов
					globalFlags.flag_isNeedApplyForceMotor = true;
					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "UP BUTTON - PRESSED\r\n", strlen("UP BUTTON - PRESSED\r\n"), 500);
					#endif
				}
			}
			break;
		}
		case 4:
		{
			prev_time_ms = curr_time_ms;
			loop_state = DEAFULT_STATE;
			break;
		}
		default:
			break;
	}
}

static void timer_control_loop(void)
{
	static MotorPosition_t prev_state_UpMotor = PULL_DOWN_POS;
	static MotorPosition_t prev_state_DownMotor = PULL_DOWN_POS;
	static MotorPosition_t prev_state_SideMotor = PULL_DOWN_POS;

	// Если актуатор движется запускаем таймер таймаута
	if ((MOVING_POS == motorState.state_UpMotor) && (prev_state_UpMotor != motorState.state_UpMotor)) {
		prev_state_UpMotor = motorState.state_UpMotor;
		// Предварительно сбрасываем ошибку
		globalErrors.error_UpMotorTimeoutError = false;
		osTimerStart(upMotor_TimerHandle, MOTOR_TIMEOUT_MS);
	} else if ((MOVING_POS != motorState.state_UpMotor) && (prev_state_UpMotor != motorState.state_UpMotor)) {
		prev_state_UpMotor = motorState.state_UpMotor;
		// Останавливаем таймер
		osTimerStop(upMotor_TimerHandle);
	}

	// Если актуатор движется запускаем таймер таймаута
	if ((MOVING_POS == motorState.state_DownMotor) && (prev_state_DownMotor != motorState.state_DownMotor)) {
		prev_state_DownMotor = motorState.state_DownMotor;
		// Предварительно сбрасываем ошибку
		globalErrors.error_DownMotorTimeoutError = false;
		osTimerStart(downMotor_TimerHandle, MOTOR_TIMEOUT_MS);
	} else if ((MOVING_POS != motorState.state_DownMotor) && (prev_state_DownMotor != motorState.state_DownMotor)) {
		prev_state_DownMotor = motorState.state_DownMotor;
		// Останавливаем таймер
		osTimerStop(downMotor_TimerHandle);
	}

	// Если актуатор движется запускаем таймер таймаута
	if ((MOVING_POS == motorState.state_SideMotor) && (prev_state_SideMotor != motorState.state_SideMotor)) {
		prev_state_SideMotor = motorState.state_SideMotor;
		// Предварительно сбрасываем ошибку
		globalErrors.error_SideMotorTimeoutError = false;
		osTimerStart(sideMotor_TimerHandle, MOTOR_TIMEOUT_MS);
	} else if ((MOVING_POS != motorState.state_SideMotor) && (prev_state_SideMotor != motorState.state_SideMotor)) {
		prev_state_SideMotor = motorState.state_SideMotor;
		// Останавливаем таймер
		osTimerStop(sideMotor_TimerHandle);
	}
}

static void current_sense_res_loop(void)
{
	static uint32_t prev_time_upMotor_ms = 0;
	static uint32_t prev_time_downMotor_ms = 0;
	static uint32_t prev_time_sideMotor_ms = 0;

	static MotorPosition_t prev_state_UpMotor = PULL_DOWN_POS;
	static MotorPosition_t prev_state_DownMotor = PULL_DOWN_POS;
	static MotorPosition_t prev_state_SideMotor = PULL_DOWN_POS;

	uint32_t curr_time_ms = HAL_GetTick();

	// Если актуатор движется запускаем задержку перед сэмплированием
	if ((MOVING_POS == motorState.state_UpMotor) && (prev_state_UpMotor != motorState.state_UpMotor)) {
		prev_state_UpMotor = motorState.state_UpMotor;
		prev_time_upMotor_ms = curr_time_ms;
		globalErrors.error_UpMotorNullCurentError = false;
	} else if ((MOVING_POS != motorState.state_UpMotor) && (prev_state_UpMotor != motorState.state_UpMotor)) {
		prev_state_UpMotor = motorState.state_UpMotor;
	}

	if ((MOVING_POS == motorState.state_DownMotor) && (prev_state_DownMotor != motorState.state_DownMotor)) {
		prev_state_DownMotor = motorState.state_DownMotor;
		prev_time_downMotor_ms = curr_time_ms;
		globalErrors.error_DownMotorNullCurentError = false;
	} else if ((MOVING_POS != motorState.state_DownMotor) && (prev_state_DownMotor != motorState.state_DownMotor)) {
		prev_state_DownMotor = motorState.state_DownMotor;
	}

	if ((MOVING_POS == motorState.state_SideMotor) && (prev_state_SideMotor != motorState.state_SideMotor)) {
		prev_state_SideMotor = motorState.state_SideMotor;
		prev_time_sideMotor_ms = curr_time_ms;
		globalErrors.error_SideMotorNullCurentError = false;
	} else if ((MOVING_POS != motorState.state_SideMotor) && (prev_state_SideMotor != motorState.state_SideMotor)) {
		prev_state_SideMotor = motorState.state_SideMotor;
	}

	// Если актуаторы в движении обрабатываем данные с current sense resistors
	if ((MOVING_POS == motorState.state_UpMotor) || (MOVING_POS == motorState.state_DownMotor) || (MOVING_POS == motorState.state_SideMotor)) {
		// Сработало прерывание ADC
		if (true == globalFlags.flag_isAdcСonvReady) {
			globalFlags.flag_isAdcСonvReady = false;

			// Запускаем обработку данных
			current_sense_process_adc_data();

			// Верхний актуатор
			if (MOVING_POS == motorState.state_UpMotor) {
				if (curr_time_ms - prev_time_upMotor_ms >= CURRENT_DETECT_DELAY_MS) {
					float up_current_amp = current_sense_get_up_current();

				#ifdef ON_DEBUG_MESSAGE
					snprintf(debug_buf, sizeof(debug_buf), "Motor UP current: %.2f\r\n", up_current_amp);
					HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
				#endif

					if ((up_current_amp >= UP_MOTOR_CURR_THRESH_A) || (0 == up_current_amp)) {
						globalFlags.flag_isNeedStopUpMotor = true;
						if (0 == up_current_amp) {
							// Устанавливаем флаг ошибки
							globalErrors.error_UpMotorNullCurentError = true;
						}
					}
				}
			}

			// Нижний актуатор
			if (MOVING_POS == motorState.state_DownMotor) {
				if (curr_time_ms - prev_time_downMotor_ms >= CURRENT_DETECT_DELAY_MS) {
					float down_current_amp = current_sense_get_down_current();

				#ifdef ON_DEBUG_MESSAGE
					snprintf(debug_buf, sizeof(debug_buf), "Motor Down current: %.2f\r\n", down_current_amp);
					HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
				#endif

					if ((down_current_amp >= DOWN_MOTOR_CURR_THRESH_A) || (0 == down_current_amp)) {
						globalFlags.flag_isNeedStopDownMotor = true;
						if (0 == down_current_amp) {
							// Устанавливаем флаг ошибки
							globalErrors.error_DownMotorNullCurentError = true;
						}
					}
				}
			}

			// Боковые актуаторы если панель не общая
			if (COMMON != globalVars.gvar_panel_type) {
				if (MOVING_POS == motorState.state_SideMotor) {
					if (curr_time_ms - prev_time_sideMotor_ms >= (4 * CURRENT_DETECT_DELAY_MS)) {						// задержка больше чем у одиночного актуатора, т.к. движение не равномерное всех боковых
						float side_current_amp = current_sense_get_side_current();

					#ifdef ON_DEBUG_MESSAGE
						snprintf(debug_buf, sizeof(debug_buf), "Motor Side current: %.2f\r\n", side_current_amp);
						HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
					#endif

						// Определение порога по току в зависимости от кол-ва боковых актуаторов
						float current_thr_a = globalVars.gvar_panel_type * (2 * SIDE_MOTOR_CURR_THRESH_A);
						if ((side_current_amp >= current_thr_a) || (0 == side_current_amp)) {
							globalFlags.flag_isNeedStopSideMotor = true;
							if (0 == side_current_amp) {
								// Устанавливаем флаг ошибки
								globalErrors.error_SideMotorNullCurentError = true;
							}
						}
					}
				}
			}
		}
	}
}

static void motor_control_loop(void)
{
	#define DEAFULT_STATE	0

	// Вспомогательная переменная
	bool flag_isConditionTrue = false;
	static uint8_t loop_state = DEAFULT_STATE;
	static uint32_t prev_time_ms = 0;
	static uint32_t stage_prev_time_ms = 0;
	static bool flag_isDownMotorWasStarted = false;

	uint32_t curr_time_ms = HAL_GetTick();

	// На случай зависания в одной из стадий
	if ((curr_time_ms - stage_prev_time_ms) > (3 * MOTOR_TIMEOUT_MS)) {
		if (DEAFULT_STATE != loop_state) {
			HAL_NVIC_SystemReset();
		}
	}

	// Необходимость выдвинуть актуаторы
	if (true == globalFlags.flag_isNeedApplyForceMotor) {
		globalFlags.flag_isNeedApplyForceMotor = false;
		// Если актуаторы не в движении, запускаем процесс
		if ((MOVING_POS != motorState.state_UpMotor) && (MOVING_POS != motorState.state_DownMotor) && (MOVING_POS != motorState.state_SideMotor) && DEAFULT_STATE == loop_state) {
			if (COMMON == globalVars.gvar_panel_type) {
				if ((PULL_UP_POS != motorState.state_UpMotor) || (PULL_UP_POS != motorState.state_DownMotor)) {
					flag_isConditionTrue = true;
				}
			} else {
				if ((PULL_UP_POS != motorState.state_UpMotor) || (PULL_UP_POS != motorState.state_DownMotor) || (PULL_UP_POS != motorState.state_SideMotor)) {
					flag_isConditionTrue = true;
				}
			}

			if (true == flag_isConditionTrue) {
				loop_state = 10;
				stage_prev_time_ms = curr_time_ms;
			}
		}

	// Необходимость задвинуть актуаторы
	} else if (true == globalFlags.flag_isNeedRemoveForceMotor) {
		globalFlags.flag_isNeedRemoveForceMotor = false;
		// Если актуатор не в движении и не внизу, запускаем процесс
		if ((MOVING_POS != motorState.state_UpMotor) && (MOVING_POS != motorState.state_DownMotor) && (MOVING_POS != motorState.state_SideMotor) && DEAFULT_STATE == loop_state) {
			if (COMMON == globalVars.gvar_panel_type) {
				if ((PULL_DOWN_POS != motorState.state_UpMotor) || (PULL_DOWN_POS != motorState.state_DownMotor)) {
					flag_isConditionTrue = true;
				}
			} else {
				if ((PULL_DOWN_POS != motorState.state_UpMotor) || (PULL_DOWN_POS != motorState.state_DownMotor) || (PULL_DOWN_POS != motorState.state_SideMotor)) {
					flag_isConditionTrue = true;
				}
			}

			if (true == flag_isConditionTrue) {
				loop_state = 20;
				stage_prev_time_ms = curr_time_ms;
			}
		}

	// Необходимость немедленно остановить все актуаторы
	} else if (true == globalFlags.flag_isNeedStopAllMotor) {
		globalFlags.flag_isNeedStopAllMotor = false;
		// Если актуаторы в движении, запускаем процесс
		if ((MOVING_POS == motorState.state_UpMotor) || (MOVING_POS == motorState.state_DownMotor) || (MOVING_POS == motorState.state_SideMotor)) {
			loop_state = 30;
			stage_prev_time_ms = curr_time_ms;
		}
	}

	switch (loop_state)
	{
		case 10:																										// case 1X - процесс ВЫДВИЖЕНИЯ актуаторов
		{
			// Защищаем доступ к ADC с помощью mutex
			osStatus ret_stat = osMutexWait(adcUse_MutexHandle, 0);

			// Если доступ получен - выполняем цикл
			if (ret_stat == osOK) {
				// Коммутируем актуаторы в направлении для движения "Вверх"
				actuators_prepare_move(UP_DIR, globalFlags.flag_isNeedRevPolMainMotorPwr);
				// Запускаем сэмплирование каналов current sense resistors
				current_sense_start_measure();

				if (COMMON != globalVars.gvar_panel_type) {
					// Меняем состояние мотора SIDE
					motorState.state_SideMotor = MOVING_POS;
					// Подаем питание на актуатор SIDE
					actuators_power_on(SIDE_ACTUATOR);

					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "Move UP Motor_SIDE\r\n", strlen("Move UP Motor_SIDE\r\n"), 500);
					#endif

					// Меняем стадию
					loop_state = 11;
				} else {
					// Меняем стадию
					loop_state = 12;
				}
			}
			break;
		}

		case 11:																										// case 1X - процесс ВЫДВИЖЕНИЯ актуаторов
		{
			// Нужно остановить мотор SIDE
			if (true == globalFlags.flag_isNeedStopSideMotor) {
				globalFlags.flag_isNeedStopSideMotor = false;
				// Останавливаем таймер
				osTimerStop(sideMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_SideMotor = PULL_UP_POS;
				// Снимаем питание с актуатора SIDE
				actuators_power_off(SIDE_ACTUATOR);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_SIDE\r\n", strlen("Stop Motor_SIDE\r\n"), 500);
				#endif

				// Меняем стадию
				loop_state = 12;
			}
			break;
		}

		case 12:																										// case 1X - процесс ВЫДВИЖЕНИЯ актуаторов
		{
			// Меняем состояние мотора UP
			motorState.state_UpMotor = MOVING_POS;
			// Подаем питание на актуатор UP
			actuators_power_on(UP_ACTUATOR);
			// Меняем стадию
			loop_state = 13;
			// Сбрасываем пред. время для след. стадии
			prev_time_ms = curr_time_ms;

			#ifdef ON_DEBUG_MESSAGE
				HAL_UART_Transmit(&huart1, "Move UP Motor_UP\r\n", strlen("Move UP Motor_UP\r\n"), 500);
			#endif

			break;
		}

		case 13:																										// case 1X - процесс ВЫДВИЖЕНИЯ актуаторов
		{
			// С задержкой в MOTOR_DOWN_MOV_DELAY_MS подаем питание на нижний актуатор
			if (MOVING_POS != motorState.state_DownMotor && PULL_UP_POS != motorState.state_DownMotor) {
				uint32_t curr_time_ms = HAL_GetTick();
				if (curr_time_ms - prev_time_ms > MOTOR_DOWN_MOV_DELAY_MS) {
					// Устанавливаем флаг
					flag_isDownMotorWasStarted = true;
					// Меняем состояние мотора DOWN
					motorState.state_DownMotor = MOVING_POS;
					// Подаем питание на актуатор DOWN
					actuators_power_on(DOWN_ACTUATOR);

					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "Move UP Motor_DOWN\r\n", strlen("Move UP Motor_DOWN\r\n"), 500);
					#endif
				}
			}

			// Нужно остановить мотор UP
			if (true == globalFlags.flag_isNeedStopUpMotor) {
				globalFlags.flag_isNeedStopUpMotor = false;
				// Останавливаем таймер
				osTimerStop(upMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_UpMotor = PULL_UP_POS;
				// Снимаем питание с актуатора UP
				actuators_power_off(UP_ACTUATOR);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_UP\r\n", strlen("Stop Motor_UP\r\n"), 500);
				#endif
			}

			// Нужно остановить мотор DOWN
			if (true == globalFlags.flag_isNeedStopDownMotor) {
				globalFlags.flag_isNeedStopDownMotor = false;
				// Останавливаем таймер
				osTimerStop(downMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_DownMotor = PULL_UP_POS;
				// Снимаем питание с актуатора DOWN
				actuators_power_off(DOWN_ACTUATOR);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_DOWN\r\n", strlen("Stop Motor_DOWN\r\n"), 500);
				#endif
			}

			// Если актуаторы не в движении и нижний актуатор уже запускался, запускаем процесс видвижения боковой панели или отключаем общее питание
			if ((MOVING_POS != motorState.state_UpMotor) && (MOVING_POS != motorState.state_DownMotor) && (true == flag_isDownMotorWasStarted)) {
				// Сбрасываем флаг
				flag_isDownMotorWasStarted = false;
				// Меняем стадию
				loop_state = 40;
			}
			break;
		}

		case 20:																										// case 2X - процесс ЗАДВИЖЕНИЯ актуаторов
		{
			// Защищаем доступ к ADC с помощью mutex
			osStatus ret_stat = osMutexWait(adcUse_MutexHandle, 0);

			// Если доступ получен - выполняем цикл
			if (ret_stat == osOK) {
				// Коммутируем актуаторы в направлении для движения "Вниз"
				actuators_prepare_move(DOWN_DIR, globalFlags.flag_isNeedRevPolMainMotorPwr);
				// Запускаем сэмплирование каналов current sense resistors
				current_sense_start_measure();

				// Меняем состояние мотора UP
				motorState.state_UpMotor = MOVING_POS;
				// Подаем питание на актуатор UP
				actuators_power_on(UP_ACTUATOR);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Move DOWN Motor_UP\r\n", strlen("Move DOWN Motor_UP\r\n"), 500);
				#endif

				// Меняем состояние мотора DOWN
				motorState.state_DownMotor = MOVING_POS;
				// Подаем питание на актуатор DOWN
				actuators_power_on(DOWN_ACTUATOR);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Move DOWN Motor_DOWN\r\n", strlen("Move DOWN Motor_DOWN\r\n"), 500);
				#endif

				// Меняем стадию
				loop_state = 21;
			}
			break;
		}

		case 21:																										// case 2X - процесс ЗАДВИЖЕНИЯ актуаторов
		{
			// Нужно остановить мотор UP
			if (true == globalFlags.flag_isNeedStopUpMotor) {
				globalFlags.flag_isNeedStopUpMotor = false;
				// Останавливаем таймер
				osTimerStop(upMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_UpMotor = PULL_DOWN_POS;
				// Снимаем питание с актуатора UP
				actuators_power_off(UP_ACTUATOR);

				// Т.к. остановка будет по концевику - сбрасываем ошибку
				globalErrors.error_UpMotorNullCurentError = false;

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_UP\r\n", strlen("Stop Motor_UP\r\n"), 500);
				#endif
			}

			// Нужно остановить мотор DOWN
			if (true == globalFlags.flag_isNeedStopDownMotor) {
				globalFlags.flag_isNeedStopDownMotor = false;
				// Останавливаем таймер
				osTimerStop(downMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_DownMotor = PULL_DOWN_POS;
				// Снимаем питание с актуатора DOWN
				actuators_power_off(DOWN_ACTUATOR);

				// Т.к. остановка будет по концевику - сбрасываем ошибку
				globalErrors.error_DownMotorNullCurentError = false;

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_DOWN\r\n", strlen("Stop Motor_DOWN\r\n"), 500);
				#endif
			}

			// Если актуаторы не в движении - задвигаем боковую панель или отключаем общее питание
			if ((MOVING_POS != motorState.state_UpMotor) && (MOVING_POS != motorState.state_DownMotor)) {
				if (COMMON != globalVars.gvar_panel_type) {
					// Меняем состояние мотора SIDE
					motorState.state_SideMotor = MOVING_POS;
					// Подаем питание на актуатор SIDE
					actuators_power_on(SIDE_ACTUATOR);

					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "Move DOWN Motor_SIDE\r\n", strlen("Move DOWN Motor_SIDE\r\n"), 500);
					#endif

					// Меняем стадию
					loop_state = 22;
				} else {
					// Меняем стадию
					loop_state = 40;
				}
			}
			break;
		}

		case 22:																										// case 2X - процесс ЗАДВИЖЕНИЯ актуаторов
		{
			// Нужно остановить мотор SIDE
			if (true == globalFlags.flag_isNeedStopSideMotor) {
				globalFlags.flag_isNeedStopSideMotor = false;
				// Останавливаем таймер
				osTimerStop(sideMotor_TimerHandle);
				// Меняем состояние мотора
				motorState.state_SideMotor = PULL_DOWN_POS;
				// Снимаем питание с актуатора SIDE
				actuators_power_off(SIDE_ACTUATOR);

				// Т.к. остановка будет по концевику - сбрасываем ошибку
				globalErrors.error_SideMotorNullCurentError = false;

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Stop Motor_SIDE\r\n", strlen("Stop Motor_SIDE\r\n"), 500);
				#endif

				// Меняем стадию
				loop_state = 40;
			}
			break;
		}

		case 30:																										// case 3X - процесс ОСТАНОВКИ ВСЕХ актуаторов
			// Останавливаем таймер
			osTimerStop(upMotor_TimerHandle);
			osTimerStop(downMotor_TimerHandle);
			osTimerStop(sideMotor_TimerHandle);

			// Снимаем питание с актуаторов
			actuators_power_off(UP_ACTUATOR);
			actuators_power_off(DOWN_ACTUATOR);

			// Устанавливаем в нейтральное положение, чтобы можно было после СТОП в любую сторону ехать
			motorState.state_UpMotor = NEUTRAL_POS;
			motorState.state_DownMotor = NEUTRAL_POS;

			if (COMMON != globalVars.gvar_panel_type) {
				actuators_power_off(SIDE_ACTUATOR);
				motorState.state_SideMotor = NEUTRAL_POS;
			}

			flag_isDownMotorWasStarted = false;

			// Меняем стадию
			loop_state = 40;
			break;

		case 40:
			// Останавливаем сэмплирование каналов current sense resistors
			current_sense_stop_measure();
			// Сбрасываем флаг
			globalFlags.flag_isAdcСonvReady = false;
			// Выключаем общее питание
			actuators_main_power_off(globalFlags.flag_isNeedRevPolMainMotorPwr);
			// Освобождаем доступ к ADC, возвращая mutex
			osMutexRelease(adcUse_MutexHandle);
			// Меняем стадию
			loop_state = DEAFULT_STATE;

			#ifdef ON_DEBUG_MESSAGE
				HAL_UART_Transmit(&huart1, "MUTEX-Release [motor_control]\r\n", strlen("MUTEX-Release [motor_control]\r\n"), 500);
			#endif
			break;

		default:
			break;
	}
}

static void check_nearby_panel_loop(void)
{
	#define PANEL_STATE_REPETITIONS_CNT			8
	#define DISCONN_NEARBY_PANEL_PERIOD_MS		(MOTOR_TIMEOUT_MS / PANEL_STATE_REPETITIONS_CNT)
	#define CONN_NEARBY_PANEL_PERIOD_MS			125

	// Начальное значение 1 для того, чтобы новое состояние обновилось не сразу, а в течении PANEL_STATE_REPETITIONS_CNT
	static uint8_t panel_state_bits = 1;
	static uint32_t prev_time_ms = 0U - CONN_NEARBY_PANEL_PERIOD_MS;
	uint32_t check_period_ms;

	// Выбираем период сэмплирования в зависимости от того, подключена соседняя панель или нет
	if (PANNEL_CONNECT == globalVars.gvar_nearby_panel_state) {
		check_period_ms = CONN_NEARBY_PANEL_PERIOD_MS;
	} else {
		check_period_ms = DISCONN_NEARBY_PANEL_PERIOD_MS;
	}

	uint32_t curr_time_ms = HAL_GetTick();

	if (curr_time_ms - prev_time_ms > check_period_ms) {
		prev_time_ms = curr_time_ms;

		// Получаем состояние соседней панели
		NearbyPanelState_t curr_panel_state = nearby_panel_get_state();

		// Меняем значение переменной устойчивых состояний
		panel_state_bits = (panel_state_bits << 1) | curr_panel_state;

		// Если в течении PANEL_STATE_REPETITIONS_CNT уровень устойчив - меняем глобальную переменную состояния соседней панели
		if (panel_state_bits == (uint8_t)(~PANNEL_CONNECT + 1)) {
			if (PANNEL_CONNECT != globalVars.gvar_nearby_panel_state) {
				globalVars.gvar_nearby_panel_state = PANNEL_CONNECT;
				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Nearby Panel - CONNECT\r\n", strlen("Nearby Panel - CONNECT\r\n"), 500);
				#endif
			}
		} else if (panel_state_bits == (uint8_t)(~PANNEL_NO_CONNECT + 1)) {
			if (PANNEL_NO_CONNECT != globalVars.gvar_nearby_panel_state) {
				globalVars.gvar_nearby_panel_state = PANNEL_NO_CONNECT;

				if (false != globalFlags.flag_isMainPowerPresent) {
					// Устанавливаем необходимость задвинуть актуаторы, т.к. соседняя панель отсоединилась
					globalFlags.flag_isNeedRemoveForceMotor = true;
				}

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "Nearby Panel - DISCONNECT\r\n", strlen("Nearby Panel - DISCONNECT\r\n"), 500);
				#endif
			}
		}
	}
}

static ErrorsType_t check_device_errors(void)
{
	ErrorsType_t error_type;

	#ifdef ON_DEBUG_MESSAGE
	static uint32_t prev_time_ms = 0;
	uint32_t curr_time_ms = HAL_GetTick();
	#endif

	// Проверка условий в порядке приоритета от High к Low
	if (true == globalErrors.error_ExtPowerError) {
		error_type = EXT_POWER_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "EXT_POWER_ERROR\r\n", strlen("EXT_POWER_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_UpMotorTimeoutError) {
		error_type = UP_MOTOR_TIMEOUT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "UP_MOTOR_TIMEOUT_ERROR\r\n", strlen("UP_MOTOR_TIMEOUT_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_DownMotorTimeoutError) {
		error_type = DOWN_MOTOR_TIMEOUT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "DOWN_MOTOR_TIMEOUT_ERROR\r\n", strlen("DOWN_MOTOR_TIMEOUT_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_SideMotorTimeoutError) {
		error_type = SIDE_MOTOR_TIMEOUT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "SIDE_MOTOR_TIMEOUT_ERROR\r\n", strlen("SIDE_MOTOR_TIMEOUT_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_UpMotorNullCurentError) {
		error_type = UP_MOTOR_NULL_CURRENT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "UP_MOTOR_NULL_CURRENT_ERROR\r\n", strlen("UP_MOTOR_NULL_CURRENT_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_DownMotorNullCurentError) {
		error_type = DOWN_MOTOR_NULL_CURRENT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "DOWN_MOTOR_NULL_CURRENT_ERROR\r\n", strlen("DOWN_MOTOR_NULL_CURRENT_ERROR\r\n"), 500);
			}
		#endif
	} else if (true == globalErrors.error_SideMotorNullCurentError) {
		error_type = SIDE_MOTOR_NULL_CURRENT_ERROR;
		#ifdef ON_DEBUG_MESSAGE
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				HAL_UART_Transmit(&huart1, "SIDE_MOTOR_NULL_CURRENT_ERROR\r\n", strlen("SIDE_MOTOR_NULL_CURRENT_ERROR\r\n"), 500);
			}
		#endif
	} else {
		error_type = NO_ERROR;
	}

	return error_type;
}

static void motor_state_indication_loop(void)
{
	MotorPosition_t motor_pos = PULL_DOWN_POS;
	static uint32_t prev_time_ms = 0;

	uint32_t curr_time_ms = HAL_GetTick();

	if ((MOVING_POS == motorState.state_UpMotor) || (MOVING_POS == motorState.state_DownMotor) || (MOVING_POS == motorState.state_SideMotor)) {
		motor_pos = MOVING_POS;
	}

	switch (motor_pos)
	{
		case PULL_DOWN_POS:
		case PULL_UP_POS:
			leds_ledOn(GREEN);
			break;
		case MOVING_POS:
			if (curr_time_ms - prev_time_ms > MOTOR_MOVING_INDICATION_PERIOD_MS) {
				prev_time_ms = curr_time_ms;
				leds_ledToggle(GREEN);
			}
			break;
		default:
			break;
	}
}

static void error_indication_loop(ErrorsType_t error_type)
{
	static ErrorsType_t prev_error_type = NO_ERROR;
	static uint32_t prev_time_ms = 0;

	uint32_t curr_time_ms = HAL_GetTick();

	if (prev_error_type != error_type) {
		prev_error_type = error_type;
		prev_time_ms = curr_time_ms;
	}

	switch (error_type)
	{
		case EXT_POWER_ERROR:
			leds_ledOn(RED);
			break;
		case UP_MOTOR_TIMEOUT_ERROR:
		case UP_MOTOR_NULL_CURRENT_ERROR:
			if (curr_time_ms - prev_time_ms > 500) {
				prev_time_ms = curr_time_ms;
				leds_ledToggle(RED);
			}
			break;
		case DOWN_MOTOR_TIMEOUT_ERROR:
		case DOWN_MOTOR_NULL_CURRENT_ERROR:
			if (curr_time_ms - prev_time_ms > 1000) {
				prev_time_ms = curr_time_ms;
				leds_ledToggle(RED);
			}
			break;
		case SIDE_MOTOR_TIMEOUT_ERROR:
		case SIDE_MOTOR_NULL_CURRENT_ERROR:
			if (curr_time_ms - prev_time_ms > 250) {
				prev_time_ms = curr_time_ms;
				leds_ledToggle(RED);
			}
			break;
		case NO_ERROR:
			leds_ledOff(RED);
			break;
		default:
			break;
	}
}

static ProcessState_t check_ext_voltage_loop(uint32_t meas_delay_ms)
{
	static ProcessState_t loop_state = READY;
	static uint32_t prev_time_ms = 0U - CHECK_EXT_VOLTAGE_PERIOD_MS;

	uint32_t curr_time_ms = HAL_GetTick();

	switch (loop_state)
	{
		case READY:
			if (curr_time_ms - prev_time_ms > CHECK_EXT_VOLTAGE_PERIOD_MS) {
				prev_time_ms = curr_time_ms;

				// Защищаем доступ к ADC с помощью mutex
				osStatus ret_stat = osMutexWait(adcUse_MutexHandle, 0);

				// Если доступ получен - запускаем стадию задержки перед измерением
				if (ret_stat == osOK) {
					loop_state = WAITING;
					#ifdef ON_DEBUG_MESSAGE
						HAL_UART_Transmit(&huart1, "MUTEX-Get [voltage_loop]\r\n", strlen("MUTEX-Get [voltage_loop]\r\n"), 500);
					#endif
				}
			}
			break;
		case WAITING:
			if (curr_time_ms - prev_time_ms >= meas_delay_ms) {
				loop_state = RUNNING;
				// Запускаем измерение
				ext_volt_start_measure();
			}
			break;
		case RUNNING:
			if (true == globalFlags.flag_isAdcСonvReady) {
				// Останавливаем измерение
				ext_volt_stop_measure();
				// Сбрасываем флаг
				globalFlags.flag_isAdcСonvReady = false;
				// Запускаем обработку данных
				ext_volt_process_adc_data();

				float main_volt = ext_volt_get_main_volt();
				float reserve_volt = ext_volt_get_reserve_volt();

				#ifdef ON_DEBUG_MESSAGE
					snprintf(debug_buf, sizeof(debug_buf), "Voltage: main = %.2f, reverse = %.2f\r\n", main_volt, reserve_volt);
					HAL_UART_Transmit(&huart1, (uint8_t*)debug_buf, strlen(debug_buf), 500);
				#endif

				// Предварительно сбрасываем ошибку внешнего питания
				globalErrors.error_ExtPowerError = false;

				if (main_volt != 0) {
					// Устанавливаем флаг наличия внешнего основного питания
					globalFlags.flag_isMainPowerPresent = true;
					// Реверс полярности управления главным питанием моторов НЕ НУЖЕН
					globalFlags.flag_isNeedRevPolMainMotorPwr = false;

					if ( (main_volt < MIN_EXT_VOLTAGE_V) || (main_volt > MAX_EXT_VOLTAGE_V) ) {
						// Устанавливаем ошибку внешнего питания
						globalErrors.error_ExtPowerError = true;
					}
				} else {
					// Реверс полярности управления главным питанием моторов НУЖЕН
					globalFlags.flag_isNeedRevPolMainMotorPwr = true;
					// Выключаем общее питание актуаторов, включив реле
					actuators_main_power_off(globalFlags.flag_isNeedRevPolMainMotorPwr);
				}

				if (reserve_volt != 0 ) {
					// Устанавливаем флаг наличия внешнего резервного питания
					globalFlags.flag_isReservePowerPresent = true;
					if ( (reserve_volt < MIN_EXT_VOLTAGE_V) || (reserve_volt > MAX_EXT_VOLTAGE_V) ) {
						// Устанавливаем ошибку внешнего питания
						globalErrors.error_ExtPowerError = true;
					}
				}

				loop_state = READY;

				// Освобождаем доступ к ADC, возвращая mutex
				osMutexRelease(adcUse_MutexHandle);

				#ifdef ON_DEBUG_MESSAGE
					HAL_UART_Transmit(&huart1, "MUTEX-Release [voltage_loop]\r\n", strlen("MUTEX-Release [voltage_loop]\r\n"), 500);
				#endif
			}
			break;
		default:
			break;
	}
	return loop_state;
}

//===================================================================================

// Callback
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
	if (hadc->Instance == ADC1) {
		globalFlags.flag_isAdcСonvReady = true;
	}
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	switch (GPIO_Pin)
	{
		case UP_BUTTON_PIN:
		{
			globalFlags.flag_isUpButtonPressed = true;
			break;
		}
		case DOWN_BUTTON_PIN:
		{
			globalFlags.flag_isDownButtonPressed = true;
			break;
		}
		case STOP_BUTTON_PIN:
		{
			globalFlags.flag_isStopButtonPressed = true;
			break;
		}
		default:

			break;
	}
}

void main_task_upMotor_timerCallback(void const * argument)
{
	// Останавливаем таймер
	osTimerStop(upMotor_TimerHandle);

	// Устанавливаем флаг ошибки
	globalErrors.error_UpMotorTimeoutError = true;

	// Устанавливаем флаг необходимости остановки мотора
	globalFlags.flag_isNeedStopUpMotor = true;

	#ifdef ON_DEBUG_MESSAGE
		HAL_UART_Transmit(&huart1, "UP_MOTOR timeout callback\r\n", strlen("UP_MOTOR timeout callback\r\n"), 500);
	#endif
}

void main_task_downMotor_timerCallback(void const * argument)
{
	// Останавливаем таймер
	osTimerStop(downMotor_TimerHandle);

	// Устанавливаем флаг ошибки
	globalErrors.error_DownMotorTimeoutError = true;

	// Устанавливаем флаг необходимости остановки мотора
	globalFlags.flag_isNeedStopDownMotor = true;

	#ifdef ON_DEBUG_MESSAGE
		HAL_UART_Transmit(&huart1, "DOWN_MOTOR timeout callback\r\n", strlen("DOWN_MOTOR timeout callback\r\n"), 500);
	#endif

}

void main_task_sideMotor_timerCallback(void const * argument)
{
	// Останавливаем таймер
	osTimerStop(sideMotor_TimerHandle);

	// Устанавливаем флаг ошибки
	globalErrors.error_SideMotorTimeoutError = true;

	// Устанавливаем флаг необходимости остановки мотора
	globalFlags.flag_isNeedStopSideMotor = true;

	#ifdef ON_DEBUG_MESSAGE
		HAL_UART_Transmit(&huart1, "SIDE_MOTOR timeout callback\r\n", strlen("SIDE_MOTOR timeout callback\r\n"), 500);
	#endif
}

//===================================================================================
