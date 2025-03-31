/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  * @by				: Alberto Lopes
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc;

TIM_HandleTypeDef htim2;
DMA_HandleTypeDef hdma_tim2_ch1;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_tx;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_ADC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
uint8_t angle = 0;
static uint8_t Heures_brt = 0;
static uint8_t Heures_U = 0;
static uint8_t Heures_D = 0;
static uint8_t Minutes_brt = 0;
static uint8_t Minutes_U = 0;
static uint8_t Minutes_D = 0;
uint8_t Cli = 0;

uint8_t interrupteur1_OLD = 0;
uint8_t interrupteur2_OLD = 0;
uint8_t interrupteur3_OLD = 0;
uint8_t interrupteur4_OLD = 0;

static uint8_t Rx_data[19];

uint16_t step = 0;

uint16_t ReadADC = 0;

bool needMeasure=false;

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  MX_ADC_Init();
  /* USER CODE BEGIN 2 */
  ws2812_start();
  HAL_UART_Receive_IT(&huart1, Rx_data, 19);
  //HAL_ADC_Calibration_Start(&hadc);



  uint8_t H =0;
  uint8_t facteurLuminosite=255;


  // Déclarez une instance de Canvas
  Canvas myCanvas;
  Canvas blackCanvas;
  // Initialisez la structure Canvas
  myCanvas.numCols = NUM_COLS;
  myCanvas.numRows = NUM_ROWS;
  // Allouez de la mémoire pour les pixels
  myCanvas.pixels = malloc(sizeof(Pixel) * NUM_COLS * NUM_ROWS);
  // Utilisez memset pour initialiser le tableau à zéro
  memset(myCanvas.pixels, 0, sizeof(Pixel) * NUM_COLS * NUM_ROWS);


  blackCanvas.numCols = NUM_COLS;
    blackCanvas.numRows = NUM_ROWS;
    // Allouez de la mémoire pour les pixels
    blackCanvas.pixels = malloc(sizeof(Pixel) * NUM_COLS * NUM_ROWS);
    // Utilisez memset pour initialiser le tableau à zéro
    memset(blackCanvas.pixels, 0, sizeof(Pixel) * NUM_COLS * NUM_ROWS);
  // Vous pouvez maintenant utiliser myCanvas et les pixels initialisé
  setCanvasColor(&blackCanvas, (Pixel){0,0,0});
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
//	  HAL_ADC_PollForConversion(&hadc, HAL_MAX_DELAY);
//
//	  ReadADC = HAL_ADC_GetValue(&hadc);



	  if(HAL_UART_Receive_IT(&huart1, Rx_data, 19) != HAL_BUSY)
	  {
		  HAL_UART_Receive_IT(&huart1, Rx_data, 19);
	  }

	  /**********Measure***************/
	  if(needMeasure){
		  facteurLuminosite = flashReadADC(&myCanvas) + LUM_CAL_OFFSET;
		  needMeasure = false;
	  }

	  /**********Background***************/

	  for(uint8_t diag=1; diag<=23; diag++){
		  colorDiagonal(&myCanvas, HSVtoPixel((RB_SPEED*H + (diag* 255 / 23))%255 , (facteurLuminosite*RB_MAX_LUX)/255), diag);
	  }

	  displayBCD(&myCanvas, 2, 3, Heures_D, 2, facteurLuminosite);
	  displayBCD(&myCanvas, 5, 3, Heures_U, 4, facteurLuminosite);
	  displayBCD(&myCanvas, 10, 3, Minutes_D, 4, facteurLuminosite);
	  displayBCD(&myCanvas, 15, 3, Minutes_U, 4, facteurLuminosite);

	  sendCanvas(&myCanvas);

	  H++;
	  if(!((RB_SPEED*H)%255)){
		  H=0;
	  }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_HSI14;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSI14State = RCC_HSI14_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.HSI14CalibrationValue = 16;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC_Init(void)
{

  /* USER CODE BEGIN ADC_Init 0 */

  /* USER CODE END ADC_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC_Init 1 */

  /* USER CODE END ADC_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc.Instance = ADC1;
  hadc.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc.Init.Resolution = ADC_RESOLUTION_8B;
  hadc.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc.Init.ScanConvMode = ADC_SCAN_DIRECTION_FORWARD;
  hadc.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc.Init.LowPowerAutoWait = DISABLE;
  hadc.Init.LowPowerAutoPowerOff = DISABLE;
  hadc.Init.ContinuousConvMode = DISABLE;
  hadc.Init.DiscontinuousConvMode = DISABLE;
  hadc.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc.Init.DMAContinuousRequests = DISABLE;
  hadc.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  if (HAL_ADC_Init(&hadc) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel to be converted.
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = ADC_RANK_CHANNEL_NUMBER;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC_Init 2 */

  /* USER CODE END ADC_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 10-1;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 250000;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel2_3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel2_3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_3_IRQn);
  /* DMA1_Channel4_5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel4_5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel4_5_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pins : SW4_Pin SW3_Pin */
  GPIO_InitStruct.Pin = SW4_Pin|SW3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : SW1_Pin SW2_Pin */
  GPIO_InitStruct.Pin = SW1_Pin|SW2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/******************************************************************************
* Fonction 	HAL_UART_RxCpltCallback											  *
*		Prototype	:void HAL_UART_RxCpltCallback (UART_HandleTypeDef * huart)*
*																		 	  *
*   Description des paramètres:												  *
*			UART qu'on utilise												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Interruption qui va mettre dans des variables la réception UART comme le	  *
* temps										 								  *
******************************************************************************/

void HAL_UART_RxCpltCallback (UART_HandleTypeDef * huart)
{

	if(&huart1 == huart)
	{
		Heures_brt = Rx_data[4];	//Stock le data des heures, "brt" = brute
		Minutes_brt = Rx_data[5];	//Stock le data des minutes

		Heures_U = Heures_brt & 0x0F;	//Traite le data pour avoir l'Unité des Heures

		Heures_D = (Heures_brt & 0xF0) >> 4;	//Traite le data pour avoir la Dizaine des Heures

		Minutes_U = Minutes_brt & 0x0F;	//Traite le data pour avoir l'Unité des Minutes

		Minutes_D = (Minutes_brt & 0xF0) >> 4;	//Traite le data pour avoir la Dizaine des Minutes

		needMeasure = true;

		if(HAL_UART_Receive_IT(&huart1, Rx_data, 19) == HAL_ERROR)	//Réception d'UART lors d'une erreur
		{
			HAL_UART_Receive_IT(&huart1, Rx_data, 19);
		}
	}
}

/******************************************************************************
* Fonction 	HAL_UART_ErrorCallback											  *
*		Prototype	:void HAL_UART_RxCpltCallback (UART_HandleTypeDef * huart)*
*																		 	  *
*   Description des paramètres:												  *
*			UART qu'on utilise												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Interruption lorsqu'il y a un problème dans la transmission UART qui permet *
* de refaire une réception									 				  *
******************************************************************************/

void HAL_UART_ErrorCallback (UART_HandleTypeDef * huart)
{
	if(&huart1 == huart)
	{
		HAL_UART_Receive_IT(&huart1, Rx_data, 19);
	}
}

/******************************************************************************
* Fonction 	LEDs_L_C_RGB													  *
*																		 	  *
*   Description des paramètres:												  *
*			Ligne allant de 0 à 4, Colonne de 0 à 18, Couleur RGB de 0 à 255  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Allume la LED qu'on veut grâce aux paramètres "Ligne" et "Colonne".		  *
* Choisi l'intensité des couleurs RGB									 	  *
******************************************************************************/

void LEDs_L_C_RGB (uint8_t Ligne, uint8_t Colonne, uint8_t Red, uint8_t Green, uint8_t Blue)
{
	uint8_t led_l_c[5][19] = {
			{4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94},	//La 94ème LED n'existe pas sur l'hardware
			{3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93},
			{2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92},
			{1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91},
			{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90}
	};


	setLEDcolor(led_l_c[Ligne][Colonne], Red, Green, Blue);
}

/******************************************************************************
* Fonction 	Chara_0															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "0" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_0 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	/*Décalage du caractère pour le placer dans la colonne des unité/dizaine ; Heure/Minute*/

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	/*Création du caractère 0 sur la matrice*/

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_1															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "1" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_1 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_2															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "2" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_2 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_3															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "3" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_3 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_4															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "4" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_4 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 0, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 0, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_5															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "5" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_5 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_6															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "6" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_6 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_7															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "7" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_7 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 0, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 0, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_8															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "8" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_8 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Chara_9															  *
*																		 	  *
*   Description des paramètres:												  *
*			Shift peut contenir ces valeurs :								  *
*				- MinuteUnite												  *
*				- MinuteDizaine												  *
*				- HeureUnite												  *
*				- HeureDizaine												  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Dessine la caractère "9" sur la matrice et la positionne aux emplacements	  *
* prédéfini des Heures(unité ou dizaine) et des Minutes(unité ou dizaine)	  *
******************************************************************************/

void Chara_9 (CharShiftTypeDef Shift)
{
	static uint8_t shiftChar;

	switch (Shift)
	{
	case MinuteUnite:
		shiftChar = 14;
		break;

	case MinuteDizaine:
		shiftChar = 10;
		break;

	case HeureUnite:
		shiftChar = 4;
		break;

	case HeureDizaine:
		shiftChar = 0;
		break;

	default:
		break;
	}

	LEDs_L_C_RGB(0, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 1+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 1+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 1+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(2, 2+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 2+shiftChar, 0, 0, 0);
	LEDs_L_C_RGB(4, 2+shiftChar, 10, 0, 0);

	LEDs_L_C_RGB(0, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(1, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(2, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(3, 3+shiftChar, 10, 0, 0);
	LEDs_L_C_RGB(4, 3+shiftChar, 10, 0, 0);
}

/******************************************************************************
* Fonction 	Clignotement1S													  *
*																		 	  *
*   Description des paramètres:												  *
*			Aucun															  *
*																			  *
*		Inclus :	main.c													  *
*																			  *
*	  Description															  *
* Fait clignoter les deux LEDs qui sépare les heures des minutes chaque		  *
* seconde																	  *
******************************************************************************/

void Clignotement1S (void)
{
	static uint16_t ComptX1000 = 0;

	/*Compte une seconde éteint et allumé*/

	if (ComptX1000 <= 42)
	{
		LEDs_L_C_RGB(1, 9, 10, 0, 0);
		LEDs_L_C_RGB(3, 9, 10, 0, 0);
	}
	else
	{
		LEDs_L_C_RGB(1, 9, 0, 0, 0);
		LEDs_L_C_RGB(3, 9, 0, 0, 0);
	}

	ComptX1000++;

	if(ComptX1000 >= 84) //Remise à zéro du compteur
	{
		ComptX1000 = 0;
	}
}

//uint8_t SetBrightness (uint16_t ValLum)
//{
//	uint8_t ChangeLum;
//
//
//}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
