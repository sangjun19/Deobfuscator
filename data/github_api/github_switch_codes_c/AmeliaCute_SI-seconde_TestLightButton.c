/**
 * @file TestLightButton.c
 * @author Ulysse (github.com/MignonPetitXelow)
 * @brief 
 * @version 0.1
 * @date 2022-10-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

bool light = false;

void setup(void)
{
  Serial.begin(9600);
}

void loop()
{
    int count = 0;
    if(digitalRead(4) == HIGH)
    {
        ++count; 
        Serial.println(count+" - Button has been pressed");
        setLedBuiltin();
    }
}

void setLedBuiltin()
{
    pinMode(LED_BUILTIN, OUTPUT);
    switch(light) // Optimiser cette chose car c'est pas opti.
    {
        case true:
            digitalWrite(LED_BUILTIN, HIGH);
            break;
        case false: 
            digitalWrite(LED_BUILTIN, LOW);
            break;
    }
}