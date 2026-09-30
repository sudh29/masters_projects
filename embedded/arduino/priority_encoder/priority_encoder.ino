/*
 * ELP-720 Telecommunication Networks Laboratory - Assignment 2
 * Problem Statement 2: 4-to-2 Hardware Priority Encoder with SD Card Truth Table Logging
 *
 * Author: Sudhanshu Chaudhary (2019JTM2207)
 * Institution: IIT Delhi
 * Platform: Arduino Uno / Hardware Kit
 *
 * Pin Connections:
 *  - Enable Switch: Pin 3 (HIGH enables encoder)
 *  - Inputs:
 *      Pin 7: Input D3 (Highest Priority)
 *      Pin 6: Input D2
 *      Pin 5: Input D1
 *      Pin 4: Input D0 (Lowest Priority)
 *  - Outputs:
 *      Pin 8: Y0 (LSB)
 *      Pin 9: Y1 (MSB)
 */

int y0 = 0;
int y1 = 0;

const int ENABLE_PIN = 3;
const int INP_D3 = 7;
const int INP_D2 = 6;
const int INP_D1 = 5;
const int INP_D0 = 4;
const int OUT_Y0 = 8;
const int OUT_Y1 = 9;

void setup() {
    pinMode(ENABLE_PIN, INPUT);
    pinMode(INP_D3, INPUT);
    pinMode(INP_D2, INPUT);
    pinMode(INP_D1, INPUT);
    pinMode(INP_D0, INPUT);

    pinMode(OUT_Y0, OUTPUT);
    pinMode(OUT_Y1, OUTPUT);

    Serial.begin(9600);
    Serial.println("=================================================");
    Serial.println(" 4-to-2 Priority Encoder Initialized");
    Serial.println(" Format: Enable | D3 D2 D1 D0 | Y1 Y0");
    Serial.println("=================================================");
}

void loop() {
    int enable = digitalRead(ENABLE_PIN);

    if (enable == LOW) {
        // System disabled: all outputs low
        y1 = 0;
        y0 = 0;
    } else {
        int val3 = digitalRead(INP_D3); // High priority
        int val2 = digitalRead(INP_D2);
        int val1 = digitalRead(INP_D1);
        int val0 = digitalRead(INP_D0); // Low priority

        if (val3 == HIGH) {
            y1 = 1;
            y0 = 1;
        } else if (val2 == HIGH) {
            y1 = 1;
            y0 = 0;
        } else if (val1 == HIGH) {
            y1 = 0;
            y0 = 1;
        } else if (val0 == HIGH) {
            y1 = 0;
            y0 = 0;
        } else {
            y1 = 0;
            y0 = 0;
        }

        // Print logged truth table row over Serial (or write to SD card file)
        Serial.print("   1    |  ");
        Serial.print(val3); Serial.print("  ");
        Serial.print(val2); Serial.print("  ");
        Serial.print(val1); Serial.print("  ");
        Serial.print(val0); Serial.print("  |  ");
        Serial.print(y1); Serial.print("  ");
        Serial.println(y0);
    }

    digitalWrite(OUT_Y0, y0);
    digitalWrite(OUT_Y1, y1);

    delay(500);
}
