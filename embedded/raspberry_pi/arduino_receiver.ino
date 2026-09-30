/*
 * ELP-720 Telecommunication Networks Laboratory - Assignment 8
 * Arduino Serial Receiver for Raspberry Pi Communication
 *
 * Author: Sudhanshu Chaudhary (2019JTM2207)
 * Institution: IIT Delhi
 * Board: Arduino Uno / Nano
 *
 * Functionality:
 *  - Listens on Hardware Serial at 9600 baud.
 *  - Buffers received bytes transmitted by Raspberry Pi.
 *  - Displays received characters and strings on Serial Monitor / connected LCD.
 */

const int BUFFER_MAX = 128;
char rxBuffer[BUFFER_MAX];
int rxIndex = 0;

void setup() {
    Serial.begin(9600);
    while (!Serial) {
        ; // Wait for serial port connection
    }
    Serial.println("=========================================");
    Serial.println(" Arduino Serial Receiver Initialized");
    Serial.println(" Listening for messages from Raspberry Pi...");
    Serial.println("=========================================");
}

void loop() {
    if (Serial.available() > 0) {
        char incomingChar = Serial.read();

        // Print character immediately
        Serial.print(incomingChar);

        if (incomingChar == '\n' || incomingChar == '\r' || rxIndex >= BUFFER_MAX - 1) {
            if (rxIndex > 0) {
                rxBuffer[rxIndex] = '\0';
                Serial.println();
                Serial.print("[Arduino] Complete Message Received: \"");
                Serial.print(rxBuffer);
                Serial.println("\"");
                rxIndex = 0;
            }
        } else {
            rxBuffer[rxIndex++] = incomingChar;
        }
    }
}
